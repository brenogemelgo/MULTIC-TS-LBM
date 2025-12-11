/*---------------------------------------------------------------------------*\
|                                                                             |
| MULTIC-TS-LBM: CUDA-based multicomponent Lattice Boltzmann Method           |
| Developed at UDESC - State University of Santa Catarina                     |
| Website: https://www.udesc.br                                               |
| Github: https://github.com/brenogemelgo/MULTIC-TS-LBM                       |
|                                                                             |
\*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*\

Copyright (C) 2023 UDESC Geoenergia Lab
Authors: Breno Gemelgo (Geoenergia Lab, UDESC)

License
    This file is part of MULTIC-TS-LBM.

    MULTIC-TS-LBM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

Description
    Thread pool class declaration

SourceFiles
    threadPool.cuh

\*---------------------------------------------------------------------------*/

#include "cuda/utils.cuh"

namespace thread
{
    class Pool
    {
    public:
        explicit Pool(size_t n) : stop(false)
        {
            for (size_t i = 0; i < n; ++i)
            {
                workers.emplace_back([this]
                                     {
                for (;;) {
                    std::function<void()> task;

                    {
                        std::unique_lock<std::mutex> lock(this->mtx);
                        this->cv.wait(lock, [this] {
                            return this->stop || !this->tasks.empty();
                        });
                        if (this->stop && this->tasks.empty())
                            return;
                        task = std::move(this->tasks.front());
                        this->tasks.pop();
                    }

                    task();
                } });
            }
        }

        template <class F>
        void enqueue(F &&f)
        {
            {
                std::unique_lock<std::mutex> lock(mtx);
                tasks.emplace(std::forward<F>(f));
            }
            cv.notify_one();
        }

        ~Pool()
        {
            {
                std::unique_lock<std::mutex> lock(mtx);
                stop = true;
            }
            cv.notify_all();
            for (std::thread &w : workers)
                w.join();
        }

    private:
        std::vector<std::thread> workers;
        std::queue<std::function<void()>> tasks;
        std::mutex mtx;
        std::condition_variable cv;
        bool stop;
    };
}
