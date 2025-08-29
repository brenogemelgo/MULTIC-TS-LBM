from pyevtk.hl import gridToVTK
from getSimInfo import *

def saveVTK3D(macrosDict, path, filenameWrite, points=True, normVal=1):

    info = retrieveSimInfo(path)

    if normVal == 0:
        normVal = info['NX']
        if points:
            normVal -= 1

    dx, dy, dz = 1.0 / normVal, 1.0 / normVal, 1.0 / normVal

    if not points:
        x = np.arange(0, info['NX'] / normVal + 0.1 * dx, dx, dtype='float32')
        y = np.arange(0, info['NY'] / normVal + 0.1 * dy, dy, dtype='float32')
        z = np.arange(0, info['NZ'] / normVal + 0.1 * dz, dz, dtype='float32')
        gridToVTK(path + filenameWrite, x, y, z, cellData=macrosDict)
    else:
        x = np.arange(0, (info['NX'] - 1) / normVal + 0.1 * dx, dx, dtype='float32')
        y = np.arange(0, (info['NY'] - 1) / normVal + 0.1 * dy, dy, dtype='float32')
        z = np.arange(0, (info['NZ'] - 1) / normVal + 0.1 * dz, dz, dtype='float32')
        gridToVTK(path + filenameWrite, x, y, z, pointData=macrosDict)
