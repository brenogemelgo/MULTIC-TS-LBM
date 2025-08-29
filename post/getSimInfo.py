import os
import glob
import re
import numpy as np

__macroNames__ = ['rho','phi','ux','uy','uz','vorticity_mag','velocity_mag']
__info__ = dict()

def getFilenamesMacro(macr_name, path):
    fileList = sorted(glob.glob(path + "*" + macr_name + "*.bin"))
    return fileList

def getMacroSteps(path):
    fileList = getFilenamesMacro(__macroNames__[0], path)
    stepSet = set()
    for file in fileList:
        stepStr = file.split(__macroNames__[0])[-1]
        stepStr = stepStr[:-4] 
        stepInt = int(stepStr)
        stepSet.add(stepInt)
    macroSteps = sorted(stepSet)
    return macroSteps

def retrieveSimInfo(path):
    global __info__
    if __info__:
        return __info__

    files = sorted(glob.glob(os.path.join(path, "*info*.txt")))
    if not files:
        print("No *info*.txt files were found in directory:", path)
        return __info__

    filename = files[0]
    try:
        with open(filename, "r") as f:
            text = f.read()
    except Exception as e:
        print("Failed to open file:", filename, "-", e)
        return __info__

    def grab(pattern, flags=re.M):
        m = re.search(pattern, text, flags)
        return m.group(1) if m else None

    _id = grab(r"^ID:\s*(\S+)")
    if _id is None:
        print("Not able to get 'ID' from info file")
    else:
        __info__['ID'] = _id

    steps = grab(r"^Timesteps:\s*(\d+)")
    if steps is None:
        steps = grab(r"^Total steps:\s*(\d+)")
    if steps is None:
        print("Not able to get 'Timesteps' from info file")
    else:
        __info__['Timesteps'] = int(steps)

    m = re.search(r"^Domain size:\s*NX\s*=\s*(\d+)\s*,\s*NY\s*=\s*(\d+)\s*,\s*NZ\s*=\s*(\d+)",
                  text, re.M)
    if m:
        __info__['NX'] = int(m.group(1))
        __info__['NY'] = int(m.group(2))
        __info__['NZ'] = int(m.group(3))
    else:
        nx = grab(r"^NX:\s*(\d+)")
        ny = grab(r"^NY:\s*(\d+)")
        nz = grab(r"^NZ:\s*(\d+)")
        if nx is None or ny is None or nz is None:
            print("Not able to get 'NX/NY/NZ' from info file")
        else:
            __info__['NX'] = int(nx)
            __info__['NY'] = int(ny)
            __info__['NZ'] = int(nz)

    return __info__

def readFileMacro3D(macr_filename, path):
    info = retrieveSimInfo(path)
    with open(macr_filename, "r") as f:
        vec = np.fromfile(f, 'f') # float32
        vec3D = np.reshape(vec, (info['NZ'], info['NY'], info['NX']), 'C')
        return np.swapaxes(vec3D, 0, 2)

def getMacrosFromStep(step, path):
    macr = dict()
    allFilenames = []

    for macr_name in __macroNames__:
        allFilenames.append(getFilenamesMacro(macr_name, path))

    flatFilenames = [filename for sublist in allFilenames for filename in sublist]

    stepFilenames = [
        filename for filename in flatFilenames
        if any([f"{macr}{step:06d}.bin" in filename for macr in __macroNames__])
    ]

    if len(stepFilenames) == 0:
        return None

    for filename in stepFilenames:
        for macr_name in __macroNames__:
            if macr_name in filename:
                macr[macr_name] = readFileMacro3D(filename, path)

    return macr

def getAllMacros(path):
    macr = dict()
    filenames = dict()

    for macr_name in __macroNames__:
        filenames[macr_name] = getFilenamesMacro(macr_name, path)

    minLength = min(len(filenames[key]) for key in filenames)

    for i in range(minLength):
        stepStr = filenames[__macroNames__[0]][i].split(__macroNames__[0])[-1]
        stepStr = stepStr[:-4]
        step = int(stepStr)

        macr[step] = dict()
        for macr_name in filenames:
            macr[step][macr_name] = readFileMacro3D(filenames[macr_name][i], path)

    return macr
