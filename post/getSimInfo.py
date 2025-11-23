import os
import glob
import re
import numpy as np

__macroNames__ = None
__info__ = dict()

_macro_re = re.compile(r'^([A-Za-z_]+)(\d+)\.bin$')

def discover_macro_names(path):
    global __macroNames__
    if __macroNames__ is not None:
        return __macroNames__

    files = glob.glob(os.path.join(path, "*.bin"))
    macros = set()

    for f in files:
        base = os.path.basename(f)
        m = _macro_re.match(base)
        if m:
            macros.add(m.group(1))

    __macroNames__ = sorted(macros)
    return __macroNames__

def getFilenamesMacro(macr_name, path):
    pattern = os.path.join(path, f"{macr_name}*.bin")
    fileList = sorted(glob.glob(pattern))
    return fileList

def getMacroSteps(path):
    macroNames = discover_macro_names(path)
    if not macroNames:
        return []

    ref_macro = macroNames[0]
    fileList = getFilenamesMacro(ref_macro, path)

    stepSet = set()
    for file in fileList:
        base = os.path.basename(file)
        m = _macro_re.match(base)
        if m:
            stepSet.add(int(m.group(2)))  

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
    with open(macr_filename, "rb") as f:
        vec = np.fromfile(f, 'f')  
    vec3D = np.reshape(vec, (info['NZ'], info['NY'], info['NX']), 'C')
    return np.swapaxes(vec3D, 0, 2)

def getMacrosFromStep(step, path):
    macroNames = discover_macro_names(path)
    macr = dict()

    stepFilenames = []
    for macr_name in macroNames:
        pattern = os.path.join(path, f"{macr_name}{step:06d}.bin")
        stepFilenames.extend(glob.glob(pattern))

    if len(stepFilenames) == 0:
        return None

    for filename in stepFilenames:
        base = os.path.basename(filename)
        m = _macro_re.match(base)
        if not m:
            continue
        macr_name = m.group(1)
        macr[macr_name] = readFileMacro3D(filename, path)

    return macr

def getAllMacros(path):
    macroNames = discover_macro_names(path)
    macr = dict()
    filenames = dict()

    for macr_name in macroNames:
        filenames[macr_name] = getFilenamesMacro(macr_name, path)

    if not filenames:
        return macr

    minLength = min(len(filenames[key]) for key in filenames)

    for i in range(minLength):
        base = os.path.basename(filenames[macroNames[0]][i])
        m = _macro_re.match(base)
        if not m:
            continue
        step = int(m.group(2))

        macr[step] = dict()
        for macr_name in macroNames:
            macr[step][macr_name] = readFileMacro3D(
                filenames[macr_name][i], path
            )

    return macr
