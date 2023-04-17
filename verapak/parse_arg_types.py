import os
import pathlib
from . import strategy_registry

def strategyType(category):
    def typeConverter(strategy):
        options = get_strategy_choices(category)
        if strategy in options:
            return strategy_registry.ALL_STRATEGIES[category][strategy]
        raise ValueError(f"unsupported {category} strategy: {strategy}")
    return typeConverter
def get_strategy_choices(category):
    if category in strategy_registry.ALL_STRATEGIES:
        return list(strategy_registry.ALL_STRATEGIES[category].keys())
    raise ValueError("unsupported strategy category: {}".format(category))


def directoryType(dirPath):
    if os.path.isdir(dirPath):
        return dirPath
    raise ValueError(f"{dirPath} is not a directory")


def fileType(filePath):
    if os.path.isfile(filePath):
        return filePath
    raise ValueError(f"{filePath} is not a file")


def graphPathType(filePath):
    filePath = fileType(filePath)
    if pathlib.Path(filePath).suffix.upper() == '.ONNX':
        return filePath
    raise ValueError(f"{filePath} unsupported model type")

def vnnPathType(filePath):
    filePath = fileType(filePath)
    if pathlib.Path(filePath).suffix.upper() == '.VNNLIB':
        return filePath
    raise ValueError(f"{filePath} unsupported vnnlib file")


def numArrayType(string):
    string = string
    l = string.split(',')
    return [float(x.strip()) for x in l]

def xNumArrayType(string):
    if "x" not in string:
        return numArrayType(string)
    else:
        return [x.strip() for x in string.split(',')]

