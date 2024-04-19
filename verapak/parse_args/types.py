import os
import pathlib
import numpy as np
from verapak.parse_args import strategy_registry

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
    return np.array([float(x.strip()) for x in l])

def xNumArrayType(string):
    if "x" not in string:
        return numArrayType(string)
    else:
        return np.array([x.strip() for x in string.split(',')], dtype=np.string_)

def type_string_to_type(type_):
    if type_ == "int":
        return int
    elif type_ == "float":
        return float
    elif type_ == "directory":
        return directoryType
    elif type_ == "file":
        return fileType
    elif type_ == "graph_path":
        return graphPathType
    elif type_ == "vnn_path":
        return vnnPathType
    elif type_ == "num_array":
        return numArrayType
    elif type_ == "x_num_array":
        return xNumArrayType
    elif hasattr(type_, "__call__"):
        return type_
    elif type_.startswith("strategy:"):
        category = type_.split(":")[1]
        return strategyType(category)
    else:
        raise ValueError(f"Bad arg type {type_}")
