import os
import pathlib
import numpy as np
from verapak.parse_args import strategy_registry
from config import ConfigValueError

def strategyType(category):
    def typeConverter(strategy):
        options = get_strategy_choices(category)
        if strategy in options:
            return strategy_registry.ALL_STRATEGIES[category][strategy]
        raise ConfigValueError(f"unsupported {category} strategy: {strategy}")
    return typeConverter
def get_strategy_choices(category):
    if category in strategy_registry.ALL_STRATEGIES:
        return list(strategy_registry.ALL_STRATEGIES[category].keys())
    raise ConfigValueError("unsupported strategy category: {}".format(category))


def directoryType(dirPath):
    if os.path.isdir(dirPath):
        return dirPath
    raise ConfigValueError("not a valid directory", path=dirPath)


def fileType(filePath):
    if os.path.isfile(filePath):
        return filePath
    raise ConfigValueError("not a valid file", path=filePath)


def graphPathType(filePath):
    filePath = fileType(filePath)
    if pathlib.Path(filePath).suffix.upper() == '.ONNX':
        return filePath
    raise ConfigValueError("unsupported model type", path=filePath)

def vnnPathType(filePath):
    filePath = fileType(filePath)
    if pathlib.Path(filePath).suffix.upper() == '.VNNLIB':
        return filePath
    raise ConfigValueError("unsupported vnnlib file", path=filePath)


def numArrayType(string):
    try:
        l = string.split(',')
        return np.array([float(x.strip()) for x in l])
    except ValueError as ex:
        raise ConfigValueError(ex.args[0])

def _numArrayFromXNumArray(radius, xnum):
    to_multiply = np.vectorize(lambda x: x.endswith(b'x'))(xnum)
    v = np.where(to_multiply, 0, xnum).astype(np.float64)
    mult = np.vectorize(lambda s: s[:-1])(np.where(to_multiply, xnum, 0)).astype(np.float64)
    return (mult * radius) + np.broadcast_to(v, radius.shape)

def xNumArrayType(string):
    try:
        if "x" not in string:
            return lambda _: numArrayType(string)
        else:
            xnum = np.array([x.strip() for x in string.split(',')], dtype=np.string_)
            return lambda radius: _numArrayFromXNumArray(radius, xnum)
    except ValueError as ex:
        raise ConfigValueError(ex.args[0])

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

