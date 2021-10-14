import argparse  # https://docs.python.org/3/library/argparse.htm
import sys
import os
from . import strategy_registry
import pathlib
from .utilities.vnnlib_lib import VNNLib


def get_strategy_choices(category):
    if category in strategy_registry.ALL_STRATEGIES:
        return list(strategy_registry.ALL_STRATEGIES[category].keys())
    raise ValueError("unsupported strategy category: {}".format(category))


def strategyType(category):
    def typeConverter(strategy):
        options = get_strategy_choices(category)
        if strategy in options:
            return strategy_registry.ALL_STRATEGIES[category][strategy]
        raise ValueError(f"unsupported {category} strategy: {strategy}")
    return typeConverter


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


def numArrayType(string):
    string = string.strip()
    l = string.split(',')
    return [float(x) for x in l]


SUPPORTED_ARGUMENTS = [
    {
        'name': 'config_file',
        'arg_params':
            {
                'type': fileType,
                'help': 'Configuration file: key value pairs (one per line) in this format KEY :: VALUE. Keys are the same as command line argument names in this help and values follow the same format.',
                'default': None
            }
    },
    {
        'name': 'vnnlib',
        'arg_params':
            {
                'type': fileType,
                'help': 'VNNLIB file: parse out the centerpoint, intended label, and radii, and use them if no others are provided in the config_file or command line flags.',
                'default': None
            }
    },
    {
        'name': 'output_dir',
        'arg_params':
            {
                'type': directoryType,
                'help': 'path to output directory where adversarial examples will be stored',
                'default': '.'
            }
    },
    {
        'name': 'initial_point',
        'arg_params':
            {
                'type': numArrayType,
                'help': 'Point around which robustness verification will occur',
            }
    },
    {
        'name': 'label',
        'arg_params':
            {
                'type': int,
                'help': 'Intended class label number (use index of logit). If not provided, class of Point is assumed as the intended class',
            }
    },
    {
        'name': 'graph',
        'arg_params':
            {
                'type': graphPathType,
                'help': 'path to serialized DNN model (ONNX, TF, etc)',
            }
    },
    {
        'name': 'radius',
        'arg_params':
            {
                'type': numArrayType,
                'help': 'Radius (single value or per dimension array) around initial point where robustness verification will occur',
            }
    },
    {
        'name': 'granularity',
        'arg_params':
            {
                'type': numArrayType,
                'help': 'Granularity (single value or per dimension array): a valid discretization of the input space (8 bit image -> 1/256)'
            }
    },
    {
        'name': 'num_abstractions',
        'arg_params':
            {
                'type': int,
                'help': 'Number of abstraction points to generate each pass',
                'default': 10
            }
    },
    {
        'name': 'balance_factor',
        'arg_params':
            {
                'type': float,
                'help': 'RFGSM balance factor (1.0 -> all FGSM, 0.0 -> random manipulations)',
                'default': 0.95
            }
    },
    {
        'name': 'verification_strategy',
        'arg_params':
            {
                'type': strategyType('verification'),
                'help': "Verification strategy: (oneof {})".format(get_strategy_choices('verification')),
                'default': strategy_registry.VERIFICATION_STRATEGIES['discrete_search']
            }
    },
    {
        'name': 'dimension_ranking_strategy',
        'arg_params':
            {
                'type': strategyType('dimension_ranking'),
                'help': "RFGSM dimension ranking strategy: (oneof {})".format(get_strategy_choices('dimension_ranking')),
                'default': strategy_registry.DIMENSION_RANKING_STRATEGIES['gradient_based']
            }
    },
    {
        'name': 'abstraction_strategy',
        'arg_params':
            {
                'type': strategyType('abstraction'),
                'help': "Abstraction strategy: (oneof {})".format(get_strategy_choices('abstraction')),
                'default': strategy_registry.ABSTRACTION_STRATEGIES['rfgsm']
            }
    },
    {
        'name': 'partitioning_strategy',
        'arg_params':
            {
                'type': strategyType('partitioning'),
                'help': "Partitioning strategy: (oneof {})".format(get_strategy_choices('partitioning')),
                'default': strategy_registry.PARTITIONING_STRATEGIES['largest_first']
            }
    },
    {
        'name': 'partitioning_divisor',
        'arg_params':
            {
                'type': int,
                'help': "Number of divisions on each dimension during partitioning",
                'default': 2
            }
    },
    {
        'name': 'partitioning_num_dimensions',
        'arg_params':
            {
                'type': int,
                'help': "Number of dimensions to partition",
                'default': 3
            }
    },
    {
        'name': 'verification_point_threshold',
        'arg_params':
            {
                'type': int,
                'help': "Threshold number of discrete points under which verification should occur",
                'default': 10000
            }
    },
    {
        'name': 'domain_lower_bound',
        'arg_params':
            {
                'type': numArrayType,
                'help': "Threshold number of discrete points under which verification should occur",
                'default': [0.0],
            }
    },
    {
        'name': 'domain_upper_bound',
        'arg_params':
            {
                'type': numArrayType,
                'help': "Threshold number of discrete points under which verification should occur",
                'default': [1.0],
            }
    },
    {
        'name': 'timeout_minutes',
        'arg_params':
            {
                'type': int,
                'help': "Number of minutes to run the program before reporting all found adversarial examples and timing out (set to 0 for 'run until interrupted')",
                'default': 5,
            }
    },
    {
        'name': 'report_interval_seconds',
        'arg_params':
            {
                'type': int,
                'help': "Number of seconds between status reports",
                'default': 2,
            }
    }]


def parse_cmdline_args(args, prog):
    parser = argparse.ArgumentParser(
        description="VERAPAK: framework for verifying adversarial robustness of DNNs", prog=prog)
    for arg in SUPPORTED_ARGUMENTS:
        parser.add_argument(f"--{arg['name']}", **arg['arg_params'])
    return parser.parse_args(args)


class FileParser:
    def __init__(self, config_file, supported_args):
        self.config_file = config_file
        self.arg_dict = {arg['name']: arg['arg_params']
                         for arg in supported_args}

    def parse_file(self):
        with open(self.config_file, 'r') as f:
            result = dict()
            lines = f.readlines()
            line_num = 0
            for line in lines:
                line_num += 1
                line = line.split("#", 1)[0]
                if len(line.strip()) == 0:
                    continue
                if "::" not in line:
                    raise ValueError(
                        f"Key/value separator \"::\" not found on line {line_num}")
                key = line.split("::", 1)[0].strip()
                value = line.split("::", 1)[1].strip()
                if not key in self.arg_dict:
                    raise ValueError(
                        f'unsupported argument "{key}" on line "{line_num}"')
                result[key] = self.arg_dict[key]['type'](value)

            return result


def _getThreads():
    """ Returns the number of available threads on a posix/win based system """
    if sys.platform == 'win32':
        return (int)(os.environ['NUMBER_OF_PROCESSORS'])
    else:
        return (int)(os.popen('grep -c ^processor /proc/cpuinfo 2>/dev/null || sysctl -n hw.ncpu').read())


def parse_file_args(config_file):
    return FileParser(config_file, SUPPORTED_ARGUMENTS).parse_file()

def parse_vnnlib_args(vnnlib_file):
    vnn = VNNLib(vnnlib_file)
    return {
            "initial_point": vnn.get_centerpoint().tolist(),
            "label": vnn.get_intended_class(),
            "radius": vnn.get_radii().tolist()
    }


def combine_args(supported_args, *arg_sets):
    """ Combine args with priority from first in the list to last in the list """
    base_dict = arg_sets[0]
    sup_arg_dict = {arg['name']: arg['arg_params'] for arg in supported_args}
    for i in range(1,len(arg_sets)):
        this_dict = arg_sets[i]
        if this_dict is None:
            continue

        for key, value in base_dict.items():
            supported_arg = sup_arg_dict[key]
            if key in this_dict:
                if value is None or ('default' in supported_arg and value == supported_arg['default']):
                    base_dict[key] = this_dict[key]
    return base_dict


def parse_args(args, prog):
    cmd_args = parse_cmdline_args(args, prog)
    file_args = parse_file_args(cmd_args.config_file)
    vnnlib_args = parse_vnnlib_args(cmd_args.vnnlib_file)
    return combine_args(SUPPORTED_ARGUMENTS, cmd_args, file_args, vnnlib_args)


if __name__ == "__main__":
    parse_args(sys.argv[1:], prog=sys.argv[0])
