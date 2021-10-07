import argparse # https://docs.python.org/3/library/argparse.htm
import sys
import os
import pkgutil
from importlib import import_module
import re
from .utilities.point_tools import AnyToSingle

def directoryType(string):
    if os.path.isdir(string):
        return string
    else:
        raise ValueError(string + " is not a valid directory")

def graphPathType(string):
    if os.path.access(string, os.R_OK):
        return string, "ONNX" # Assume ONNX (for now)
    else:
        raise ValueError(string + " is not a file or is otherwise inaccessible (e.g. insufficient permissions)")

def perDimensionType(string):
    string = string.strip()
    if "," in string:
        l = string.split(",")
    elif " " in string:
        l = string.split(" ")
    retVal = list()
    for item in l:
        if len(item.strip()) > 0:
            retVal.append(float(item.strip()))
    if len(retVal) == 1:
        return AnyToSingle(retVal[0])
    elif len(retVal) == 0:
        raise ValueError("No value given")
    return retVal

def get_strategy(module):
    if module in ["dimension_ranking", "partitioning"]:
        return lambda name : name
    def import_submodule(name):
        try:
            if "." in name:
                submodule = name.split(".", 1)[0]
                function = name.split(".", 1)[1]
                getattr(import_module(f"{module}.{submodule}"), function)() # Get the submodule `submodule`, and call the specified method.
            else:
                getattr(import_module(f"{module}.{name}"), "IMPL")() # Get the submodule `name`, and call the default method.
        except ModuleNotFoundError:
            print(f"No module \"{module}.{name}\" was found")
            exit(1)
        except AttributeError:
            if "." in name:
                submodule = name.split(".", 1)[0]
                function = name.split(".", 1)[1]
                print(f"Module \"{module}.{name}\" has no \"{function}\" function")
            else:
                print(f"Module \"{module}.{name}\" has no \"IMPL\" function")
            exit(1)
        except Exception as ex:
            raise Exception(ex)
            exit(1)

    return import_submodule

def get_strategy_choices(module):
    submodules = []
    for submodule in pkgutil.iter_modules([module]):
        submodule_obj = submodule.module_finder.find_module(submodule.name)
        with open(submodule_obj.path, "r") as f:
            for line in f:
                if re.match(r"\A[ ]*def[ ]*IMPL[ ]*\([ ]*\)[ ]*:", line):
                    submodules.append(submodule.name)
                    break

    ret_str = "("
    ret_str += " | ".join(submodules)
    ret_str += " | ...)"
    return ret_str

class PrintAction(argparse.Action):
    def __init__(self,
                 option_strings,
                 string,
                 dest=argparse.SUPPRESS,
                 default=argparse.SUPPRESS,
                 help=None):
        super(PrintAction, self).__init__(
            option_strings=option_strings,
            dest=dest,
            default=default,
            nargs=0,
            help=help)
        self.string = string

    def __call__(self, parser, namespace, values, option_string=None):
        print(self.string)
        parser.exit()

class FileParser:
    def __init__(self, arg_list, defaults):
        self.required_str = "\033[31mREQUIRED\033[0m"
        self.arg_list = arg_list

        self.arg_count = len([arg for arg in self.arg_list if arg is not None])

        self.arg_dict = dict()
        for arg in self.arg_list:
            if arg is None:
                continue
            self.arg_dict[arg['name']] = arg
        self.defaults = defaults

    def get_help(self, description=""):
        help_string = description
        if len(help_string) > 0:
            help_string += "\n\n"
        for arg in self.arg_list:
            if arg is None:
                help_string += "\n"
                continue
            help_string += f"{arg['name']} :: {arg['help']}"
            if 'required' in arg and arg['required']:
                help_string += f" {self.required_str}"
            elif 'default' in arg and arg['default'] is not None:
                help_string += f" (Default: {self.defaults[arg['default']]})"
            if 'options' in arg:
                help_string += f"\n\t\t Options: "
                help_string += arg['options']
            help_string += "\n"
        return help_string
    
    def parse_lines(self, lines):
        result = dict()
        line_num = 0
        for line in lines:
            line_num += 1
            line = line.split("#", 1)[0]
            if len(line.strip()) == 0:
                continue
            if "::" not in line:
                raise ValueError(f"Key/value separator \"::\" not found on line {line_num}")
            key = line.split("::", 1)[0].strip()
            value = line.split("::", 1)[1].strip()
            if not key in self.arg_dict:
                raise ValueError(f"No variable named \"{key}\" on line {line_num}")
            else:
                result[key] = self.arg_dict[key]['type'](value)

        for arg in self.arg_list:
            if arg is not None and arg['name'] not in result:
                if 'default' in arg:
                    if not arg['required'] and arg['default'] is not None and arg['default'] in self.defaults:
                        result[arg['name']] = self.defaults[arg['default']]
                    elif arg['default'] == argparse.SUPPRESS:
                        result[arg['name']] = self.defaults[arg['short']]
                    elif arg['default'] is None:
                        result[arg['name']] = None

            if arg is not None and arg['name'] not in result:
                raise KeyError(arg['name'])

        return result

    def parse_overhead(self, args, prog, prog_name=None, description=None):
        parser = argparse.ArgumentParser(prog=prog, description=description, usage="%(prog)s [-h | -f? | options] filepath")
        parser.add_argument("-f?", string=self.get_help(description=f"{prog_name} Config File Help"), action=PrintAction, dest="filehelp", help="show config file help message and exit")
        parser.add_argument("filepath", type=argparse.FileType('r'), help="Path to the config file")
        for arg in self.arg_list:
            if arg is not None and arg['flag']:
                if 'default' in arg and arg['default'] is not None and arg['default'] != argparse.SUPPRESS:
                    default = self.defaults[arg['default']]
                else:
                    default = arg['default']
                parser.add_argument("--" + arg['short'], help=arg['help'], required=False, default=default)
        namespace = parser.parse_args(args)
        self.defaults = vars(namespace)
        return namespace.filepath


# To add a flag, the value must be optional and have a default value.
ARG_LIST = [
        {'flag': True, 'short': 'dir', 'name': "Output Directory", 'type': directoryType,                       'help': "The path where adversarial examples should be sent",                     'required': True, 'default': argparse.SUPPRESS},
        {'flag': True, 'short': 'pnt', 'name': "Point",            'type': lambda s: np.fromstring(s, sep=','), 'help': "The point at which to test: per-dimension coordinates, comma separated", 'required': True, 'default': argparse.SUPPRESS},
        None,
        {'flag': False, 'short': 'grf', 'name': "Graph",        'type': graphPathType,    'help': "Path to the graph",                                                                                                  'required': True},
        {'flag': False, 'short': 'in',  'name': "Input",        'type': str,              'help': "Graph's input node (If not given, will try to guess)",                                                               'required': False, 'default': None},
        {'flag': False, 'short': 'out', 'name': "Output",       'type': str,              'help': "Graph's output node (If not given, will try to guess)",                                                              'required': False, 'default': None},
        {'flag': False, 'short': 'rad', 'name': "Radius",       'type': perDimensionType, 'help': "Single radius or per-dimension radii, comma separated",                                                              'required': True},
        {'flag': False, 'short': 'grn', 'name': "Granularity",  'type': perDimensionType, 'help': "Single granularity or per-dimension granularities, comma/space separated",                                           'required': True},
        {'flag': False, 'short': 'lbl', 'name': "Label",        'type': int,              'help': "Intended class label number (use index of logit). If not provided, class of Point is assumed as the intended class", 'required': False, 'default': None},
        None,
        {'flag': True, 'short': 'thr', 'name': "Threads",      'type': int,   'help': "Number of threads to use",                           'required': False, 'default': 'thr'},
        {'flag': True, 'short': 'abs', 'name': "Abstractions", 'type': int,   'help': "Number of abstraction points to generate each pass", 'required': False, 'default': 'abs'},
        {'flag': True, 'short': 'bal', 'name': "FGSM Balance", 'type': float, 'help': "FGSM vs. Random balance factor",                     'required': False, 'default': 'bal'},
        None,
        {'flag': True, 'short': 's-ver', 'name': "Verification Strategy",      'type': get_strategy('verification'),      'help': "Which verification strategy to use",      'required': False, 'default': 's_ver',      'options': get_strategy_choices('verification')},
        {'flag': True, 'short': 's-dim', 'name': "Dimension-Ranking Strategy", 'type': get_strategy('dimension_ranking'), 'help': "Which dimension-ranking strategy to use", 'required': False, 'default': 's_dim', 'options': get_strategy_choices('dimension_ranking')},
        {'flag': True, 'short': 's-abs', 'name': "Abstraction Strategy",       'type': get_strategy('abstraction'),       'help': "Which abstraction strategy to use",       'required': False, 'default': 's_abs',       'options': get_strategy_choices('abstraction')},
        {'flag': True, 'short': "s-par", 'name': "Partitioning Strategy",      'type': get_strategy('partitioning'),      'help': "Which partitioning strategy to use",      'required': False, 'default': 's_par',      'options': get_strategy_choices('partitioning')},
]


def _getThreads():
    """ Returns the number of available threads on a posix/win based system """
    if sys.platform == 'win32':
        return (int)(os.environ['NUMBER_OF_PROCESSORS'])
    else:
        return (int)(os.popen('grep -c ^processor /proc/cpuinfo 2>/dev/null || sysctl -n hw.ncpu').read())

ABSOLUTE_DEFAULTS = {
        'thr': _getThreads(),
        'abs': 10,
        'bal': 0.9,
        's_ver': "discrete_search",
        's_dim': "gradient_based",
        's_abs': "modfgsm",
        's_par': "largest_first",
}

def parse_args(args, prog=None):
    fp = FileParser(ARG_LIST, ABSOLUTE_DEFAULTS)
    file = fp.parse_overhead(args, prog=prog, prog_name="VERAPAK", description="")
    with file:
        config = fp.parse_lines(file.readlines())
    return config

if __name__ == "__main__":
    parse_args(sys.argv[1:], prog=sys.argv[0])

