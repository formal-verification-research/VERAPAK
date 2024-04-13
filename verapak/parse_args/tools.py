import argparse  # https://docs.python.org/3/library/argparse.htm
import sys
import os
import json
from verapak.parse_args import strategy_registry
from verapak.utilities.vnnlib_lib import VNNLib, NonMaximalVNNLibError
from verapak.parse_args.types import type_string_to_type

SUPPORTED_ARGUMENTS = json.load(open("verapak/parse_args/args.json", 'r'))


def parse_cmdline_args(args, prog):
    parser = argparse.ArgumentParser(
        description="VERAPAK: framework for verifying adversarial robustness of DNNs", prog=prog)
    for arg in SUPPORTED_ARGUMENTS:
        if "type" in arg["arg_params"]:
            arg["arg_params"]["type"] = type_string_to_type(arg["arg_params"]["type"])
        parser.add_argument(f"--{arg['name']}", **arg['arg_params'])
    add_per_strategy_groups(parser, prog)
    return parser.parse_args(args)

def get_strategy_groups_supported():
    supported = list()
    for strategy_type in strategy_registry.ALL_STRATEGIES.values():
        for strategy in strategy_type.values():
            params = strategy.get_config_parameters()
            if params is not None and len(params) > 0:
                supported.extend(params)
    return supported

def add_per_strategy_groups(parser, prog):
    params_listed = {}
    strat_dict = {}
    for strategy_type, v1 in strategy_registry.ALL_STRATEGIES.items():
        for strategy, v2 in v1.items():
            params = v2.get_config_parameters()
            if params is not None and len(params) > 0:
                name = f"{strategy_type}/\033[1m{strategy}\033[0m"
                d = {"desc": [], "args": []}
                for arg in params:
                    if arg["name"] in params_listed:
                        d["desc"].append(arg["name"])
                    else:
                        d["args"].append(arg)
                        params_listed[arg["name"]] = name
                if len(d["desc"]) == 0:
                    d["desc"] = None
                else:
                    desc = ""
                    first = True
                    for desc_part in d["desc"]:
                        if not first:
                            desc += "\n"
                        else:
                            first = False
                        desc += f"  --{desc_part} (see definition above in {params_listed[desc_part]})"
                    d["desc"] = desc
                strat_dict[name] = d
                
    for strat, d in strat_dict.items():
        group = parser.add_argument_group(" " + strat, d["desc"])
        for arg in d["args"]:
            group.add_argument(f"--{arg['name']}", **arg["arg_params"])


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
                result[key] = type_string_to_type(self.arg_dict[key]['type'])(value)

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
    domain = vnn.get_domain()
    return {
            "initial_point": vnn.get_centerpoint().tolist(),
            "label": vnn.get_intended_class(),
            "radius": vnn.get_radii().tolist(),
            "domain_lower_bound": domain[0],
            "domain_upper_bound": domain[1]
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
        if "error" in this_dict:
            if "error" not in base_dict:
                base_dict["error"] = str(this_dict["error"])
            else:
                base_dict["error"] += "+" + str(this_dict["error"])
    return base_dict


def parse_args(args, prog):
    try:
        cmd_args = parse_cmdline_args(args, prog)
    except ValueError as e:
        print(e.args)
        setattr(cmd_args, "error", "bad_cmd_args")

    if hasattr(cmd_args, "config_file"):
        try:
            file_args = parse_file_args(cmd_args.config_file)
        except ValueError as e:
            print(e.args)
            file_args = {"error": "bad_config_args"}
    else:
        file_args = None

    try:
        if hasattr(cmd_args, "vnnlib_file"):
            vnnlib_args = parse_vnnlib_args(cmd_args.vnnlib_file)
        elif file_args is not None and "vnnlib" in file_args:
            vnnlib_args = parse_vnnlib_args(file_args["vnnlib"])
        else:
            vnnlib_args = None
    except NonMaximalVNNLibError as e:
        vnnlib_args = {"error": "nonmaximal"}
    except:
        vnnlib_args = {"error": "bad_vnnlib_args"}

    args = combine_args([*SUPPORTED_ARGUMENTS, *get_strategy_groups_supported()], vars(cmd_args), file_args, vnnlib_args)
    
    if "error" in args:
        return args

    return args

if __name__ == "__main__":
    parse_args(sys.argv[1:], prog=sys.argv[0])
