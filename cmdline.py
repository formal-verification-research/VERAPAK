import argparse # https://docs.python.org/3/library/argparse.htm
import sys
import os

parser = argparse.ArgumentParser(prog="VERAPAK", description="")
parser.add_argument("filepath", type=argparse.FileType('r'), help="Path to the config file") # Places the config file in parsed_args.filepath, already opened and in read mode.

def directory(string):
    if os.path.isdir(string):
        return string
    else:
        raise ValueError(string + " is not a valid directory")

fileparser = argparse.ArgumentParser()
fileparser.add_argument("output_directory", type=directory, help="The path where adversarial examples should be sent")

graphgroup = fileparser.add_argument_group("graph")
graphgroup.add_argument("graph", type=argparse.FileType('r'), help="Path to the graph")
graphgroup.add_argument("-i", "--input", type=str, help="Graph's input node")
graphgroup.add_argument("-o", "--output", type=str, help="Graph's output node")
rad = graphgroup.add_mutually_exclusive_group(required=True)
rad.add_argument("--radius", nargs=1, help="Single radius to apply to all dimensions")
rad.add_argument("--radii", nargs="+", help="Per-dimension radii, separated by either commas or spaces") # Pre-parse, split on commas or set nargs to 1
gran = graphgroup.add_mutually_exclusive_group(required=True)
gran.add_argument("--granularity", nargs=1, help="Single granularity to apply to all dimensions")
gran.add_argument("--granularities", nargs="+", help="Per-dimension granularities, separated by commas or spaces") # Pre-parse, split on commas or set nargs to 1


# TODO: Make everything file-based as opposed to command line arguments (except maybe a few smaller, highly-probable-to-change flags like strategies)

def parse_args(args, exit_on_error=True):
    global parsed_args
    parsed_args = parser.parse_args(args)
    config_lines = parsed_args.filepath.readlines()


#    for line in config_lines:
            

#if __name__ == "__main__":
#    parse_args(sys.argv[1:])



