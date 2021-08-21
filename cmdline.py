import argparse
import sys

parser = argparse.ArgumentParser(description="")
subparsers = parser.add_subparsers(help="sub-command help")

file_parser = subparsers.add_parser("conf", help="Use a config file")
file_parser.add_argument("filepath", type=argparse.FileType('r'), help="Path to the config file") # Places the config file in parsed_args.filepath, already opened and in read mode.

arg_parser = subparsers.add_parser("cmd", help="Supply manually")
arg_parser.add_argument("initial_point", type=list)


def parse_args(args):
    global parsed_args
    parsed_args = parser.parse_args(args)

if __name__ == "__main__":
    parse_args(sys.argv[1:])
