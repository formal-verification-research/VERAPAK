import os
import sys
import time
import numpy as np
import queue
import traceback
from config import Config
from verapak.parse_arg_tools import parse_args
from verapak.constraints import SafetyPredicate
from verapak.verification.ve import ALL_SAFE, ALL_UNSAFE, SOME_UNSAFE, TOO_BIG
import verapak_utils
from verapak.utilities.point_tools import point_in_region

class DoneInterrupt(Exception):
    pass

def snap_point(config, point):
    return snap.point_to_domain(point, config['domain'])

class Reporter:
    def __init__(self):
        self.started = False

    def get_area(self, region):
        area = (region[1] - region[0]) / self.scaling
        return np.prod(area)

    def setup(self, config, start_time=time.time()):
        self.config = config
        self.start_time = start_time
        self.last_report = start_time
        self.set_initial_region(config['initial_region'])
        self.found_first_example = False
        self.started = True
    
    def set_initial_region(self, initial_region):
        self.initial_region = initial_region
        self.scaling = initial_region[1] - initial_region[0] # Yields the total range of inputs - thus scaling to a 1x1x1x...x1 hypercube
        self.total_area = 1                                  # <-- which has a hypervolume of exactly 1
        self.all_safe_area = 0
        self.all_unsafe_area = 0
        self.unsafe_area = 0
        self.unknown_area = 0
        self.adversarial_examples = verapak_utils.PointSet()
        assert self.total_area > 0, "Initial region has no or negative area"

    def move_area(self, set_from, set_to, amount):
        if set_from == set_to:
            return
        print(set_from + " => " + set_to + f" ({amount})")
        self.add_area(set_from, -amount)
        self.add_area(set_to, amount)

    def add_area(self, which, amount):
        if which == "all_safe":
            self.all_safe_area += amount
        elif which == "all_unsafe":
            self.all_unsafe_area += amount
        elif which == "some_unsafe":
            self.unsafe_area += amount
        elif which == "unknown":
            self.unknown_area += amount
        elif which is not None:
            raise ValueError("Bad set name \"" + which + "\"")

    def add_adversarial_example(self, example):
        elapsed_time = self.get_elapsed_time()
        show = None
        if example is not None:
            print(f"Found adversarial example @ {elapsed_time}")
            self.adversarial_examples.insert(example)
            show = ["loose", "strict"]
        if example is None:
            show = ["loose"]
        if self.config['halt_on_first'] in show:
            output_file = os.path.join(self.config['output_dir'], 'time_to_first.txt')
            print(f"Saving time to '{output_file}'")
            output_file = open(output_file, "a")
            output_file.write(f"{elapsed_time} seconds\n")
            output_file.close()
            self.halt_reason = "first"
            raise DoneInterrupt()
    
    def report_status(self):
        elapsed_time = time.time() - self.last_report
        if elapsed_time < self.config['raw']['report_interval_seconds']:
            return False
        self.do_report_status()
        return True
    def do_report_status(self):
        percent_unknown = (self.unknown_area / self.total_area) * 100
        percent_all_safe = (self.all_safe_area / self.total_area) * 100
        percent_all_unsafe = (self.all_unsafe_area / self.total_area) * 100
        percent_unsafe = (self.unsafe_area / self.total_area) * 100
        percent_unaccounted = 100 - percent_unknown - percent_all_safe - percent_all_unsafe \
            - percent_unsafe
        print(f"@ {self.get_elapsed_time()}")
        print(f"Percent unknown: {percent_unknown}%")
        print(f"Percent known safe: {percent_all_safe}%")
        print(f"Percent known unsafe: {percent_all_unsafe}%")
        print(f"Percent partially unsafe: {percent_unsafe}%")
        print(f"Adversarial examples: {self.adversarial_examples.size()}")
        if percent_unaccounted > 0:
            print(f"Points unaccounted for: {percent_unaccounted} (/{self.total_area})")
        print()
        self.last_report = time.time()

    def give_final_report(self):
        print('\n')
        print("Final Report")
        print("#############################")
        self.do_report_status()

    def get_elapsed_time(self):
        return time.time() - self.start_time
    def get_halt_reason(self):
        return self.halt_reason
    def get_adversarial_examples(self):
        return self.adversarial_examples

def main(config, reporter):
    start_time = time.time()
    reporter.setup(config, start_time)
    
    unknown_set = verapak_utils.RegionSet()
    all_safe_set = verapak_utils.RegionSet()
    all_unsafe_set = verapak_utils.RegionSet()
    unsafe_queue = queue.Queue()

    safety_predicate = config["safety_predicate"]
    if config['initial_point'] is not None and not safety_predicate(config['initial_point']): # Initial point doesn't follow target label
        # Add initial region to unsafe queue, with the initial point as the adversarial point
        unsafe_queue.put((config['initial_region'], config['initial_point']))
        reporter.add_area("some_unsafe", reporter.total_area)
        reporter.add_adversarial_example(config['initial_point'])
    else:
        # Add initial region to unknown set
        unknown_set.insert(*config['initial_region'])
        reporter.add_area("unknown", reporter.total_area)
    
    while (unknown_set.size() > 0 or not unsafe_queue.empty()) and not (config['timeout'] > 0 and reporter.get_elapsed_time() > config['timeout']):
        print("Loop")
        reporter.report_status()

        if unknown_set.size() > 0:
            region = unknown_set.pop_random()[1]
            adv_example = None
            was_unsafe = False
        else:
            region, adv_example = unsafe_queue.get_nowait()
            was_unsafe = True

        region = [x.reshape(config['graph'].input_shape).astype(config['graph'].input_dtype) for x in region]
        region_area = reporter.get_area(region)

        if ((region[1] - region[0]) <= 0).any():
            print("Empty region")
            continue # Grab the next one

        if was_unsafe: # We grabbed an unsafe region
            partition = config['strategy']['partitioning'].partition_impl(region)
            for r in partition:
                r_area = reporter.get_area(r)
                if adv_example is not None and point_in_region(r, adv_example):
                    unsafe_queue.put_nowait((r, adv_example))
                    #reporter.move_area("some_unsafe", "some_unsafe", r_area) # Redundant
                else:
                    unknown_set.insert(*r)
                    reporter.move_area("some_unsafe", "unknown", r_area)
            continue

        verification, adv_example = config['strategy']['verification'].verification_impl(region, safety_predicate)

        if verification == ALL_SAFE:
            all_safe_set.insert(*region)
            reporter.move_area("unknown", "all_safe", region_area)
        elif verification == ALL_UNSAFE:
            all_unsafe_set.insert(*region)
            reporter.move_area("unknown", "all_unsafe", region_area)
        elif verification == SOME_UNSAFE:
            unsafe_queue.put_nowait((region, adv_example))
            reporter.move_area("unknown", "some_unsafe", region_area)
            reporter.add_adversarial_example(adv_example)
        elif verification == TOO_BIG:
            partition = config['strategy']['partitioning'].partition_impl(region)
            for partitioned in partition:
                abstraction = config['strategy']['abstraction'].abstraction_impl(partitioned, config['num_abstractions'])
                abstraction = [snap_point(config, x) for x in abstraction] # Snap
                r_area = reporter.get_area(partitioned)
                for point in abstraction:
                    if not safety_predicate(point):
                        if point_in_region(partitioned, point):
                            unsafe_queue.put_nowait((partitioned, point))
                            reporter.move_area("unknown", "some_unsafe", r_area)
                            reporter.add_adversarial_example(point)
                            break
                        else:
                            found, reg = unknown_set.get_and_remove_region_containing_point(point)
                            if found:
                                reg = [x.reshape(config['graph'].input_shape)
                                        .astype(config['graph'].input_dtype)
                                        for x in reg]
                                reg_area = reporter.get_area(reg)
                                unsafe_queue.put_nowait((reg, point))
                                reporter.move_area("unknown", "some_unsafe", reg_area)
                                reporter.add_adversarial_example(point)
                            else:
                                pass # No region was removed
                else: # Did not `break` out of the for loop
                    unknown_set.insert(*partitioned)
                    #reporter.move_area("unknown", "unknown", r_area)
    if config['timeout'] > 0 and reporter.get_elapsed_time() >= config['timeout']:
        reporter.halt_reason = "timeout"
    else:
        reporter.halt_reason = "done"

def create_witness(config, adversarial_example):
    input_values = adversarial_example.flatten(),
    output_values = config['graph'].evaluate(adversarial_example).flatten()

    witness = "("
    for idx, x in np.ndenumerate(input_values):
        witness += f"(X_{idx[0]} {x})\\n"
    for idx, y in np.ndenumerate(output_values):
        witness += f"(Y_{idx[0]} {y})\\n"
    witness += ")"
    return witness

def write_results(config, adversarial_examples, halt_reason, elapsed_time):
    witness = ""
    adv_count = 0
    if adversarial_examples and adversarial_examples.size() > 0:
        witness = create_witness(next(adversarial_examples.elements()))
        adv_count = adversarial_examples.size()
        adv_examples_numpy = np.array([x for x in adversarial_examples.elements()])
        output_file = os.path.join(config['output_dir'], 'adversarial_examples.npy')
        np.save(output_file, adv_examples_numpy)
    if halt_reason in ["done", "first"]:
        halt_reason = "sat" if (adv_count > 0) else "unsat"

    output_file = os.path.join(config['output_dir'], 'report.csv')
    output_file = open(output_file, 'w')
    output_file.write(f"{halt_reason},{witness},{adv_count},{elapsed_time}\n")
    output_file.close()

def run(config):
    global reporter
    reporter = Reporter()
    try:
        main(config, reporter)
    except KeyboardInterrupt as e:
        reporter.halt_reason = "keyboard"
    except DoneInterrupt as e:
        pass
    except BaseException as e:
        reporter.halt_reason = "error"
        traceback.print_exception(type(e), e, e.__traceback__)
    
    if reporter.started:
        reporter.give_final_report()
        et = reporter.get_elapsed_time()
    else:
        et = 0
    adversarial = reporter.get_adversarial_examples()
    halt_reason = reporter.get_halt_reason
    write_results(config, adversarial, halt_reason, et)
    print('done')

if __name__ == "__main__":
    args = sys.argv
    if len(args) == 1:
        args = [args[0], "--help"]
    config = parse_args(args[1:], prog=args[0])
    if "error" in config:
        print(f"\033[38;2;255;0;0mERROR: {config['error']}\033[0m")
        write_results(config, None, "error_" + config["error"], 0, None)
    else:
        config = Config(config)
        for strategy in config["strategy"].values():
            strategy.set_config(config["raw"], config)

        run(config)

        for strategy in config["strategy"].values():
            strategy.shutdown()
