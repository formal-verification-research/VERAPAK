import queue
from verapak.verification.ve import ALL_SAFE, ALL_UNSAFE, SOME_UNSAFE, TOO_BIG, UNKNOWN, BOUNDARY
import verapak_utils
import numpy as np
import time
import os

class DoneInterrupt(Exception):
    pass

def _make_RegionSet(reporter, name):
    s = verapak_utils.RegionSet()
    def wrapped(region, area, from_=None):
        s.insert(*region)
        if from_ is not None:
            reporter.move_area(from_, name, area)
        else:
            reporter.add_area(name, area)
    wrapped.set = s
    wrapped.name = name
    return wrapped

def _make_Queue(reporter, name):
    q = queue.Queue()
    def wrapped(region, e, area, from_=None):
        q.put((region, e))
        if from_ is not None:
            reporter.move_area(from_, name, area)
        else:
            reporter.add_area(name, area)
        reporter.add_adversarial_example(e)
    wrapped.queue = q
    wrapped.name = name
    return wrapped

VALID_SET_NAMES = [UNKNOWN, ALL_SAFE, ALL_UNSAFE, SOME_UNSAFE, BOUNDARY]
DISPLAY_NAMES = {
    UNKNOWN: "Unknown",
    ALL_SAFE: "All Safe",
    ALL_UNSAFE: "All Unsafe",
    SOME_UNSAFE: "Some Unsafe",
    BOUNDARY: "Boundary",
}
def make_sets(reporter):
    return {
        UNKNOWN: _make_RegionSet(reporter, UNKNOWN),
        ALL_SAFE: _make_RegionSet(reporter, ALL_SAFE),
        ALL_UNSAFE: _make_RegionSet(reporter, ALL_UNSAFE),
        SOME_UNSAFE: _make_Queue(reporter, SOME_UNSAFE),
        BOUNDARY: _make_RegionSet(reporter, BOUNDARY),
        "reporter": reporter,
    }

class Reporter:
    def __init__(self):
        self.started = False
        self.areas = {}
        for set_name in VALID_SET_NAMES:
            self.areas[set_name] = 0

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
        self.adversarial_examples = verapak_utils.PointSet()
        assert self.total_area > 0, "Initial region has no or negative area"

    def move_area(self, set_from, set_to, amount):
        if set_from == set_to:
            return
        print(set_from, "=>", set_to, "(" + str(amount) + ")")
        self.add_area(set_from, -amount)
        self.add_area(set_to, amount)

    def add_area(self, which, amount):
        if which in self.areas:
            self.areas[which] += amount
        elif which is not None:
            raise ValueError("Bad set name \"" + which + "\"")

    def add_adversarial_example(self, example):
        elapsed_time = self.get_elapsed_time()
        show = None
        if example is not None:
            print("Found adversarial example @", elapsed_time)
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
        if elapsed_time < self.config.raw['report_interval_seconds']:
            return False
        self.do_report_status()
        return True
    def do_report_status(self):
        print(f"@ {self.get_elapsed_time()}")
        percent_unaccounted = 100
        for area_name in self.areas:
            area_percent = (self.areas[area_name] / self.total_area) * 100
            percent_unaccounted -= area_percent
            print(f"Percent {DISPLAY_NAMES[area_name]}: {area_percent}%")
        print(f"Adversarial examples: {self.adversarial_examples.size()}")
        if percent_unaccounted > 0:
            print(f"Points unaccounted for: {percent_unaccounted}% (Likely due to floating point rounding)")
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


