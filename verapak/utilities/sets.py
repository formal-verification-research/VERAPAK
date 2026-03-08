from verapak.verification.ve import ALL_SAFE, ALL_UNSAFE, SOME_UNSAFE, TOO_BIG, UNKNOWN, BOUNDARY
import verapak_utils
import numpy as np
import time
import os
import random
from collections import deque

class DoneInterrupt(Exception):
    pass
class WrappedRegionSet:
    def __init__(self, reporter, name):
        self.reporter = reporter
        self.name = name
        self.set = verapak_utils.RegionSet()
    def __call__(self, region, area, from_=None):
        self.append(region)
        if from_ is not None:
            self.reporter.move_area(from_, self.name, area)
        else:
            self.reporter.add_area(self.name, area)
    def _pop(self, idx):
        v = self.set[idx]
        del self.set[idx]
        return v
    def get_next(self):
        return self._pop(random.randrange(len(self.set)))
    def append(self, v):
        return self.set.append(v)
    def __len__(self):
        return len(self.set)
class WrappedRegionQueue:
    def __init__(self, reporter, name):
        self.reporter = reporter
        self.name = name
        self.queue = deque()
    def __call__(self, region, area, from_=None):
        self.append(region)
        if from_ is not None:
            self.reporter.move_area(from_, self.name, area)
        else:
            self.reporter.add_area(self.name, area)
    def get_next(self):
        return self.queue.popleft()
    def append(self, v):
        return self.queue.append(v)
    def __len__(self):
        return len(self.queue)
class WrappedErrorRegionQueue:
    def __init__(self, reporter, name):
        self.reporter = reporter
        self.name = name
        self.queue = deque()
    def __call__(self, region, e, area, from_=None):
        self.append((region, e))
        if from_ is not None:
            self.reporter.move_area(from_, self.name, area)
        else:
            self.reporter.add_area(self.name, area)
        if e is not None:
            self.reporter.add_adversarial_example(e)
    def get_next(self):
        return self.queue.popleft()
    def append(self, v):
        return self.queue.append(v)
    def __len__(self):
        return len(self.queue)

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
        UNKNOWN: WrappedErrorRegionQueue(reporter, UNKNOWN),
        ALL_SAFE: WrappedRegionSet(reporter, ALL_SAFE),
        ALL_UNSAFE: WrappedRegionSet(reporter, ALL_UNSAFE),
        BOUNDARY: WrappedRegionSet(reporter, BOUNDARY),
        "reporter": reporter,
    }

class Reporter:
    def __init__(self):
        self.started = False
        self.areas = {}
        for set_name in VALID_SET_NAMES:
            self.areas[set_name] = 0

    def get_area(self, region):
        # Adding self.ignore prevents division by zero on ignored dimensions
        # NOTE: Should we add self.ignore to the numerator? What if there *is* a value in an ignored dimension?
        area = (region.high - region.low) / (self.scaling + self.ignore)
        # Adding self.ignore prevents area from being zero when multiplying by ignored dimensions
        # NOTE: Implementing the above note would mean we don't need to add self.ignore here (0/1 + 1 vs. 1/1)
        return np.prod(area + self.ignore)

    def setup(self, config, start_time=time.time()):
        self.config = config
        self.start_time = start_time
        self.last_report = start_time
        self.set_initial_region(config['initial_region'], config["ignored_dimensions"])
        self.found_first_example = False
        self.started = True
    
    def set_initial_region(self, initial_region, ignored):
        self.initial_region = initial_region
        self.scaling = initial_region.high - initial_region.low # Yields the total range of inputs - thus scaling to a 1x1x1x...x1 hypercube
        self.total_area = 1                                  # <-- which has a hypervolume of exactly 1
        self.ignore = ignored # Before using np.prod or dividing, remove ignored dimensions
                              # Either by adding ignore to the numerator and denominator on each, or filtering them out
        self.adversarial_examples = []
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
            print("Found adversarial example @ ", elapsed_time)
            self.adversarial_examples.append(example)
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
        for area_idx in self.areas:
            area_percent = (self.areas[area_idx] / self.total_area) * 100
            percent_unaccounted -= area_percent
            print(f"Percent {DISPLAY_NAMES[area_idx]}: {area_percent}%")
        print(f"Adversarial examples: {len(self.adversarial_examples)}")
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


