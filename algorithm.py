import time
import numpy as np

from verapak.verification.ve import ALL_SAFE, ALL_UNSAFE, SOME_UNSAFE, TOO_BIG, UNKNOWN, BOUNDARY
from verapak.utilities.point_tools import point_in_region
from verapak.utilities.sets import make_sets

def main(config, reporter):
    start_time = time.time()
    reporter.setup(config, start_time)
    
    sets = make_sets(reporter)

    safety_predicate = config["safety_predicate"]
    # Check initial point safety
    initial_point_safe = config["initial_point"] is None or safety_predicate(config["initial_point"])
    sets[UNKNOWN](config["initial_region"], None if initial_point_safe else config["initial_point"], reporter.total_area)
    
    if config['timeout'] <= 0:
        timed_out = lambda: False
    else:
        timed_out = lambda: reporter.get_elapsed_time() > config['timeout']

    # Main Loop: Stop if timeout expires or if UNKNOWN is empty
    while True: # sets[UNKNOWN].set.size() > 0 and not (config['timeout'] > 0 and reporter.get_elapsed_time() > config['timeout']):
        reporter.report_status()

        if timed_out() or len(sets[UNKNOWN]) == 0:
            # If timeout expires or there are no more regions, stop.
            break
            
        region, adv_example = sets[UNKNOWN].get_next()

        if region.low.shape != config['graph'].input_shape:
            region.low = region.low.reshape(config['graph'].input_shape).astype(config['graph'].input_dtype)
        if region.high.shape != config['graph'].input_shape:
            region.high = region.high.reshape(config['graph'].input_shape).astype(config['graph'].input_dtype)
        region_area = reporter.get_area(region)

        if ((region.high - region.low) <= 0).any():
            # Empty regions should be pretty rare, but they are possible in discrete cases
            continue # Grab the next one

        if adv_example is not None: # We grabbed an unsafe region
            partitions = config['strategy']['partitioning'].partition(region)
            for r in partitions:
                r_area = reporter.get_area(r)
                sets[UNKNOWN](r, None if not point_in_region(r, adv_example) else adv_example)
            continue

        verify(config, region, region_area, sets)

    if timed_out():
        reporter.halt_reason = "timeout"
    else:
        reporter.halt_reason = "done"

def verify(config, region, area, sets, from_=UNKNOWN):
    # TODO: Check confidence level, and sometimes send directly to Falsify

    verification_engine = config['strategy']['verification'].verify

    verification, adv_example = verification_engine(region)

    if verification == ALL_SAFE or verification == ALL_UNSAFE:
        sets[verification](region, area, from_=from_)
    elif verification == SOME_UNSAFE:
        sets[UNKNOWN](region, adv_example, area, from_=from_)
    elif verification == UNKNOWN or verification == TOO_BIG:
        partitions = config['strategy']['partitioning'].partition(region)
        for partition in partitions:
            falsify(config, partition, sets['reporter'].get_area(partition), sets, from_=from_)

def falsify(config, region, area, sets, from_=UNKNOWN):
    falsification_engine = config['strategy']['falsification'].abstract
    n = config['num_abstractions']
    safety_predicate = config['safety_predicate']

    abstractions = falsification_engine(region, n)

    for point in abstractions:
        if not safety_predicate(point):
            # TODO: Move the point in region check to the falsification engines
            if point_in_region(region, point):
                sets[UNKNOWN](region, point, area, from_=from_)
                break
        else:
            # All abstracted points were safe
            sets[UNKNOWN](region, None, area, from_=from_)
