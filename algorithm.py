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
    if config["initial_point"] is not None and not safety_predicate(config["initial_point"]):
        # UNSAFE: Add I to SOME_UNSAFE queue
        sets[SOME_UNSAFE](config["initial_region"], config["initial_point"], reporter.total_area)
    else:
        # SAFE: Add I to UNKNOWN set
        sets[UNKNOWN](config["initial_region"], reporter.total_area)
    
    if config['timeout'] <= 0:
        timed_out = lambda: False
    else:
        timed_out = lambda: reporter.get_elapsed_time() > config['timeout']

    # Main Loop: Stop if timeout expires or if UNKNOWN and SOME_UNSAFE are both empty
    while True: #(sets[UNKNOWN].set.size() > 0 or not sets[SOME_UNSAFE].queue.empty()) \
            #and not (config['timeout'] > 0 and reporter.get_elapsed_time() > config['timeout']):
        reporter.report_status()

        if timed_out():
            # If timeout expires, stop.
            break

        if len(sets[UNKNOWN]) > 0:
            # Pull from UNKNOWN first, if any (TODO: prove maintains largest-first)
            region = sets[UNKNOWN].get_next()
            adv_example = None
            was_unsafe = False
        elif len(sets[SOME_UNSAFE]) > 0:
            # If UNKNOWN is empty, pull from SOME_UNSAFE
            region, adv_example = sets[SOME_UNSAFE].get_next()
            was_unsafe = True
        else:
            # If both are empty, we're done!
            break

        if region.low.shape != config['graph'].input_shape:
            region.low = region.low.reshape(config['graph'].input_shape).astype(config['graph'].input_dtype)
        if region.high.shape != config['graph'].input_shape:
            region.high = region.high.reshape(config['graph'].input_shape).astype(config['graph'].input_dtype)
        region_area = reporter.get_area(region)

        if ((region.high - region.low) <= 0).any(): # NOTE: Only necessary for UNKNOWN
            # Empty regions should be pretty rare, but they are possible in discrete cases
            continue # Grab the next one

        if was_unsafe: # We grabbed an unsafe region
            partition = config['strategy']['partitioning'].partition(region)
            for r in partition:
                r_area = reporter.get_area(r)
                if adv_example is not None and point_in_region(r, adv_example):
                    sets[SOME_UNSAFE].append((r, adv_example))
                    #reporter.move_area(SOME_UNSAFE, SOME_UNSAFE, r_area) # Redundant
                    #reporter.add_adversarial_example(adv_example) # Already known
                    # TODO: Handle case where verifier can improve some_unsafe to all_unsafe
                else:
                    sets[UNKNOWN](r, r_area, from_=SOME_UNSAFE)
            continue # Regions added to the Unknown set will be the only ones there, and will be processed first
            # NOTE: This does NOT preserve largest-first ordering

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
        sets[SOME_UNSAFE](region, adv_example, area, from_=from_)
    elif verification == UNKNOWN or verification == TOO_BIG:
        partitions = config['strategy']['partitioning'].partition(region)
        for partition in partitions:
            falsify(config, partition, sets['reporter'].get_area(partition), sets, from_=from_)

def falsify(config, region, area, sets, from_=UNKNOWN):
    # TODO: Pass parent data to child
    abstraction_engine = config['strategy']['abstraction'].abstract
    n = config['num_abstractions']
    safety_predicate = config['safety_predicate']

    abstractions = abstraction_engine(region, n)

    for point in abstractions:
        if not safety_predicate(point):
            if point_in_region(region, point):
                sets[SOME_UNSAFE](region, point, area, from_=from_)
                break
        else:
            # All abstracted points were safe
            sets[UNKNOWN](region, area, from_=from_)


TREND_EPSILON = 0.01
RECURSION_DEPTH = 0
TINY_SUBREGIONS = 8
TINY_SUBREGION_SIZE_DIVISOR = 1000
def check_boundary(config, region):
    """
    Returns:
        True - Should continue
        False - Should halt
    Prints:
        Notification, if possible
    """
    data = region.data

    if not data.initialized or isnan(data.confidence) or isnan(data.confidence_parent):
        return True # Need more data...
    this_trend = data.confidence_parent - data.confidence

    if abs(this_trend) < TREND_EPSILON:
        return True # Good Progress
    if not isnan(data.confidence_grandparent):
        parent_trend = data.confidence_grandparent - data.confidence_parent
        if abs(parent_trend) < TREND_EPSILON:
            return True # May be temporary - previously Good Progress

    if not data.siblings_equal: # Recursion
        if data.recursion_depth > RECURSION_DEPTH:
            return False # Beyond recursion depth
        else:
            data.recursion_depth += 1
            return True # Refine the lines
    # Check for Too Big vs. Fuzzy & Even
    else:
        verification_engine = config['strategy']['verification'].verify
        region_size = region.high - region.low
        for _ in range(TINY_SUBREGIONS):
            tiny_region = np.random.random(size=len(region.low)) * region_size + region.low
            tiny_region = Region(tiny_region, tiny_region + region_size / TINY_SUBREGION_SIZE_DIVISOR)
            verification, adv_example = verification_engine(tiny_region)
            if abs(verification - this[0]) < TREND_EPSILON:
                print("Very large region detected")
                return True # Too Big
        return False # Fuzzy & Even


