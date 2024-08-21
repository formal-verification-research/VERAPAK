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

        if sets[UNKNOWN].set.size() > 0:
            # Pull from UNKNOWN first, if any (TODO: prove maintains largest-first)
            region = sets[UNKNOWN].set.pop_random()[1]
            adv_example = None
            was_unsafe = False
        elif not sets[SOME_UNSAFE].queue.empty():
            # If UNKNOWN is empty, pull from SOME_UNSAFE
            region, adv_example = sets[SOME_UNSAFE].queue.get_nowait()
            was_unsafe = True
        else:
            # If both are empty, we're done!
            break

        region = (
            region[0].reshape(config['graph'].input_shape).astype(config['graph'].input_dtype),
            region[1].reshape(config['graph'].input_shape).astype(config['graph'].input_dtype),
            region[2]
        )
        region_area = reporter.get_area(region)

        if ((region[1] - region[0]) <= 0).any(): # NOTE: Only necessary for UNKNOWN
            # Empty regions should be pretty rare, but they are possible in discrete cases
            continue # Grab the next one

        if was_unsafe: # We grabbed an unsafe region
            partition = config['strategy']['partitioning'].partition_impl(region)
            for r in partition:
                r_area = reporter.get_area(r)
                if adv_example is not None and point_in_region(r, adv_example):
                    sets[SOME_UNSAFE].queue.put_nowait((r, adv_example))
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

    verification_engine = config['strategy']['verification'].verification_impl
    safety_predicate = config['safety_predicate']

    verification, adv_example = verification_engine(region, safety_predicate)

    if verification == ALL_SAFE or verification == ALL_UNSAFE:
        sets[verification](region, area, from_=from_)
    elif verification == SOME_UNSAFE:
        sets[SOME_UNSAFE](region, adv_example, area, from_=from_)
    elif verification == UNKNOWN or verification == TOO_BIG:
        partitions = config['strategy']['partitioning'].partition_impl(region)
        for partition in partitions:
            falsify(config, partition, sets['reporter'].get_area(partition), sets, from_=from_)

def falsify(config, region, area, sets, from_=UNKNOWN):
    # TODO: Pass parent data to child
    abstraction_engine = config['strategy']['abstraction'].abstraction_impl
    n = config['num_abstractions']
    safety_predicate = config['safety_predicate']

    abstractions = abstraction_engine(region, n)

    for point in abstractions:
        if not safety_predicate(point):
            if point_in_region(region, point):
                sets[SOME_UNSAFE](region, point, area, from_=from_)
                break
            else: # In case our abstraction engine gives a point outside this region
                # Only check UNKNOWN because SOME_UNSAFE is redundant, and ALL_UNSAFE and ALL_SAFE should be impossible
                found, found_region = sets[UNKNOWN].set.get_and_remove_region_containing_point(point)
                if found:
                    found_region = [x.reshape(config['graph'].input_shape)
                                    .astype(config['graph'].input_dtype)
                                    for x in found_region]
                    # TODO: define get_area(partition) in a better location
                    found_region_area = sets['reporter'].get_area(found_region)
                    sets[SOME_UNSAFE](found_region, point, found_region_area, from_=from_)
    else:
        # All abstracted points were safe
        sets[UNKNOWN](region, area, from_=from_)


TREND_EPSILON = 0.01
RECURSION_DEPTH = 0
TINY_SUBREGIONS = 8
def check_boundary(config, region):
    """
    data[0] = This %
    data[1] = Parent %
    data[2] = Grandparent %
    data[3] = Sibling %'s
    data[4] = Recursion depth

    Returns:
        True - Should continue
        False - Should halt
    Prints:
        Notification, if possible
    """
    data = region[2]

    if data[0] is None or data[1] is None:
        return True
    this_trend = data[1] - data[0]

    if data[2] is not None:
        parent_trend = data[2] - data[1]
        if abs(parent_trend) < TREND_EPSILON:
            return True # May be temporary - previously Good Progress
    if abs(this_trend) < TREND_EPSILON:
        return True # Good Progress

    siblings_equal = True
    for sibling in data[3]:
        if sibling != data[0]:
            siblings_equal = False
            break
    
    if not siblings_equal: # Recursion
        if data[4] > RECURSION_DEPTH:
            return False # Beyond recursion depth
        else:
            region[2] = (data[0], data[1], data[2], data[3], data[4] + 1)
            return True # Refine the lines
    # Check for Too Big vs. Fuzzy & Even
    else:
        verification_engine = config['strategy']['verification'].verification_impl
        region_size = region[1] - region[0]
        for _ in range(TINY_SUBREGIONS):
            tiny_region = np.random.random(size=len(region[0])) * region_size + region[0]
            tiny_region = (tiny_region, tiny_region + region_size / 1000)
            verification, adv_example = verification_engine(region, safety_predicate)
            if abs(verification - this[0]) < TREND_EPSILON:
                print("Very large region detected")
                return True # Too Big
        return False # Fuzzy & Even

def handle_boundary(config, region, area, sets, from_=UNKNOWN):
def should_stop_partitioning(config, region, safety_predicate):
    # Check for too big vs. fuzzy & even
    verification_engine = config['strategy']['verification'].verification_impl
    for _ in range(TINY_SUBREGIONS):
        tiny_region = np.random.random(size=len(region[0])) * (region[1] - region[0]) + region[0]
        tiny_region = (tiny_region, tiny_region + granularity) # TODO: Get/pick granularity
        verification, adv_example = verification_engine(region, safety_predicate)
        if abs(verification - this[0]) > TREND_EPSILON:
            return False # Too big
    return True # Fuzzy & even

