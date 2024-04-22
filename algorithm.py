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
# Region[2]:
#  0 - This %
#  1 - Parent
#  2 - Recursions
#  3 - Children %
def should_stop_partitioning(config, region, safety_predicate):
    get_trend = lambda x: x[1][0] - x[0]
    this = region[2]
    parent = this[1]

    if parent is None:
        return False # Only the first iteration; no trend data
    elif abs(get_trend(this)) > TREND_EPSILON:
        return False # Making progress

    # Check for likeness of siblings
    siblings = parent[3]
    alike = True
    for sibling in siblings:
        if abs(sibling - parent[0]) > TREND_EPSILON:
            alike = False
            break

    if not alike:
        # Check for recursion
        grandfather = parent[1]
        if grandfather is None:
            return False # Only the second iteration; no recursion trend data
        uncles = sorted(grandfather[3])
        siblings = sorted(siblings)
        for i in range(len(siblings)):
            if abs(uncles[i] - siblings[i]) > TREND_EPSILON:
                break
        else:
            this[2] += 1
        if this[2] > RECURSION_DEPTH:
            return True
        return False # Refine the lines a little bit

    # Check for too big vs. fuzzy & even
    verification_engine = config['strategy']['verification'].verification_impl
    for _ in range(TINY_SUBREGIONS):
        tiny_region = np.random.random(size=len(region[0])) * (region[1] - region[0]) + region[0]
        tiny_region = (tiny_region, tiny_region + granularity) # TODO: Get/pick granularity
        verification, adv_example = verification_engine(region, safety_predicate)
        if abs(verification - this[0]) > TREND_EPSILON:
            return False # Too big
    return True # Fuzzy & even

