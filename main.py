import numpy as np
import os
import time
import queue
import sys
import traceback
from verapak.model_tools import model_base
from verapak.parse_arg_tools import parse_args
from verapak.utilities.point_tools import *
from verapak.utilities.vnnlib_lib import NonMaximalVNNLibError
import tensorflow as tf
import verapak_utils
from verapak import snap
from verapak.verification.ve import *

class DoneInterrupt(Exception):
    pass

def setup(config):
    graph_path = config["graph"]
    config["graph"] = model_base.load_graph_by_type(graph_path, 'ONNX')
    config["input_shape"] = config["graph"].input_shape
    config["input_dtype"] = config["graph"].input_dtype
    config["output_shape"] = config["graph"].output_shape
    config["output_dtype"] = config["graph"].output_dtype

    config["initial_point"] = np.array(
        config["initial_point"], dtype=config["input_dtype"]).reshape(config["input_shape"])

    if config["label"] is None:
        config["label"] = np.argmax(
            config["graph"].evaluate(config["initial_point"]).flatten())

    config['label'] = tf.keras.utils.to_categorical(
        config['label'], num_classes=np.abs(np.prod(config['output_shape'])), dtype=config['output_dtype'])

    if len(config["radius"]) == 1:
        config["radius"] = np.full_like(
            config["initial_point"], config["radius"][0])
    else:
        config["radius"] = np.array(config["radius"]).reshape(
            config["input_shape"]).astype(config["input_dtype"])

    def gradient_function(point):
        return config['graph'].gradient_of_loss_wrt_input(point, config['label'])

    config['gradient_function'] = gradient_function
    config['abstraction_strategy'] = config['abstraction_strategy'](**config)

    if config['partitioning_num_dimensions'] > config['initial_point'].size:
        config['partitioning_num_dimensions'] = config['initial_point'].size

    config['partitioning_strategy'] = config['partitioning_strategy'](**config)
    config['dimension_ranking_strategy'] = config['dimension_ranking_strategy'](
        **config)

    config['verification_strategy'] = config['verification_strategy'](**config)
    if len(config['domain_upper_bound']) == 1:
        dub = np.full_like(
            config['initial_point'], config['domain_upper_bound'][0])
    else:
        dub = np.array(config['domain_upper_bound']).reshape(
            config['input_shape']).astype(config['input_dtype'])

    if len(config['domain_lower_bound']) == 1:
        dlb = np.full_like(
            config['initial_point'], config['domain_lower_bound'][0])
    else:
        dlb = np.array(config['domain_lower_bound']).reshape(
            config['input_shape']).astype(config['input_dtype'])

    config['domain'] = [dlb, dub]
    config['initial_region'] = snap.region_to_domain([config['initial_point'] - config['radius'],
                                                      config['initial_point'] + config['radius']], config['domain'])
    # for i in range(0,config['initial_region'][0].size):
    #    reg_idx = np.unravel_index(i, config['initial_region'][0].shape)
    #    if (config['initial_region'][1][reg_idx] - config['initial_region'][0][reg_idx]) == 0.0:
    #        print(str(reg_idx) + "\t\t:\t\t" + str(config['radius'][reg_idx]) + "\t\t:\t\t" + str(config['initial_point'][reg_idx]))


def main(config):
    unknown_set = verapak_utils.RegionSet()
    adversarial_examples = verapak_utils.PointSet()
    unsafe_set = queue.Queue()
    safe_set = verapak_utils.RegionSet()

    start_time = time.time()

    try:
        setup(config)

        label_classification = np.argmax(config['label'].flatten())

        def safety_predicate(point):
            classification = np.argmax(
                config['graph'].evaluate(point).flatten())
            return classification == label_classification

        def snap_point_to_domain_and_valid(point):
            return snap.point_to_domain(point, config['domain'])
        
        initial_area = get_size(config['initial_region'])
        assert initial_area > 0, "Input region is empty"

        unsafe_area = 0
        unknown_area = 0
        safe_area = 0

        start_time = time.time()

        if not safety_predicate(config['initial_point']):
            adversarial_examples.insert(config['initial_point'])
            unsafe_set.put((config['initial_region'], config['initial_point']))
            unsafe_area = initial_area
        else:
            unknown_set.insert(*config['initial_region'])
            unknown_area = initial_area

        elapsed_time = time.time() - start_time
        use_timeout = config['timeout'] != 0

        first_time_flag = False

        def report_first_adversarial_example():
            nonlocal first_time_flag
            if adversarial_examples.size() <= 0:
                return
            if first_time_flag:
                return
            first_time_flag = True
            et = time.time() - start_time
            print(f'found first adversarial example in {et} seconds')
            output_file = os.path.join(config['output_dir'], 'time_to_first.txt')
            print(f'saving first adversarial example time to "{output_file}"...')
            output_file = open(output_file, "w")
            output_file.write(f"{et} seconds\n")
            output_file.close()
            if config['halt_on_first']:
                halt_reason = "done"
                raise DoneInterrupt()

        def partition_unsafe_region(region, adv_example):
            nonlocal unsafe_area
            nonlocal unknown_area
            partition = config['partitioning_strategy'].partition_impl(
                region)
            for r in partition:
                if point_in_region(r, adv_example):
                    unsafe_set.put_nowait((r, adv_example))
                    unsafe_area += get_size(r)
                else:
                    unknown_set.insert(*r)
                    unknown_area += get_size(r)

        last_report_time = time.time()

        def report_region_percentages(bypass=False):
            nonlocal last_report_time
            et = time.time() - last_report_time
            if et < config['report_interval_seconds'] and not bypass:
                return
            percent_unsafe = unsafe_area / initial_area * 100
            percent_safe = safe_area / initial_area * 100
            percent_unknown = unknown_area / initial_area * 100
            print(f'percent unknown: {percent_unknown}%')
            print(f'percent found unsafe: {percent_unsafe}%')
            print(f'percent found safe: {percent_safe}%')
            print('number adversarial examples', adversarial_examples.size())

            percent_left_over = 100 - percent_unknown - percent_unsafe - percent_safe
            if percent_left_over > 0:
                print(f'currently processing: {percent_left_over}%')
            last_report_time = time.time()
            print()

        while (unknown_set.size() > 0 or not unsafe_set.empty()) and (not use_timeout or elapsed_time < config['timeout']):
            print("Loop")
            report_first_adversarial_example()
            report_region_percentages()

            region, adv_example = [unknown_set.pop_random(
            )[1], None] if unknown_set.size() > 0 else unsafe_set.get_nowait()
            region = [x.reshape(config['input_shape']).astype(
                config['input_dtype']) for x in region]

            region_area = get_size(region)

            if adv_example is None:
                unknown_area -= region_area
            else:
                unsafe_area -= region_area

            if region_area <= 0:
                elapsed_time = time.time() - start_time
                continue

            if not adv_example is None:
                partition_unsafe_region(region, adv_example)
                elapsed_time = time.time() - start_time
                continue

            verif_result, adv_example = config['verification_strategy'].verification_impl(
                region, safety_predicate)

            if verif_result == SAFE:
                safe_area += region_area
                safe_set.insert(*region)
            elif verif_result == UNSAFE:
                adversarial_examples.insert(adv_example)
                partition_unsafe_region(region, adv_example)
            else:
                partition = config['partitioning_strategy'].partition_impl(region)
                for r in partition:
                    abstraction = config['abstraction_strategy'].abstraction_impl(
                        r, config['num_abstractions'])
                    snapped_abstraction = [
                        snap_point_to_domain_and_valid(x) for x in abstraction]
                    found_adv_in_r = False
                    r_area = get_size(r)
                    abstraction_evaluated = [(p, safety_predicate(p))
                                             for p in snapped_abstraction]
                    for point, safe in abstraction_evaluated:
                        if not safe:
                            adversarial_examples.insert(point)
                    for point, safe in abstraction_evaluated:
                        if not safe:
                            if point_in_region(r, point):
                                unsafe_area += r_area
                                unsafe_set.put_nowait((r, point))
                                found_adv_in_r = True
                                break
                            else:  # attempt to find the potential adversarial example in unknown region set
                                find_success, potential_region = unknown_set.get_and_remove_region_containing_point(
                                    point)
                                if find_success:
                                    potential_region = [x.reshape(config['input_shape']).astype(
                                        config['input_dtype']) for x in potential_region]
                                    pr_area = get_size(potential_region)
                                    unsafe_area += pr_area
                                    unknown_area -= pr_area
                                    unsafe_set.put_nowait(
                                        (potential_region, point))
                    if not found_adv_in_r:
                        unknown_area += r_area
                        unknown_set.insert(r[0], r[1])
            elapsed_time = time.time() - start_time
        if use_timeout and elapsed_time >= config['timeout']:
            halt_reason = "timeout"
        else:
            halt_reason = "done"
    except KeyboardInterrupt as e:
        halt_reason = "halted"
        print(e)
    except DoneInterrupt as e:
        halt_reason = "done"
    except BaseException as e:
        halt_reason = "error"
        traceback.print_exception(type(e), e, e.__traceback__)

    print('\n')
    print('Final Report')
    print('############################')
    if "report_region_percentages" in locals():
        report_region_percentages(True)

    et = time.time() - start_time
    if adversarial_examples.size() > 0:
        first_adversarial = next(adversarial_examples.elements())
        witness_data = [
            first_adversarial.flatten(),
            config['graph'].evaluate(first_adversarial).flatten()
        ]
    else:
        witness_data = None
    write_results(config['output_dir'], adversarial_examples, halt_reason, et, witness_data)
    print('done')

def get_size(region):
    return abs(np.prod(region[1] - region[0]))

def create_witness(input_values, output_values):
    witness = "("
    for idx, x in np.ndenumerate(input_values):
        witness += f"(X_{idx[0]} {x}); " # Semicolons should become newlines for VNNCOMP, but CSV can't store literal newline characters.
    for idx, y in np.ndenumerate(output_values):
        witness += f"(Y_{idx[0]} {y}); " # Semicolons should become newlines for VNNCOMP, but CSV can't store literal newline characters.
    witness += ")"
    return witness

def write_results(output_dir, adversarial_examples, halt_reason, et, witness_data):
    witness = ""
    adv_count = 0
    if witness_data:
        witness = create_witness(witness_data[0], witness_data[1])
    if adversarial_examples:
        adv_count = adversarial_examples.size()
        adv_examples_numpy = np.array([x for x in adversarial_examples.elements()])
        output_file = os.path.join(
                output_dir, 'adversarial_examples.npy')
        print(f'saving adversarial examples to "{output_file}"...')
        np.save(output_file, adv_examples_numpy)

    if halt_reason == "done":
        halt_reason = "sat" if (adv_count > 0) else "unsat"

    output_file = os.path.join(output_dir, 'report.csv')
    print(f'saving report to "{output_file}"...')
    output_file = open(output_file, 'w')
    output_file.write(f"{halt_reason},{witness},{adv_count},{et}\n")
    output_file.close()

if __name__ == "__main__":
    config = parse_args(sys.argv[1:], prog=sys.argv[0])
    if "error" in config:
        print(f"\033[38;2;255;0;0mERROR: {config['error']}\033[0m")
        write_results(config['output_dir'], None, "error_" + config["error"], 0, None)
    else:
        main(config)
