import numpy as np
import os
import time
import queue
import sys
from verapak.model_tools import model_base
from verapak.parse_arg_tools import parse_args
from verapak.utilities.point_tools import *
import tensorflow as tf
import verapak_utils
from verapak import snap
from verapak.verification.ve import *


def setup(config):
    graph_path = config["graph"]
    config["graph"] = model_base.load_graph_by_type(graph_path, 'ONNX')
    config["input_shape"] = config["graph"].input_shape
    config["input_dtype"] = config["graph"].input_dtype
    config["output_shape"] = config["graph"].output_shape
    config["output_dtype"] = config["graph"].output_dtype

    if len(config["granularity"]) == 1:
        config['granularity'] = np.full(
            config['input_shape'], config['granularity'][0]).astype(config['input_dtype'])
    else:
        config['granularity'] = np.array(
            config['granularity'], dtype=config['input_dtype']).reshape(config['input_shape'])

    config["initial_point"] = np.array(
        config["initial_point"], dtype=config["input_dtype"]).reshape(config["input_shape"])

    if config["label"] is None:
        config["label"] = np.argmax(
            config["graph"].evaluate(config["initial_point"]).flatten())

    config['label'] = tf.keras.utils.to_categorical(
        config['label'], num_classes=np.prod(config['output_shape']), dtype=config['output_dtype'])

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


def main(config):
    try:
        setup(config)
        start_time = time.time()
        unknown_set = verapak_utils.RegionSet()
        adversarial_examples = verapak_utils.PointSet()
        unsafe_set = queue.Queue()
        safe_set = verapak_utils.RegionSet()

        label_classification = np.argmax(config['label'].flatten())

        def safety_predicate(point):
            classification = np.argmax(
                config['graph'].evaluate(point).flatten())
            return classification == label_classification

        def snap_point_to_domain_and_valid(point):
            return snap.point_to_domain(snap.to_nearest_valid_point(point, config['initial_point'], config['granularity']), config['domain'])

        num_valid_points_in_initial_region = get_amount_valid_points(
            config['initial_region'], config['granularity'], config['initial_point'])

        num_valid_points_in_unsafe_set = 0
        num_valid_points_in_unknown_set = 0
        num_valid_points_in_safe_set = 0

        if not safety_predicate(config['initial_point']):
            adversarial_examples.insert(config['initial_point'])
            unsafe_set.put((config['initial_region'], config['initial_point']))
            num_valid_points_in_unsafe_set += num_valid_points_in_initial_region
        else:
            unknown_set.insert(*config['initial_region'])
            num_valid_points_in_unknown_set += num_valid_points_in_initial_region

        elapsed_time = (time.time() - start_time) / 60.0
        use_timeout = config['timeout_minutes'] != 0

        first_time_flag = False

        def report_first_adversarial_example():
            nonlocal first_time_flag
            if adversarial_examples.size() <= 0:
                return
            if first_time_flag:
                return
            first_time_flag = True
            et = (time.time() - start_time)
            print(f'found first adversarial example in {et} seconds')
            for adv_example in adversarial_examples.elements():
                print(adv_example.shape)

        def partition_unsafe_region(region, adv_example):
            nonlocal num_valid_points_in_unsafe_set
            nonlocal num_valid_points_in_unknown_set
            partition = config['partitioning_strategy'].partition_impl(
                region)
            for r in partition:
                num_valid_points_in_r = get_amount_valid_points(
                    r, config['granularity'], config['initial_point'])
                if point_in_region(r, adv_example):
                    unsafe_set.put_nowait((r, adv_example))
                    num_valid_points_in_unsafe_set += num_valid_points_in_r
                else:
                    unknown_set.insert(*r)
                    num_valid_points_in_unknown_set += num_valid_points_in_r

        last_report_time = time.time()

        def report_region_percentages(bypass=False):
            nonlocal last_report_time
            et = time.time() - last_report_time
            if et < config['report_interval_seconds'] and not bypass:
                return
            percent_unsafe = num_valid_points_in_unsafe_set / \
                num_valid_points_in_initial_region * 100
            percent_safe = num_valid_points_in_safe_set / \
                num_valid_points_in_initial_region * 100
            percent_unknown = num_valid_points_in_unknown_set / \
                num_valid_points_in_initial_region * 100
            print(f'percent unknown: {percent_unknown}%')
            print(f'percent found unsafe: {percent_unsafe}%')
            print(f'percent found safe: {percent_safe}%')
            print('number adversarial examples', adversarial_examples.size())

            percent_left_over = 100 - percent_unknown - percent_unsafe - percent_safe
            if percent_left_over > 0:
                print(f'currently processing: {percent_left_over}%')
            last_report_time = time.time()

        while (unknown_set.size() > 0 or not unsafe_set.empty()) and (not use_timeout or elapsed_time < config['timeout_minutes']):
            report_first_adversarial_example()
            report_region_percentages()

            region, adv_example = [unknown_set.pop_front(
            )[1], None] if unknown_set.size() > 0 else unsafe_set.get_nowait()
            region = [x.reshape(config['input_shape']).astype(
                config['input_dtype']) for x in region]

            num_valid_points = get_amount_valid_points(
                region, config['granularity'], config['initial_point'])

            if adv_example is None:
                num_valid_points_in_unknown_set -= num_valid_points
            else:
                num_valid_points_in_unsafe_set -= num_valid_points

            if num_valid_points <= 0:
                continue

            if not adv_example is None:
                if num_valid_points > 1:
                    partition_unsafe_region(region, adv_example)
                elif num_valid_points == 1:
                    num_valid_points_in_unsafe_set += 1
                continue

            verif_result, adv_example = config['verification_strategy'].verification_impl(
                region, safety_predicate)

            if verif_result == SAFE:
                num_valid_points_in_safe_set += num_valid_points
                safe_set.insert(*region)
                continue
            elif verif_result == UNSAFE:
                adversarial_examples.insert(adv_example)
                if num_valid_points > 1:
                    partition_unsafe_region(region, adv_example)
                continue

            partition = config['partitioning_strategy'].partition_impl(region)
            for r in partition:
                abstraction = config['abstraction_strategy'].abstraction_impl(
                    r, config['num_abstractions'])
                snapped_abstraction = [
                    snap_point_to_domain_and_valid(x) for x in abstraction]
                found_adv_in_r = False
                num_valid_points_in_r = get_amount_valid_points(
                    r, config['granularity'], config['initial_point'])
                abstraction_evaluated = [(p, safety_predicate(p))
                                         for p in snapped_abstraction]
                for point, safe in abstraction_evaluated:
                    if not safe:
                        adversarial_examples.insert(point)
                for point, safe in abstraction_evaluated:
                    if not safe:
                        if point_in_region(r, point):
                            num_valid_points_in_unsafe_set += num_valid_points_in_r
                            unsafe_set.put_nowait((r, point))
                            found_adv_in_r = True
                            break
                        else:  # attempt to find the potential adversarial example in unknown region set
                            find_success, potential_region = unknown_set.get_and_remove_region_containing_point(
                                point)
                            if find_success:
                                potential_region = [x.reshape(config['input_shape']).astype(
                                    config['input_dtype']) for x in potential_region]
                                num_valid_points_in_pr = get_amount_valid_points(
                                    potential_region, config['granularity'], config['initial_point'])
                                num_valid_points_in_unsafe_set += num_valid_points_in_pr
                                num_valid_points_in_unknown_set -= num_valid_points_in_pr
                                unsafe_set.put_nowait(
                                    (potential_region, point))
                if not found_adv_in_r:
                    num_valid_points_in_unknown_set += num_valid_points_in_r
                    unknown_set.insert(r)

            elapsed_time = (time.time() - start_time) / 60.0
    except (KeyboardInterrupt, Exception) as e:
        print(e)

    print('\n')
    print('Final Report')
    print('############################')
    report_region_percentages(True)

    adv_examples_numpy = np.array([x for x in adversarial_examples.elements()])
    output_file = os.path.join(
        config['output_dir'], 'adversarial_examples.npy')
    print(f'saving adversarial examples to "{output_file}"...')
    np.save(output_file, adv_examples_numpy)
    print('done')


if __name__ == "__main__":
    config = parse_args(sys.argv[1:], prog=sys.argv[0])
    main(config)
