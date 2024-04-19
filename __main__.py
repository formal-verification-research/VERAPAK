import os
import sys
import traceback
import numpy as np

from config import Config, ConfigError
from verapak.parse_args.tools import parse_args
from algorithm import main
from verapak.utilities.sets import Reporter, DoneInterrupt


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

def save_state(config, reporter):
    pass

def run(config):
    reporter = Reporter()
    try:
        # Run the algorithm
        main(config, reporter)
    except KeyboardInterrupt as e:
        reporter.halt_reason = "keyboard"
    except DoneInterrupt as e:
        pass
    except BaseException as e:
        reporter.halt_reason = "error"
        traceback.print_exception(type(e), e, e.__traceback__)
    
    save_state(config, reporter)

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
    if len(sys.argv) == 1:
        sys.argv = [sys.argv[0], "--help"]
    config = parse_args(sys.argv[1:], prog=sys.argv[0])
    if "error" in config:
        print(f"\033[38;2;255;0;0mERROR: {config['error']}\033[0m")
        write_results(config, None, "error_" + config["error"], 0)
    else:
        try:
            config = Config(config)
        except ConfigError as ex:
            print(ex)
        else: # Valid Config
            for strategy in config["strategy"].values():
                strategy.set_config(config)

            run(config)

            for strategy in config["strategy"].values():
                strategy.shutdown()

