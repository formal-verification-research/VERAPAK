import sys
from utilities.vnnlib_lib import VNNLib
from model_tools.onnx_model import ONNXModel
import numpy as np

def check_counterexamples(model_path, vnnlib_path, npy_path):
    
    a = np.squeeze(np.load(npy_path))
    v = VNNLib(vnnlib_path)
    m = ONNXModel(model_path)

    intended_class = v.mat[0].argmax()

    num_good = 0
    num_bad = 0

    results = []
    
    print("index, classification, within_bounds, is_intended_class")
    for i in range(a.shape[0]):
        classification = m.evaluate(a[i]).argmax()
        within_bounds = (a[i].flatten() >= v.inputs[0]).all() and (a[i].flatten() <= v.inputs[1]).all()
        is_intended_class = classification == intended_class
        print(i, classification, within_bounds, is_intended_class)
        if within_bounds and not is_intended_class:
            num_good += 1
            results.append((True, i, classification, within_bounds, is_intended_class))
        else:
            num_bad += 1
            results.append((False, i, classification, within_bounds, is_intended_class))

    print()
    print("Number of good counterexamples:", num_good)
    print("Number of bad counterexamples:", num_bad)

    return num_good, num_bad, results


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(f"{sys.argv[0]} <.onnx> <.vnnlib> <.npy>")
        sys.exit(1)
    
    check_counterexamples(sys.argv[1], sys.argv[2], sys.argv[3])
