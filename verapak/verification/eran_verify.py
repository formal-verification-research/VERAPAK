import constraint_utils
from eran import ERAN
from read_net_file import read_onnx_net
import numpy as np
import tensorflow as tf

non_layer_operation_types = ['NoOp', 'Assign', 'Const', 'RestoreV2', 'SaveV2', 'PlaceholderWithDefault', 'IsVariableInitialized', 'Placeholder', 'Identity']
def createERANInstance(netname):
    if netname.endswith(".onnx"):
        model, is_conv = read_onnx_net(netname)
        return ERAN(model, is_onnx=True)
    elif netname.endswith(".pb"):
        tf.logging.set_verbosity(tf.logging.ERROR)
        sess = tf.Session()
        with tf.gfile.GFile(netname, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            sess.graph.as_default()
            tf.graph_util.import_graph_def(graph_def, name='')
        ops = sess.graph.get_operations()
        last_layer_index = -1
        while ops[last_layer_index].type in non_layer_operation_types:
            last_layer_index -= 1
        model = sess.graph.get_tensor_by_name(ops[last_layer_index].name + ":0")
        return ERAN(model, sess)
    else:
        raise ValueError("Did not recognize net file extension. Must be either .onnx or .pb")

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print(f"Usage: {sys.argv[0]} netname specLB specUB constraints_file.eran")
    eran = createERANInstance(sys.argv[1])
    specLB = np.array(sys.argv[2].replace('[','').replace(']','').split(','))
    specUB = np.array(sys.argv[3].replace('[','').replace(']','').split(','))
    constraints = constraint_utils.get_constraints_from_file(sys.argv[4])
    percent, nn, nlb, nub, _, _ = eran.analyze_box(specLB, specUB, "deeppoly", 1, 1, True, constraints)
    print(percent)

