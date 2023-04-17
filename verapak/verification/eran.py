import os
import docker
import numpy as np
from .ve import *

client = docker.from_env()


def _region_to_eran_box(region):
    lb_point = region[0].flatten()
    ub_point = region[1].flatten()
    out = []
    for lb, ub in np.vstack([lb_point, ub_point]).T:
        s = f"[{lb},{ub}]"
        print(s)
        out.append(s)
    return "[" + ",".join(out) + "]"

class ERAN(VerificationEngine):
    @staticmethod
    def get_config_parameters():
        pass

    def __init__(self):
        pass

    def set_safety_predicate(self, safety_predicate):
        self.safety_predicate_file = "safety_predicate.eran"
        path = os.path.join(self.in_folder, self.safety_predicate_file)
        f = open(path, 'w')
        f.write(repr(safety_predicate))
        f.close()

    def verification_impl(self, region, safety_predicate):
        eran_box = _region_to_eran_box(region)

        print("Entering ERAN in Docker:")
        out = self.container.exec_run(f"python3 . --netname /ERAN/in/{self.graph_name} --input_box {eran_box} --output_constraints ERAN/in/{self.safety_predicate_file}")
        print("ERAN in Docker exited")
        print(out)
        global eran_out
        eran_out = out
        return None, None

    def set_config(self, config, data):
        print("ERAN setup")
        graph_path = config["graph"]
        self.in_folder, self.graph_name = os.path.split(graph_path)
        for container in client.containers.list():
            if container.name == "eran":
                container.remove(v=True, force=True)
        self.container = client.containers.run("eran:latest", "/bin/sh", detach=True,
                name="eran", volumes=[self.in_folder + ":/ERAN/in"], tty=True)

    def shutdown(self):
        pass
        #self.container.stop()
        #self.container.remove(v=True)


