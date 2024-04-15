import os
import io
import math
import docker
import numpy as np
import tarfile
from .ve import *

# TODO: FIX DOCKER
client = None#docker.from_env()

# TODO: Check if ERAN on its own works - where does UNKNOWN/TOO_BIG come from?
class ERAN(VerificationEngine):
    @classmethod
    def get_config_parameters(cls):
        return [{
            "name": "eran_timeout",
            "arg_params": {
                "type": "float",
                "help": "ERAN timeout. 0 for no timeout, negative timeout is a multiple of the full timeout",
                "default": -0.1
            }
        }]

    @classmethod
    def evaluate_args(cls, args, v, errors):
        v["eran_timeout"] = args["eran_timeout"]
        if v["eran_timeout"] < 0:
            v["eran_timeout"] = -v["eran_timeout"] / v["timeout"]
        v["eran_timeout"] = math.ceil(v["eran_timeout"])

    def __init__(self):
        pass

    def _write_bytes(self, b, name, cleanup=True):
        with tarfile.open(name + '.tar', 'w') as tar:
            tinfo = tarfile.TarInfo(name)
            tinfo.size = len(b)
            tar.addfile(tinfo, io.BytesIO(b))
        with open(name + '.tar', 'rb') as f:
            self.container.put_archive("/ERAN/in", f.read())
        if cleanup:
            os.remove(name + ".tar")
    def _write_string(self, s, name, cleanup=True):
        self._write_bytes(s.encode('utf-8'), name, cleanup=cleanup)

    def set_safety_predicate(self, safety_predicate):
        self.safety_predicate_file = "safety_predicate.eran"
        self._write_string(repr(safety_predicate), self.safety_predicate_file)
        self.unsafety_predicate_file = "unsafety_predicate.eran"
        self._write_string(repr(~safety_predicate), self.unsafety_predicate_file)

    def verification_impl(self, region, safety_predicate):
        r0 = str(list(region[0].flatten())).replace(' ', '')
        r1 = str(list(region[1].flatten())).replace(' ', '')
        print("Proving with ERAN in Docker")
        is_good = self.container.exec_run(f"python3 eran_verify.py /ERAN/in/{self.graph_name} {r0} {r1} /ERAN/in/{self.safety_predicate_file}")
        print("Docker exited", end='')
        if is_good.output.endswith(b"\nsat\n"):
            print(": Completely SAFE")
            return ALL_SAFE, None
        elif is_good.output.endswith(b"\nunsat\n"):
            print("\nDisproving with ERAN in Docker")
            is_bad = self.container.exec_run(f"python3 eran_verify.py /ERAN/in/{self.graph_name} {r0} {r1} /ERAN/in/{self.unsafety_predicate_file}")
            print("Docker exited")
            if is_bad.output.endswith(b"\nsat\n"):
                print("Completely UNSAFE")
                return ALL_UNSAFE, None
            else:
                print("Some UNSAFE")
                return SOME_UNSAFE, None
        else:
            return TOO_BIG, None

    def set_config(self, v):
        graph_path = v["graph"]
        self.in_folder, self.graph_name = os.path.split(graph_path)
        try:
            client.containers.get("eran").remove(v=True, force=True)
        except docker.errors.NotFound:
            pass
        self.container = client.containers.run("yodarocks1/eran:latest", "/bin/sh", detach=True,
                name="eran", tty=True)
        self.container.exec_run("mkdir /ERAN/in")
        with open(graph_path, 'rb') as f:
            self._write_bytes(f.read(), self.graph_name)

        with open("/src/verapak/verification/eran_verify.py", 'rb') as f:
            self._write_bytes(f.read(), "eran_verify.py")
        self.container.exec_run("cp /ERAN/in/eran_verify.py /ERAN/tf_verify/eran_verify.py")

        self.timeout = v["eran_timeout"]

        self.set_safety_predicate(v["safety_predicate"])

    def shutdown(self):
        self.container.stop()
        self.container.remove(v=True)


