from math import isclose

class PartitioningEngine:
    @classmethod
    def get_config_parameters(cls):
        return []

    @classmethod
    def evaluate_args(cls, args, v, errors):
        pass

    def partition(self, region):
        return self.partition_impl(region)

    def partition_impl(self, region):
        raise NotImplementedError(
            "PartitioningEngine does not implement partition_impl(region)")

    def set_config(self, v):
        # Save the verification engine for later use
        self.ve = v["strategy"]["verification"]

    def shutdown(self):
        pass # Do nothing during shutdown by default

