class PartitioningEngine:
    @classmethod
    def get_config_parameters(cls):
        return []

    @classmethod
    def evaluate_args(cls, v):
        return v

    def partition_impl(self, region):
        raise NotImplementedError(
            "PartitioningEngine does not implement partition_impl(region)")

    def set_config(self, config, data):
        pass # Do nothing with the config by default

    def shutdown(self):
        pass # Do nothing during shutdown by default

