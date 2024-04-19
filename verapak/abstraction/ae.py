class AbstractionEngine:
    @classmethod
    def get_config_parameters(cls):
        return []

    @classmethod
    def evaluate_args(cls, args, v, errors):
        pass

    def abstraction_impl(self, region, num_abstractions):
        raise NotImplementedError("AbstractionEngine did not implement abstraction_impl(region, num_abstractions)")

    def set_config(self, v):
        pass # Do nothing with the config by default

    def shutdown(self):
        pass # Do nothing during shutdown by default

