from .ae import AbstractionEngine

class FallbackStrategy(AbstractionEngine):
    
    def __init__(self, when, what):
        self._should_fallback = when
        self._what = what

    def should_fallback(self, region, num_abstractions):
        return self._should_fallback(region, num_abstractions)

    def abstraction_impl(self, region, num_abstractions):
        self._what(region, num_abstractions)
    
    def set_config(self, config):
        pass # Does not need any info from config