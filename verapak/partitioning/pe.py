from math import isclose

class PartitioningEngine:
    @classmethod
    def get_config_parameters(cls):
        return []

    @classmethod
    def evaluate_args(cls, args, v, errors):
        pass

    def partition(self, region):
        subregions = self.partition_impl(region)
        return self.setup_partition_data(self, region, subregions)

    def setup_partition_data(self, region, subregions):
        percent = self.ve.get_percent(region)
        siblings_equal = True
        for subregion in subregions:
            # Create a new RegionData for the child
            subregion.data = region.data.make_child()

            # Grab (and cache) the confidence
            new_percent = self.ve.get_percent(subregion)

            # Check sibling equality by comparing to parent (Tolerance: 2 decimal points in scientific notation)
            if siblings_equal and not isclose(percent, new_percent, rel_tol=1e-2):
                siblings_equal = False

        for subregion in subregions:
            # Set sibling equal value
            subregion.data.siblings_equal = siblings_equal

        return subregions

    def partition_impl(self, region):
        raise NotImplementedError(
            "PartitioningEngine does not implement partition_impl(region)")

    def set_config(self, v):
        # Save the verification engine for later use
        self.ve = v["strategy"]["verification"]

    def shutdown(self):
        pass # Do nothing during shutdown by default

