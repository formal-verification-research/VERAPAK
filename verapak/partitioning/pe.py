class PartitioningEngine:
    def __init__(self):
        pass

    def partition_impl(self, region):
        raise NotImplementedError(
            "PartitioningEngine does not implement partition_impl(region)")
