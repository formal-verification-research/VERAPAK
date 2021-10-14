from .modfgsm import ModFGSM


class FGSM(ModFGSM):

    def __init__(self, gradient_function, granularity, **kwargs):
        super().__init__(gradient_function=gradient_function, granularity=granularity)
