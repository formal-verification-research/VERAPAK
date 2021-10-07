from .modfgsm import ModFGSM


class FGSM(ModFGSM):

    def __init__(self, grad_func, granularity):
        super().__init__(grad_func=grad_func, granularity=granularity)
