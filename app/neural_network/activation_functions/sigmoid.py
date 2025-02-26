import numpy as np

class SigmoidFunction:
    @staticmethod
    def function(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def backward(x):
        sig = SigmoidFunction.function(x)
        return sig * (1 - sig)