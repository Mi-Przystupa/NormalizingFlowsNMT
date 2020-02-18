import torch

class LossCompute:
    """A simple loss compute and train function."""

    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm):
        ValueError("You have to subclass this")

    def getOptimizerStateDict(self):
        return self.opt.get_state()
    
    def setOptimizerStateDict(self, state_dict):
        return self.opt.set_state(state_dict)