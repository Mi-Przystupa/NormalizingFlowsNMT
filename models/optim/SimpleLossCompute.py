import torch
from models.optim.LossCompute import LossCompute

class SimpleLossCompute(LossCompute):
    """A simple loss compute and train function."""

    def __init__(self, generator, criterion, opt=None):
        super(SimpleLossCompute, self).__init__(generator, criterion, opt)

    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1))
        loss = loss / norm

        if self.opt is not None:
            loss.backward()          
            self.opt.step()
            self.opt.zero_grad()

        return loss.data.item() * norm