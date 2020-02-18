import torch.nn as nn
import torch.nn.functional as F

class Maxout(nn.Module):
    #source of this class : https://github.com/pytorch/pytorch/issues/805
    def __init__(self, pool_size):
        super().__init__()
        self._pool_size = pool_size

    def forward(self, x):
        #we use the version 2nd from last on the list. 
        #assumed shape of tensors: [B, T, H], 
        #B = batch size, T = Longest sequence in batch, H is hidden dimensions
        #H is the dimension we want to maxout over
        assert x.shape[-1] % self._pool_size == 0, \
            'Wrong input last dim size ({}) for Maxout({})'.format(x.shape[1], self._pool_size)
        #m, i = x.view(*x.shape[:1], x.shape[1] // self._pool_size, self._pool_size, *x.shape[2:]).max(2)
        m, i = x.view(*x.shape[:-1], x.shape[-1] // self._pool_size, self._pool_size).max(-1)
        return m

class Generator(nn.Module):
    """Define standard linear + softmax generation step."""
    def __init__(self, hidden_size, vocab_size, pool_size=2, use_maxout=True):
        super(Generator, self).__init__()
        self.max_out = Maxout(pool_size) if use_maxout else None
        self.proj = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, x):
        if self.max_out is not None: 
            x = self.max_out(x)
        return F.log_softmax(self.proj(x), dim=-1)