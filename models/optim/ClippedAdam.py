from __future__ import absolute_import, division, print_function

import math

import torch
from torch.optim.optimizer import Optimizer
from torch.nn.utils import clip_grad_norm_

#this is literally just a copy from the the pyro version
#only difference is that we clip the gradient with the norm instead of just clamping it
class ClippedAdam(Optimizer):
    """
    :param params: iterable of parameters to optimize or dicts defining parameter groups
    :param lr: learning rate (default: 1e-3)
    :param Tuple betas: coefficients used for computing
        running averages of gradient and its square (default: (0.9, 0.999))
    :param eps: term added to the denominator to improve
        numerical stability (default: 1e-8)
    :param weight_decay: weight decay (L2 penalty) (default: 0)
    :param clip_norm: magnitude of norm to which gradients are clipped (default: 10.0)
    :param lrd: rate at which learning rate decays (default: 1.0)

    Small modification to the Adam algorithm implemented in torch.optim.Adam
    to include gradient clipping and learning rate decay.

    Reference

    `A Method for Stochastic Optimization`, Diederik P. Kingma, Jimmy Ba
    https://arxiv.org/abs/1412.6980
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, clip_norm=10.0, lrd=1.0):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay,
                        clip_norm=clip_norm, lrd=lrd)
        super(ClippedAdam, self).__init__(params, defaults)

    def step(self, closure=None):
        """
        :param closure: An optional closure that reevaluates the model and returns the loss.

        Performs a single optimization step.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            group['lr'] *= group['lrd']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                clip_grad_norm_(p, group['clip_norm'])  #only line that is different from current pyro version 
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(grad)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(grad)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg, denom)

        return loss
