import math

from torch.distributions import kl_divergence, register_kl

from pyro.distributions.delta import Delta
from pyro.distributions.distribution import Distribution
from pyro.distributions.torch import Independent, MultivariateNormal, Normal
from pyro.distributions.util import sum_rightmost
from torch.distributions.transformed_distribution import TransformedDistribution

from torch.distributions.utils import _sum_rightmost

@register_kl(TransformedDistribution, TransformedDistribution)
def _kl_transformed_transformed(p, q):
    if p.transforms != q.transforms:
        raise NotImplementedError
    if p.event_shape != q.event_shape:
        raise NotImplementedError
    extra_event_dim = len(p.event_shape) - len(p.base_dist.event_shape)
    base_kl_divergence = kl_divergence(p.base_dist, q.base_dist)
    return _sum_rightmost(base_kl_divergence, extra_event_dim)
