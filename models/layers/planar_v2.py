import math

import torch
import torch.nn as nn
from torch.distributions import constraints
import torch.nn.functional as F

from torch.distributions import Transform
from pyro.distributions.conditional import ConditionalTransformModule
from pyro.distributions.torch_transform import TransformModule
from pyro.distributions.util import copy_docs_from

#code is copy of planar flows of the stable version I was working off of.
#added to my code base because of some bug that isn't in the transformation yet
#link oo original source code: https://github.com/pyro-ppl/pyro/blob/dev/pyro/distributions/transforms/planar.py
#source of changes I made: https://github.com/pyro-ppl/pyro/pull/2192/files#diff-50045618e05b03d2206065a1a9b4dd26

class ConditionedPlanarFlow_v2(Transform):
    def __init__(self, bias=None, u=None, w=None):
        super(ConditionedPlanarFlow_v2, self).__init__(cache_size=1)
        self.bias = bias
        self.u = u
        self.w = w
        self._cached_logDetJ = None

    # This method ensures that torch(u_hat, w) > -1, required for invertibility
    def u_hat(self, u, w):
        alpha = torch.matmul(u.unsqueeze(-2), w.unsqueeze(-1)).squeeze(-1)
        a_prime = -1 + F.softplus(alpha)
        #w.div(w.pow(2).sum(...)) is only line that is really different...
        return u + (a_prime - alpha) * w.div(w.pow(2).sum(dim=-1, keepdim=True))

    def _call(self, x):
        """
        :param x: the input into the bijection
        :type x: torch.Tensor
        Invokes the bijection x => y; in the prototypical context of a TransformedDistribution `x` is a
        sample from the base distribution (or the output of a previous flow)
        """

        # x ~ (batch_size, dim_size, 1)
        # w ~ (batch_size, 1, dim_size)
        # bias ~ (batch_size, 1)
        act = torch.tanh(torch.matmul(self.w.unsqueeze(-2), x.unsqueeze(-1)).squeeze(-1) + self.bias)
        u_hat = self.u_hat(self.u, self.w)
        y = x + u_hat * act

        psi_z = (1. - act.pow(2)) * self.w
        self._cached_logDetJ = torch.log(
            torch.abs(1 + torch.matmul(psi_z.unsqueeze(-2), u_hat.unsqueeze(-1)).squeeze(-1).squeeze(-1)))

        return y

    def _inverse(self, y):
        """
        :param y: the output of the bijection
        :type y: torch.Tensor
        Inverts y => x. As noted above, this implementation is incapable of inverting arbitrary values
        `y`; rather it assumes `y` is the result of a previously computed application of the bijector
        to some `x` (which was cached on the forward call)
        """

        raise KeyError("ConditionalPlanarFlow expected to find key in intermediates cache but didn't")

    def log_abs_det_jacobian(self, x, y):
        """
        Calculates the elementwise determinant of the log jacobian
        """
        return self._cached_logDetJ


class PlanarFlow_v2(ConditionedPlanarFlow_v2, TransformModule):
    """
    A 'planar' normalizing flow that uses the transformation

        :math:`\\mathbf{y} = \\mathbf{x} + \\mathbf{u}\\tanh(\\mathbf{w}^T\\mathbf{z}+b)`

    where :math:`\\mathbf{x}` are the inputs, :math:`\\mathbf{y}` are the outputs, and the learnable parameters
    are :math:`b\\in\\mathbb{R}`, :math:`\\mathbf{u}\\in\\mathbb{R}^D`, :math:`\\mathbf{w}\\in\\mathbb{R}^D` for input
    dimension :math:`D`. For this to be an invertible transformation, the condition
    :math:`\\mathbf{w}^T\\mathbf{u}>-1` is enforced.

    Together with `TransformedDistribution` this provides a way to create richer variational approximations.

    Example usage:

    >>> base_dist = dist.Normal(torch.zeros(10), torch.ones(10))
    >>> plf = PlanarFlow(10)
    >>> pyro.module("my_plf", plf)  # doctest: +SKIP
    >>> plf_dist = dist.TransformedDistribution(base_dist, [plf])
    >>> plf_dist.sample()  # doctest: +SKIP
        tensor([-0.4071, -0.5030,  0.7924, -0.2366, -0.2387, -0.1417,  0.0868,
                0.1389, -0.4629,  0.0986])

    The inverse of this transform does not possess an analytical solution and is left unimplemented. However,
    the inverse is cached when the forward operation is called during sampling, and so samples drawn using
    planar flow can be scored.

    :param input_dim: the dimension of the input (and output) variable.
    :type input_dim: int

    References:

    Variational Inference with Normalizing Flows [arXiv:1505.05770]
    Danilo Jimenez Rezende, Shakir Mohamed

    """

    domain = constraints.real
    codomain = constraints.real
    bijective = True
    event_dim = 1

    def __init__(self, input_dim):
        super(PlanarFlow_v2, self).__init__()

        self.bias = nn.Parameter(torch.Tensor(1,)) #changed this line
        self.u = nn.Parameter(torch.Tensor(input_dim,))
        self.w = nn.Parameter(torch.Tensor(input_dim,))
        self.input_dim = input_dim
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.u.size(0))
        self.w.data.uniform_(-stdv, stdv)
        self.u.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-1, 1)


@copy_docs_from(ConditionalTransformModule)
class ConditionalPlanarFlow_v2(ConditionalTransformModule):
    """
    A conditional 'planar' normalizing flow that uses the transformation

        :math:`\\mathbf{y} = \\mathbf{x} + \\mathbf{u}\\tanh(\\mathbf{w}^T\\mathbf{z}+b)`

    where :math:`\\mathbf{x}` are the inputs with dimension :math:`D`, :math:`\\mathbf{y}` are the outputs,
    and the pseudo-parameters :math:`b\\in\\mathbb{R}`, :math:`\\mathbf{u}\\in\\mathbb{R}^D`, and
    :math:`\\mathbf{w}\\in\\mathbb{R}^D` are the output of a function, e.g. a NN, with input
    :math:`z\\in\\mathbb{R}^{M}` representing the context variable to condition on. For this to be an
    invertible transformation, the condition :math:`\\mathbf{w}^T\\mathbf{u}>-1` is enforced.

    Together with `ConditionalTransformedDistribution` this provides a way to create richer variational
    approximations.

    Example usage:

    >>> from pyro.nn.dense_nn import DenseNN
    >>> input_dim = 10
    >>> context_dim = 5
    >>> batch_size = 3
    >>> base_dist = dist.Normal(torch.zeros(input_dim), torch.ones(input_dim))
    >>> hypernet = DenseNN(context_dim, [50, 50], param_dims=[1, input_dim, input_dim])
    >>> plf = ConditionalPlanarFlow(hypernet)
    >>> z = torch.rand(batch_size, context_dim)
    >>> plf_dist = dist.ConditionalTransformedDistribution(base_dist, [plf]).condition(z)
    >>> plf_dist.sample(sample_shape=torch.Size([batch_size])) # doctest: +SKIP

    The inverse of this transform does not possess an analytical solution and is left unimplemented. However,
    the inverse is cached when the forward operation is called during sampling, and so samples drawn using
    planar flow can be scored.

    :param nn: a function inputting the context variable and outputting a triplet of real-valued parameters
        of dimensions :math:`(1, D, D)`.
    :type nn: callable

    References:
    Variational Inference with Normalizing Flows [arXiv:1505.05770]
    Danilo Jimenez Rezende, Shakir Mohamed

    """

    domain = constraints.real
    codomain = constraints.real
    bijective = True
    event_dim = 1

    def __init__(self, nn):
        super(ConditionalPlanarFlow_v2, self).__init__()
        self.nn = nn

    def condition(self, context):
        bias, u, w = self.nn(context)
        return ConditionedPlanarFlow_v2(bias, u, w)