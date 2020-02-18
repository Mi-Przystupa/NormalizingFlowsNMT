import torch
import torch.nn as nn
from torch.distributions import constraints

from pyro.distributions.torch_transform import TransformModule
from pyro.distributions.util import copy_docs_from
from pyro.distributions.transforms.utils import clamp_preserve_gradients

from pyro.distributions.conditional import ConditionalTransformModule
from pyro.distributions.torch_transform import TransformModule
#cannot stress this enough, most of this is copied form the PYRO implementation, the edits are just my tweeks so I condition on some input
#link to what they name it now : http://docs.pyro.ai/en/latest/distributions.html#affineautoregressive
#link to doc in pyro where this is from: http://docs.pyro.ai/en/0.5.0/_modules/pyro/distributions/transforms/iaf.html#InverseAutoregressiveFlowStable
#again, most of this is NOT my own code, i just add the conditioning
class CondInverseAutoregressiveFlowStable(TransformModule, ConditionalTransformModule):
    """
    An implementation of an Inverse Autoregressive Flow, using Eqs (13)/(14) from Kingma Et Al., 2016,

        :math:`\\mathbf{y} = \\sigma_t\\odot\\mathbf{x} + (1-\\sigma_t)\\odot\\mu_t`

    where :math:`\\mathbf{x}` are the inputs, :math:`\\mathbf{y}` are the outputs, :math:`\\mu_t,\\sigma_t`
    are calculated from an autoregressive network on :math:`\\mathbf{x}`, and :math:`\\sigma_t` is
    restricted to :math:`(0,1)`.

    This variant of IAF is claimed by the authors to be more numerically stable than one using Eq (10),
    although in practice it leads to a restriction on the distributions that can be represented,
    presumably since the input is restricted to rescaling by a number on :math:`(0,1)`.
    """

    domain = constraints.real
    codomain = constraints.real
    bijective = True
    event_dim = 1

    def __init__(self, cond_autoregressive_nn, sigmoid_bias=2.0):
        super(CondInverseAutoregressiveFlowStable, self).__init__(cache_size=1)
        self.arn = cond_autoregressive_nn
        self.sigmoid = nn.Sigmoid()
        self.logsigmoid = nn.LogSigmoid()
        self.sigmoid_bias = sigmoid_bias
        self._cached_log_scale = None

    def _call(self, x):
        """
        :param x: the input into the bijection
        :type x: torch.Tensor

        Invokes the bijection x=>y; in the prototypical context of a TransformedDistribution `x` is a
        sample from the base distribution (or the output of a previous flow)
        """
        mean, logit_scale = self.arn(x, context=self.context)
        logit_scale = logit_scale + self.sigmoid_bias
        scale = self.sigmoid(logit_scale)
        log_scale = self.logsigmoid(logit_scale)
        self._cached_log_scale = log_scale

        y = scale * x + (1 - scale) * mean
        return y

    def _inverse(self, y):
        """
        :param y: the output of the bijection
        :type y: torch.Tensor

        Inverts y => x.
        """
        x_size = y.size()[:-1]
        perm = self.arn.permutation
        input_dim = y.size(-1)
        x = [torch.zeros(x_size, device=y.device)] * input_dim

        # NOTE: Inversion is an expensive operation that scales in the dimension of the input
        for idx in perm:
            mean, logit_scale = self.arn(torch.stack(x, dim=-1), context=self.context)
            inverse_scale = 1 + torch.exp(-logit_scale[..., idx] - self.sigmoid_bias)
            x[idx] = inverse_scale * y[..., idx] + (1 - inverse_scale) * mean[..., idx]
            self._cached_log_scale = inverse_scale

        x = torch.stack(x, dim=-1)
        return x

    def log_abs_det_jacobian(self, x, y):
        """
        Calculates the elementwise determinant of the log jacobian
        """
        if self._cached_log_scale is not None:
            log_scale = self._cached_log_scale
        else:
            _, logit_scale = self.arn(x, context=self.context)
            log_scale = self.logsigmoid(logit_scale + self.sigmoid_bias)
        return log_scale.sum(-1)
    
    def condition(self, context):
        #really the only change to original code
        self.context = context
        return self