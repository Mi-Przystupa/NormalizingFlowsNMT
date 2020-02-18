import torch.nn as nn
import torch

class DenseLinear(nn.Module):
    """
    An implementation of a simple dense Linaer Layer, for use in, e.g., some conditional flows such as
    :class:`pyro.distributions.transforms.ConditionalPlanarFlow`.

    Mostly the same as DenseNN just...a linear layer

    """

    def __init__(
            self,
            input_dim,
            param_dims=[1, 1]):
        super(DenseLinear, self).__init__()

        self.input_dim = input_dim
        self.param_dims = param_dims
        self.count_params = len(param_dims)
        self.output_multiplier = sum(param_dims)

        # Calculate the indices on the output corresponding to each parameter
        ends = torch.cumsum(torch.tensor(param_dims), dim=0)
        starts = torch.cat((torch.zeros(1).type_as(ends), ends[:-1]))
        self.param_slices = [slice(s.item(), e.item()) for s, e in zip(starts, ends)]

        # Create masked layers
        layers = [nn.Linear(input_dim, self.output_multiplier)]
        self.layers = nn.ModuleList(layers)


    def forward(self, x):
        """
        The forward method
        """
        h = x
        h = self.layers[-1](h)

        # Shape the output, squeezing the parameter dimension if all ones
        if self.output_multiplier == 1:
            return h
        else:
            h = h.reshape(list(x.size()[:-1]) + [self.output_multiplier])

            if self.count_params == 1:
                return h
            else:
                return tuple([h[..., s] for s in self.param_slices])
