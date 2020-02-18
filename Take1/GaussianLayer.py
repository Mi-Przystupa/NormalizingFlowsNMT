import torch.nn as nn
from torch.nn.functional import softplus

class GaussLayer(nn.Module):

    def __init__(self, inputs, hidden_size, latent_z, activation=None):
        super(GaussLayer, self).__init__()
        self.linear = nn.Linear(inputs, hidden_size )
        self.mu = nn.Linear(hidden_size, latent_z)
        self.logsig = nn.Linear(hidden_size, latent_z)
        self.activation = nn.Tanh() if activation is None else activation

    def forward(self, x):
       x = self.activation(self.linear(x))
       mu = self.mu(x)
       sig = softplus(self.logsig(x))
       return mu, sig