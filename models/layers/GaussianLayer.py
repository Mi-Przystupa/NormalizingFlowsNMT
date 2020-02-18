import torch.nn as nn
from torch.nn.functional import softplus

class GaussLayer(nn.Module):

    def __init__(self, inputs, hidden_size, latent_z, do_projection=True, sigma_act=None, activation=None):
        super(GaussLayer, self).__init__()
        hidden_size = hidden_size if do_projection else inputs
        self.linear = nn.Linear(inputs, hidden_size ) if do_projection else None
        self.mu = nn.Linear(hidden_size, latent_z)
        self.logsig = nn.Linear(hidden_size, latent_z)
        self.activation = nn.Tanh() if activation is None else activation
        self.sigma_act = softplus if sigma_act is None else sigma_act

    def forward(self, x):
        #in GNMT, the authors seem to just directly pass concatenated vectors to model
        if self.linear is not None:
            #this is equation 8 in VNMT paper
            x = self.activation(self.linear(x))
        #these are eq 9 in VNMT paper
        mu = self.mu(x)

        sig = self.sigma_act(self.logsig(x)) #in GNMT they used exp
        return mu, sig