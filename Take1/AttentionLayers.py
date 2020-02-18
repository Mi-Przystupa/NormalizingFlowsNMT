import torch
import torch.nn as nn
import torch.nn.functional as F

#Modified code from this tutorial:
#https://colab.research.google.com/github/bastings/annotated_encoder_decoder/blob/master/annotated_encoder_decoder.ipynb#scrollTo=yygWWAJ9oBsT
class BahdanauAttention(nn.Module):
    """Implements Bahdanau (MLP) attention"""
    
    def __init__(self, hidden_size, key_size=None, query_size=None):
        super(BahdanauAttention, self).__init__()
        
        # We assume a bi-directional encoder so key_size is 2*hidden_size
        key_size = 2 * hidden_size if key_size is None else key_size
        query_size = hidden_size if query_size is None else query_size

        self.key_layer = nn.Linear(key_size, hidden_size, bias=False)
        self.query_layer = nn.Linear(query_size, hidden_size, bias=False)
        self.energy_layer = nn.Linear(hidden_size, 1, bias=False)
        
        # to store attention scores
        self.alphas = None
        
    def forward(self, query=None, proj_key=None, value=None, mask=None):
        assert mask is not None, "mask is required"

        # We first project the query (the decoder state).
        # The projected keys (the encoder states) were already pre-computated.
        query = self.query_layer(query)
        
        # Calculate scores.
        scores = self.energy_layer(torch.tanh(query + proj_key))
        scores = scores.squeeze(2).unsqueeze(1)
        
        # Mask out invalid positions.
        # The mask marks valid positions so we invert it using `mask & 0`.
        scores.data.masked_fill_(mask == 0, -float('inf'))
        
        # Turn scores to probabilities.
        alphas = F.softmax(scores, dim=-1)
        self.alphas = alphas        
        
        # The context vector is the weighted sum of the values.
        context = torch.bmm(alphas, value)
        
        # context shape: [B, 1, 2D], alphas shape: [B, 1, M]
        return context, alphas


#A (hopefully) implementation of global attention from
#This was my attempt at global attention
class GlobalAttention(nn.Module):

    def __init__(self, s_hid_dim, h_j_hid_dim):
        super(GlobalAttention, self).__init__()
        self.linear =  nn.Linear(s_hid_dim + h_j_hid_dim, 1)
        self.weight = nn.Softmax(dim=1)

    def forward(self, s_i, h, batch_first=True):
        #it is assumed the following:
        #s_i is dimensions of batch x hidden
        #h is either seq x batch x hidden or batch x seq x hidden if batch_first is True
        #makes it so we can run linear over all of them at once

        #TxBxh_dim -> BxTxh_dim
        if not batch_first:
            h = h.permute(1,0,2)
        #B x s_dim -> B x T x s_dim
        s_i = s_i.unsqueeze(1).repeat(1, h.size()[1], 1)

        #create B x T x s_dim + h_dim
        x = torch.cat([s_i, h], dim=2)

        #calculate alpha for each s_i;h_t component
        x = torch.tanh(self.linear(x))
        a = self.weight(x)

        #the dimension are funny,i'll have to think about this...
        result = torch.bmm(a.transpose(1, 2), h)
        return result.squeeze(1)

if __name__ == "__main__":
    T = 3
    b = 194
    h_j = 5
    s_i = 3
    #h = torch.ones(T, b, h_j,requires_grad=True)
    h = torch.arange(0, h_j).float()
    h = h.repeat(T,b, 1)
    h.requires_grad=True
    s = torch.ones(b, s_i, requires_grad=True)
    attn = GlobalAttention(s_i, h_j)

    output = attn(s, h, batch_first=False)
    #output.squeeze().backward()
    output = output.squeeze(1)
    output.sum(dim=0).sum(dim=0).backward()
    print(h.grad)
    print(s.grad)
    print(output)
    print(output.size())

