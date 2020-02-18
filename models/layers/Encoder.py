import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch

class Encoder(nn.Module):
    """Encodes a sequence of word embeddings"""
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.):
        super(Encoder, self).__init__()
        self.num_layers = num_layers
        self.rnn = nn.GRU(input_size, hidden_size, num_layers, 
                          batch_first=True, bidirectional=True, dropout=dropout)
        
    def forward(self, x, mask, lengths, pad_pack=True, hidden=None):
        """
        Applies a bidirectional GRU to sequence of embeddings x.
        The input mini-batch x needs to be sorted by length.
        x should have dimensions [batch, time, dim].

        hidden: if none should behave the same way
        """
        if pad_pack:
            packed = pack_padded_sequence(x, lengths, batch_first=True)
        else:
            packed = x
        output, final = self.rnn(packed, hidden)
        if pad_pack:
            output, _ = pad_packed_sequence(output, batch_first=True)
        else:
            output = output * mask.float().unsqueeze(2) #mask out values beyond length of sequence 

        #FYI, if you try to use final without pad_packing, you're gonna have a bad time
        #Your hidden states for padded sequences will progressively have more uninformative information 
        #because w/o PackedSequence RNNS just keep feeding inputs in and if your padding is a 0 vector
        #you'll just keep dampening the hidden state

        # we need to manually concatenate the final states for both directions
        fwd_final = final[0:final.size(0):2]
        bwd_final = final[1:final.size(0):2]
        final = torch.cat([fwd_final, bwd_final], dim=2)  # [num_layers, batch, 2*dim]

        return output, final