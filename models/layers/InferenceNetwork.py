import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch

class SourceInferenceNetwork(nn.Module):
    """Encodes 2 sequences and generates sets of parameters based on requested values"""
    def __init__(self, src_size, hidden_size, dist_x, use_avg=True, num_layers=1, dropout=0.):
        super(SourceInferenceNetwork, self).__init__()
        self.num_layers = num_layers

        self.src_rnn = nn.GRU(src_size, hidden_size, num_layers, 
                          batch_first=True, bidirectional=True, dropout=dropout)
        
        self.use_avg = use_avg

        self.X = None

        self.dist_x = dist_x

    def meanPool(self, hidden_states, lengths):
        return torch.sum(hidden_states,dim=1) / lengths.float().unsqueeze(1)

    def generate_output(self, embeds, lengths, mask, network, pad_pack, ):
        if pad_pack:
            packed = pack_padded_sequence(embeds, lengths, batch_first=True)
        else:
            packed = embeds

        output, final = network(packed)

        if pad_pack:
            output, _ = pad_packed_sequence(output, batch_first=True)
        else:
            if len(mask.size()) < 3:
                #this ...is weird...
                mask = mask.unsqueeze(2)
            output = output * mask.float()

        fwd_final = final[0:final.size(0):2]
        bwd_final = final[1:final.size(0):2]
        final = torch.cat([fwd_final, bwd_final], dim=2)  # [num_layers, batch, 2*dim]

        return output, final
        
    def forward(self, x, x_mask, x_lengths, pad_pack_x=True):
        """
        Applies a bidirectional GRU to sequence of embeddings x.
        The input mini-batch x needs to be sorted by length.
        x should have dimensions [batch, time, dim].
        """
         #mask out values beyond length of sequence 

        #FYI, if you try to use final without pad_packing, you're gonna have a bad time
        #Your hidden states for padded sequences will progressively have more uninformative information 
        #because w/o PackedSequence RNNS just keep feeding inputs in and if your padding is a 0 vector
        #you'll just keep dampening the hidden state
        x_output, x_final = self.generate_output(x, x_lengths, x_mask, self.src_rnn, pad_pack_x)

        if self.use_avg:
            X = self.meanPool(x_output, x_lengths)
        else:
            X = x_final.squeeze(0)

        # we need to manually concatenate the final states for both directions
        mu_x, sig_x = self.dist_x(X)
        
        return mu_x, sig_x, X

class SourceTargetInferenceNetwork(nn.Module):
    """Encodes 2 sequences and generates sets of parameters based on requested values"""
    def __init__(self, src_size, trg_size, hidden_size, dist_xy, dist_x, use_avg=True, share_params=False, num_layers=1, dropout=0.):
        super(SourceTargetInferenceNetwork, self).__init__()
        self.num_layers = num_layers

        self.src_rnn = nn.GRU(src_size, hidden_size, num_layers, 
                          batch_first=True, bidirectional=True, dropout=dropout)

        self.trg_rnn = nn.GRU(trg_size, hidden_size, num_layers, 
                          batch_first=True, bidirectional=True, dropout=dropout) if not share_params else self.src_rnn
        self.use_avg = use_avg

        self.X = None
        self.Y = None

        self.dist_xy = dist_xy 
        self.dist_x = dist_x

    def meanPool(self, hidden_states, lengths):
        return torch.sum(hidden_states,dim=1) / lengths.float().unsqueeze(1)

    def generate_output(self, embeds, lengths, mask, network, pad_pack, ):
        if pad_pack:
            packed = pack_padded_sequence(embeds, lengths, batch_first=True)
        else:
            packed = embeds

        output, final = network(packed)


        if pad_pack:
            output, _ = pad_packed_sequence(output, batch_first=True)
        else:
            if len(mask.size()) < 3:
                #this ...is weird...
                mask = mask.unsqueeze(2)
            output = output * mask.float()

        fwd_final = final[0:final.size(0):2]
        bwd_final = final[1:final.size(0):2]
        final = torch.cat([fwd_final, bwd_final], dim=2)  # [num_layers, batch, 2*dim]

        return output, final
        
    def forward(self, x, x_mask, x_lengths, y, y_mask, y_lengths, pad_pack_x=True, pad_pack_y=False):
        """
        Applies a bidirectional GRU to sequence of embeddings x.
        The input mini-batch x needs to be sorted by length.
        x should have dimensions [batch, time, dim].
        """
         #mask out values beyond length of sequence 

        #FYI, if you try to use final without pad_packing, you're gonna have a bad time
        #Your hidden states for padded sequences will progressively have more uninformative information 
        #because w/o PackedSequence RNNS just keep feeding inputs in and if your padding is a 0 vector
        #you'll just keep dampening the hidden state
        x_output, x_final = self.generate_output(x, x_lengths, x_mask, self.src_rnn, pad_pack_x)
        y_output, y_final = self.generate_output(y, y_lengths, y_mask, self.trg_rnn, pad_pack_y)

        if self.use_avg:
            X = self.meanPool(x_output, x_lengths)
            Y = self.meanPool(y_output, y_lengths)
        else:
            X = x_final
            Y = y_final

        XY = torch.cat([X, Y], dim=1)

        # we need to manually concatenate the final states for both directions
        mu_xy, sig_xy = self.dist_xy(XY)        
        mu_x, sig_x = self.dist_x(X)
        
        return mu_xy, sig_xy, mu_x, sig_x, XY