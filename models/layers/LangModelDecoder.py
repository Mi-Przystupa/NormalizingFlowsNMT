import torch
import torch.nn as nn

class LangModelDecoder(nn.Module):
    """A conditional RNN decoder with attention."""
    
    def __init__(self, emb_size, hidden_size, num_layers=1, dropout=0.5, z_dim=100, preoutput_concat=False):
        super(LangModelDecoder, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.z_dim = z_dim
        self.dropout = dropout
        self.preoutput_concat = preoutput_concat

        self.rnn = nn.GRU(emb_size + z_dim, hidden_size, num_layers,
                          batch_first=True, dropout=dropout)
                 
        # to initialize from the final encoder state
        self.dropout_layer = nn.Dropout(p=dropout)
        addit_inp_dim = z_dim if preoutput_concat else 0        
        #output 2x's hidden_size because of max out layerr
        self.pre_output_layer = nn.Linear(hidden_size + addit_inp_dim,
                                          hidden_size , bias=False)
        
    def forward_step(self, prev_embed, src_mask, hidden, additional_preout_inp=None):
        """Perform a single decoder step (1 word)"""

        # compute context vector using attention mechanism
        # update rnn hidden state
        rnn_input = prev_embed # there is no context
        output, hidden = self.rnn(rnn_input, hidden)
        
        if additional_preout_inp is not None and self.preoutput_concat:
            pre_output = torch.cat([output, additional_preout_inp], dim=2)
        else:
            pre_output = output
        pre_output = self.dropout_layer(pre_output)
        pre_output = self.pre_output_layer(pre_output)

        return output, hidden, pre_output
    
    def forward(self, trg_embed, src_mask, trg_mask, hidden=None, max_len=None, z=None):
        """Unroll the decoder one step at a time."""
                                         
        # the maximum number of steps to unroll the RNN
        if max_len is None:
            max_len = trg_mask.size(-1)
                
        rnn_input = trg_embed
        if z is not None:
            #assumes additional_rnn_input: [B, D] -> [B, 1, D]
            trg_size = trg_embed.size(1)
            z_rnn_in = z.unsqueeze(1).repeat(1, trg_size, 1)
            rnn_input = torch.cat([rnn_input, z_rnn_in], dim=2)
            if len(z.size()) < 3:
                #convert (assumed) [B x D] => B x 1 X D
                z = z.unsqueeze(1)

       
        # here we store all intermediate hidden states and pre-output vectors
        decoder_states = []
        pre_output_vectors = []
        
        # unroll the decoder RNN for max_len steps
        for i in range(max_len):
            
            prev_embed = rnn_input[:, i].unsqueeze(1)
            output, hidden, pre_output = self.forward_step(
              prev_embed, src_mask, hidden, additional_preout_inp=z)
            decoder_states.append(output)
            pre_output_vectors.append(pre_output)

        decoder_states = torch.cat(decoder_states, dim=1)
        pre_output_vectors = torch.cat(pre_output_vectors, dim=1)
        return decoder_states, hidden, pre_output_vectors  # [B, N, D]
