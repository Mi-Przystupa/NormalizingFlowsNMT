import torch
import torch.nn as nn
import logging

class Decoder(nn.Module):
    """A conditional RNN decoder with attention."""
    
    def __init__(self, emb_size, hidden_size, attention, num_layers=1, dropout=0.5,
                 bridge=True, additional_input_dim=0, embed_concat=False, rnn_input_concat=False, pool_size=2):
        """
            embed_size: size of word embedding
            hidden_size: size of hidden states in GRU
            attention: Attention mechanism used
            num_layers: number of layers of RNN
            dropout: dropout used IN RNN & on pre_output layer
            bridge: a transform for an input hidden state to make sure dimensions line up
            embed_concat: if additional inputs given, concat on word embeddings (change embed size to match)
            rnn_input_concat: concat additional input only on feeding to RNN
            pool_size: has to do with max out layer that...people for some reason forget to mention
        """
        super(Decoder, self).__init__()
        #either both False, or only 1 is true
        assert (not embed_concat and not rnn_input_concat) or (embed_concat != rnn_input_concat), "Tried to set embed_concat and rnn_input_concat both true"
        if embed_concat or rnn_input_concat:
            assert additional_input_dim > 0, "Your additional input dimensions must be bigger than 0 if using embed_concat or rnn_input_concat"
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.attention = attention
        self.dropout = dropout
        self.embed_concat = embed_concat
        self.rnn_input_concat = rnn_input_concat

        if not self.rnn_input_concat and not self.embed_concat:
            logging.warning("You are not using rnn input concat or embed concat, additional inputs to model are ignored")

        context_dim = 2*hidden_size if self.attention is not None else 0
        self.rnn_dims = emb_size + context_dim + (additional_input_dim if embed_concat or rnn_input_concat else 0)
        self.pre_input_dims = hidden_size + context_dim + emb_size + (additional_input_dim if embed_concat else 0)

        self.rnn = nn.GRU(self.rnn_dims, hidden_size, num_layers,
                          batch_first=True, dropout=dropout)
                 
        # to initialize from the final encoder state
        self.bridge = nn.Linear(2*hidden_size, hidden_size, bias=True) if bridge else None

        self.dropout_layer = nn.Dropout(p=dropout)
        
        self.pre_output_layer = nn.Linear(self.pre_input_dims,
                                          pool_size * hidden_size, bias=False)
        
    def forward_step(self, prev_embed, encoder_hidden, src_mask, proj_key, hidden, additional_rnn_input=None):
        """Perform a single decoder step (1 word)"""

        # compute context vector using attention mechanism
        if self.attention is not None:
            query = hidden[-1].unsqueeze(1)  # [#layers, B, D] -> [B, 1, D]
            context, attn_probs = self.attention(
                query=query, proj_key=proj_key,
                value=encoder_hidden, mask=src_mask)
            rnn_input = torch.cat([prev_embed, context, ], dim=2)
        else:
            context = None
            rnn_input = prev_embed
            

        # update rnn hidden state
        if additional_rnn_input is not None:
            #assumes additional_rnn_input: [B, D] -> [B, 1, D]
            rnn_input = torch.cat([rnn_input, additional_rnn_input], dim=2)

        output, hidden = self.rnn(rnn_input, hidden)
        if context is not None: 
            pre_output = torch.cat([prev_embed, output, context], dim=2)
        else:
            pre_output = torch.cat([prev_embed, output], dim=2)
        pre_output = self.dropout_layer(pre_output)
        pre_output = self.pre_output_layer(pre_output)

        return output, hidden, pre_output
    
    def forward(self, trg_embed, encoder_hidden, encoder_final, 
                src_mask, trg_mask, hidden=None, max_len=None, additional_input=None):
        """Unroll the decoder one step at a time."""
                                         
        # the maximum number of steps to unroll the RNN
        if max_len is None:
            max_len = trg_mask.size(-1)

        # initialize decoder hidden state
        if hidden is None:
            hidden = self.init_hidden(encoder_final)

        #logic for rnn business
        rnn_additional_inputs = None
        if additional_input is not None:
            trg_size = trg_embed.size(1)
            additional_input = additional_input.unsqueeze(1).repeat(1, trg_size, 1)
            if self.embed_concat:
                #assumes trg_embed: [B, T, D] & additional input: [B, D]
                #you want additional input included in pre_output layer
                trg_embed = torch.cat([trg_embed, additional_input], dim=2)
            elif self.rnn_input_concat:
                #you want it just in rnn
                rnn_additional_inputs = additional_input
        
        # pre-compute projected encoder hidden states
        # (the "keys" for the attention mechanism)
        # this is only done for efficiency
        if self.attention is not None:
            proj_key = self.attention.key_layer(encoder_hidden) 
        else:
            proj_key = None
        
        # here we store all intermediate hidden states and pre-output vectors
        decoder_states = []
        pre_output_vectors = []
        
        # unroll the decoder RNN for max_len steps
        for i in range(max_len):
            
            prev_embed = trg_embed[:, i].unsqueeze(1)
            #makes it easier to debug this way...
            rnn_additional_input = None
            if rnn_additional_inputs is not None:
                rnn_additional_input = rnn_additional_inputs[:, i].unsqueeze(1)
            output, hidden, pre_output = self.forward_step(
              prev_embed, encoder_hidden, src_mask, proj_key, hidden, additional_rnn_input=rnn_additional_input)
            decoder_states.append(output)
            pre_output_vectors.append(pre_output)

        decoder_states = torch.cat(decoder_states, dim=1)
        pre_output_vectors = torch.cat(pre_output_vectors, dim=1)
        return decoder_states, hidden, pre_output_vectors  # [B, N, D]

    def init_hidden(self, encoder_final):
        """Returns the initial decoder state,
        conditioned on the final encoder state."""

        if encoder_final is None:
            return None  # start with zeros

        return torch.tanh(self.bridge(encoder_final))