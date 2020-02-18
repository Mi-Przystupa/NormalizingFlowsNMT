import torch
import torch.nn as nn
from models.NormalizingFlowsEncoderDecoder import NormalizingFlowsEncoderDecoder
from models.layers.GaussianLayer import GaussLayer
import pyro
from pyro.distributions import Normal, Categorical, Bernoulli
from pyro.nn import AutoRegressiveNN
from pyro import poutine


class VanillaEncoderDecoder(NormalizingFlowsEncoderDecoder):

    def __init__(self, encoder, decoder, src_embed, trg_embed, generator):
        super(VanillaEncoderDecoder, self).__init__(encoder, decoder, src_embed, trg_embed, generator)
        #I included this because for the guide / model these would all be redundant calculations
        self.encoder_hidden_x = None
        self.encoder_final = None
        self.X_avg = None
        self.z = None
        #again, super hacky but this is just for use in eval
        self.posterior_params = {}
        self.prior_params = {}

    def encode(self, src, src_mask, src_lengths, pad_pack=True, deterministic=True):
        hidden_states, encoder_final = super(VanillaEncoderDecoder, self).encode(src, src_mask, src_lengths, pad_pack=pad_pack)        
        return hidden_states, encoder_final 
    def decode(self, encoder_hidden, encoder_final, src_mask, trg, trg_mask,
               decoder_hidden=None):
        inputs = self.trg_embed(trg)      
        return self.decoder(inputs, encoder_hidden, encoder_final,
                            src_mask, trg_mask, hidden=decoder_hidden)
    
    def meanPool(self, hidden_states, lengths):
        return torch.sum(hidden_states,dim=1) / lengths.float().unsqueeze(1)
    
    def encodeHiddenMeanPool(self, embeds, wrd_mask, wrd_lengths, pad_pack=True):
        enc_hidden, enc_final = self.encoder(embeds, wrd_mask, wrd_lengths, pad_pack=pad_pack)
        enc_mean = self.meanPool(enc_hidden, wrd_lengths)
        return enc_hidden, enc_final, enc_mean

    def guide(self, src, trg, src_mask, trg_mask, src_lengths, trg_lengths, y_trgs, kl=1.0):
        pyro.module('VanillaNMT', self)
        #basically do nothing
        self.posterior_params = {'mu': torch.zeros_like(src).float(), 'sig': torch.ones_like(src).float()} #this is for evaluation of KL later
        self.batch_size = len(trg_lengths) 
        
    def model(self, src, trg, src_mask, trg_mask, src_lengths, trg_lengths,y_trg, kl=1.0 ):
        pyro.module('VanillaNMT', self)
        self.encoder_hidden_x, self.encoder_final = self.encoder(self.src_embed(src), src_mask, src_lengths)
        encoder_hidden, encoder_final = self.encoder_hidden_x, self.encoder_final

        with pyro.plate('data'):
            #for consistency, although word dropout ...supposedly makes less sense with out latent variables
            inputs = self.getWordEmbeddingsWithWordDropout(self.trg_embed, trg, trg_mask)
            #key thing is HERE, i am directly calling our decoder
            _, _, pre_output = self.decoder(inputs, encoder_hidden, encoder_final,
                            src_mask, trg_mask)
            logits = self.generator(pre_output)
            obs = y_trg.contiguous().view(-1)
            mask = trg_mask.contiguous().view(-1)
            try:
                mask = mask.bool()
            except AttributeError as e:
                #do nothing, is just a versioning issue
                _ = 0
            #My assumption is this will usually just sum the loss so we need to average it ourselves
            with poutine.scale(scale=self.get_reconstruction_const(scale=kl)): 
                pyro.sample('preds', Categorical(logits=logits.contiguous().view(-1, logits.size(-1))).mask(mask),
                    obs=obs)            

    def aux_guide(self, src, trg, src_mask, trg_mask, src_lengths, trg_lengths, y_trgs, kl=1.0):
        pyro.module('VanillaNMT', self)
        #do nothing
        
    def aux_model(self, src, trg, src_mask, trg_mask, src_lengths, trg_lengths, y_trgs, kl=1.0): 
        pyro.module('VanillaNMT', self)
        #do nothing

    def get_batch_params(self, ret_posterior=True):
        #this is for evaluation purposes, which is why we clone and detach the tensor
        #this is meaningless in the vanilla model case
        modes = self.posterior_params  
        return modes['mu'].clone().detach(), modes['sig'].clone().detach()
            
    def forward(self, src, trg, src_mask, trg_mask, src_lengths, trg_lengths):
        """Take in and process masked src and target sequences."""
        encoder_hidden, encoder_final = self.encode(src, src_mask, src_lengths)
        return self.decode(encoder_hidden, encoder_final, src_mask, trg, trg_mask)