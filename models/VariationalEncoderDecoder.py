import torch
import torch.nn as nn
from models.NormalizingFlowsEncoderDecoder import NormalizingFlowsEncoderDecoder
from models.layers.GaussianLayer import GaussLayer
import pyro
from pyro.distributions import Normal, Categorical, Bernoulli
from pyro.nn import AutoRegressiveNN
from pyro import poutine


class VariationalEncoderDecoder(NormalizingFlowsEncoderDecoder):

    def __init__(self, encoder, decoder, src_embed, trg_embed, generator, prior, posterior, projection):
        super(VariationalEncoderDecoder, self).__init__(encoder, decoder, src_embed, trg_embed, generator)
        self.prior = prior
        self.posterior = posterior
        self.projection = projection
        #I included this because for the guide / model these would all be redundant calculations
        self.encoder_hidden_x = None
        self.encoder_final = None
        self.X_avg = None
        self.z = None
        #again, super hacky but this is just for use in eval
        self.posterior_params = {}
        self.prior_params = {}

    def encode(self, src, src_mask, src_lengths, pad_pack=True, deterministic=True):
        hidden_states, encoder_final = super(VariationalEncoderDecoder, self).encode(src, src_mask, src_lengths, pad_pack=pad_pack)        
        X = self.meanPool(hidden_states, src_lengths)
        mu, sig = self.prior(X) 
        if self.use_latent:
            if deterministic:
                self.z = mu
            else:
                self.z = Normal(mu, sig).sample()

            if len(self.nf) > 0:
                self.z = self.applyFlows(self.z, cond_inp=X)
                
            self.z = self.z if self.projection is None else self.projection(self.z)
        else:
            self.z = torch.zeros_like(mu) # do nothing
            self.z = self.z if self.projection is None else self.projection(self.z)
        return hidden_states, encoder_final 
    def decode(self, encoder_hidden, encoder_final, src_mask, trg, trg_mask,
               decoder_hidden=None):
        assert self.z is not None, "You have to encode a sentence first"
        z = self.z 
        inputs = self.trg_embed(trg)      
        return self.decoder(inputs, encoder_hidden, encoder_final,
                            src_mask, trg_mask, hidden=decoder_hidden, additional_input=z)
    
    def meanPool(self, hidden_states, lengths):
        return torch.sum(hidden_states,dim=1) / lengths.float().unsqueeze(1)
    
    def encodeHiddenMeanPool(self, embeds, wrd_mask, wrd_lengths, pad_pack=True):
        enc_hidden, enc_final = self.encoder(embeds, wrd_mask, wrd_lengths, pad_pack=pad_pack)
        enc_mean = self.meanPool(enc_hidden, wrd_lengths)
        return enc_hidden, enc_final, enc_mean
    
    def getVariationalDistribution(self, src, trg, src_mask, trg_mask, src_lengths, trg_lengths, y_trgs):
        self.encoder_hidden_x, self.encoder_final, self.X_avg =  \
            self.encodeHiddenMeanPool(self.src_embed(src), src_mask, src_lengths)
        X = self.X_avg
        latent_inputs = None
        if self.posterior is not None:
            _, _, Y = self.encodeHiddenMeanPool(self.trg_embed(trg), trg_mask, trg_lengths, pad_pack=False)
            latent_inputs = torch.cat([X, Y], dim=1)   
            z_mean, z_sig = self.posterior(latent_inputs)
        else:
            #if posterior is None, assumes we are just sampling directly from prior
            latent_inputs = X
            z_mean, z_sig = self.prior(latent_inputs)
        self.posterior_params = {'mu': z_mean, 'sig': z_sig} #this is for evaluation of KL later
        use_transform_if_can=True
        dist = self.getDistribution(z_mean, z_sig, extra_cond=use_transform_if_can, cond_input=X)
        return dist

    def guide(self, src, trg, src_mask, trg_mask, src_lengths, trg_lengths, y_trgs, kl=1.0):
        pyro.module('VNMT', self)

        #Using a posterior distribution is for original VNMT, 
        self.batch_size = len(trg_lengths) 
        dist = self.getVariationalDistribution(src, trg, src_mask, trg_mask, src_lengths, trg_lengths, y_trgs)
        with pyro.plate('data'):
            with poutine.scale(scale = self.get_guide_kl_const(scale=kl)):
                z = pyro.sample('z', dist) #to_event on a gaussian treats it as a "Multivariate" gaussian w/ diag covar
    
        
    def model(self, src, trg, src_mask, trg_mask, src_lengths, trg_lengths,y_trg, kl=1.0 ):
        pyro.module('VNMT', self)
        encoder_hidden, encoder_final = self.encoder_hidden_x, self.encoder_final
        X = self.X_avg

        if self.posterior is not None:
            #regular VNMT
            z_mean, z_sig = self.prior(X)
        else:
            #match our...own parameters, should just mean KL(...) = 0 ery time
            mu_post, sig_post =  self.get_batch_params(ret_posterior=True)
            z_mean, z_sig = mu_post, sig_post

        self.prior_params = {'mu': z_mean, 'sig': z_sig}
        with pyro.plate('data'):
            #TODO FYI: technically, the correct scaling is 1./ size_of_data
            with poutine.scale(scale=self.get_model_kl_const(scale=kl)): 
                #TODO probably...a good idea to test this with flows also on prior...you know, so it's correct?
                use_flows = True
                dist = self.getDistribution(z_mean, z_sig, use_cached_flows=True, extra_cond=use_flows, cond_input=None)
                z = pyro.sample('z', dist)
            #TODO, need to add the latent z as input to decoder

            z = z if self.projection is None else self.projection(z)

            inputs = self.getWordEmbeddingsWithWordDropout(self.trg_embed, trg, trg_mask)
            #key thing is HERE, i am directly calling our decoder
            _, _, pre_output = self.decoder(inputs, encoder_hidden, encoder_final,
                            src_mask, trg_mask, additional_input=z)
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
        pyro.module('VNMT', self)
        self.encoder_hidden_x, self.encoder_final = self.encoder(self.src_embed(src), src_mask, src_lengths)
        encoder_hidden_y, _ = self.encoder(self.trg_embed(trg), trg_mask, trg_lengths, pad_pack=False)

        #mean pooling operation
        X = self.meanPool(self.encoder_hidden_x, src_lengths)
        self.X_avg = X
        Y = self.meanPool(encoder_hidden_y, trg_lengths)
        
        z = torch.cat([X, Y], dim=1)
        #TODO there...might need to be another layer in here
        z_mean, z_sig = self.posterior(z)
        
        self.batch_size = len(trg_lengths) 
        with pyro.plate('data_eval'):
            with poutine.scale(scale= 1.0):
                use_transform_if_can=True
                dist = self.getDistribution(z_mean, z_sig, extra_cond=use_transform_if_can, cond_input=X)
                z = pyro.sample('z_eval', dist) #to_event on a gaussian treats it as a "Multivariate" gaussian w/ diag covar

    def aux_model(self, src, trg, src_mask, trg_mask, src_lengths, trg_lengths, y_trgs, kl=1.0): 
        pyro.module('VNMT', self)
        encoder_hidden, encoder_final = self.encoder_hidden_x, self.encoder_final

        X = self.X_avg
        z_mean, z_sig = self.prior(X)
        with pyro.plate('data_eval'):
            with poutine.scale(scale=1.0): 
                    dist = Normal(z_mean, z_sig).to_event(1)
                    z = pyro.sample('z_eval', dist)

    def get_batch_params(self, ret_posterior=True):
        #this is for evaluation purposes, which is why we clone and detach the tensor
        modes = self.posterior_params if ret_posterior else self.prior_params
        return modes['mu'].clone().detach(), modes['sig'].clone().detach()
            
    def forward(self, src, trg, src_mask, trg_mask, src_lengths, trg_lengths):
        """Take in and process masked src and target sequences."""
        encoder_hidden, encoder_final = self.encode(src, src_mask, src_lengths)
        return self.decode(encoder_hidden, encoder_final, src_mask, trg, trg_mask)
