import torch
import torch.nn as nn
from models.NormalizingFlowsEncoderDecoder import NormalizingFlowsEncoderDecoder
from models.layers.GaussianLayer import GaussLayer
import pyro
from pyro.distributions import Normal, Categorical
from pyro.nn import AutoRegressiveNN
from pyro import poutine

class GenerativeEncoderDecoder(NormalizingFlowsEncoderDecoder):
    def __init__(self, encoder, decoder, src_embed, trg_embed, generator,
     lang_model, lm_generator, inference_network, projection_z, num_layers, z_dim=100, train_lm=True, train_mt=True):
        #this will set parameters for Neural Translation Modeling: P(Y | X, Z , theta)
        super(GenerativeEncoderDecoder, self).__init__(encoder, decoder, src_embed, trg_embed, generator)

        #Neural Language Modeling: P(X | Z, theta)
        self.lang_model = lang_model
        self.lm_generator = lm_generator

        #Encoder for variational distribution q(Z  | X, Y, phi)
        self.inference_network = inference_network

        #projection on latent variable z

        self.projection_z = projection_z

        #misc things I need to keep track of 
        self.z_dim = z_dim
        self.num_layers = num_layers
        self.decodeTarget = True
        self.train_lm, self.train_mt = train_lm, train_mt
        #TODO this is...rather a bit of a hack, but this lm part is unique to GNMT stuff so...maybe it's ok, for now 
        self.src_tok_count = 0

    def meanPool(self, hidden_states, lengths):
        return torch.sum(hidden_states,dim=1) / lengths.float().unsqueeze(1)

    def project(self, z):
        return z if self.projection_z is None else self.projection_z(z)
        
    def encode(self, src, src_mask, src_lengths, pad_pack=True, calc_z=True, deterministic=True):
        #TODO need to add option to use surrogate...not ...that important atm
        X = self.src_embed(src)
        mu_x, sig_x,latent_input = self.inference_network(X, src_mask, src_lengths,pad_pack_x=True) 

        if self.use_latent:
            if deterministic:
                self.z = mu_x
            else:
                self.z = (Normal(mu_x, sig_x).to_event(1)).sample()
            #z is otherwise only used as additional input where as project is for initializing hidden states so has to align with rnn hidden state
            self.z = self.applyFlows(self.z, cond_inp=latent_input )
            self.z_hid = self.project(self.z)
        else:
            self.z =  torch.zeros_like(mu_x)
            self.z_hid = self.project(self.z)

        z_hid = self.resize_z(self.z_hid, 2 * self.num_layers)
        hidden_states, encoder_final = super(GenerativeEncoderDecoder, self).encode(src, src_mask, src_lengths,
         pad_pack=pad_pack, hidden=z_hid)
        return hidden_states, encoder_final

    def decode(self, encoder_hidden, encoder_final, src_mask, trg, trg_mask,
               decoder_hidden=None): 
        assert self.z is not None, "You have to encode a sentence first"
        if self.decodeTarget:
            inputs = self.trg_embed(trg)       
            return self.decoder(inputs, encoder_hidden, encoder_final,
                                src_mask, trg_mask, hidden=decoder_hidden, additional_input=self.z)
        else:
            #in this case src = trg
            if decoder_hidden is None:
                z_x = self.resize_z(self.z_hid, self.num_layers)
                decoder_hidden = z_x
            inputs = self.src_embed(trg)
            return self.lang_model(inputs, src_mask, trg_mask, hidden=decoder_hidden, z=self.z)

    def callGenerator(self, pre_output):
        if self.decodeTarget:
            return self.generator(pre_output)
        else:
            return self.lm_generator(pre_output)

    def setDecodeTarget(self, val):
        self.decodeTarget = val

    def getSRCTokCount(self):
        return self.src_tok_count

    def resize_z(self, z, size):
        #unsize middle dimension
        return z.unsqueeze(0).repeat(size, 1, 1)

    def languageModelOptimization(self, z, z_hid, src, src_lengths, src_input_mask, kl):
        src = src.clone()  #pretty sure that's a bug anyways...
        #need to redo src side as batch doesn't handle it
        #TODO...probably should be handled in rebatch
        src_indx = src[:,:-1] 
        src_trgs = src[:,1:]
        self.src_tok_count = (src_trgs != self.pad_index).data.sum().item()
        src_output_mask = (src_trgs != self.pad_index) #similar to what is done in Batch class for trg
        z_x = self.resize_z(z_hid, self.num_layers)

        inputs = self.getWordEmbeddingsWithWordDropout(self.src_embed, src_indx, src_output_mask)
        _, _, pre_output = self.lang_model(inputs, src_input_mask, src_output_mask, hidden=z_x, z=z)
        logits = self.lm_generator(pre_output)
        logits = logits.contiguous().view(-1, logits.size(-1))
        obs = src_trgs.contiguous().view(-1)
        mask = src_output_mask.contiguous().view(-1)
        try:
            mask = mask.bool()
        except AttributeError as e:
            #do nothing, is a versionining thing to supress a warning
            _ = 0
        
        with poutine.scale(scale=self.get_reconstruction_const(scale=kl)):
            pyro.sample('lm_preds', Categorical(logits=logits).mask(mask),
                obs=obs)

    def translationModelOptimization(self, z, z_hid, src, src_mask, src_lengths, trg, trg_mask, trg_lengths, y_trg, kl):
        #self.num_layers*2 because encoder is bidirectional
        z_hid = self.resize_z(z_hid, self.num_layers*2)

        encoder_hidden, encoder_final = self.encoder(self.src_embed(src), src_mask, src_lengths, hidden=z_hid)
        inputs = self.getWordEmbeddingsWithWordDropout(self.trg_embed, trg, trg_mask)
        #key thing is HERE, i am directly calling our decoder
        _, _, pre_output = self.decoder(inputs, encoder_hidden, encoder_final,
                        src_mask, trg_mask, additional_input=z)
        logits = self.generator(pre_output)
        logits = logits.contiguous().view(-1, logits.size(-1))
        obs = y_trg.contiguous().view(-1)
        mask = trg_mask.contiguous().view(-1)
        try:
            mask = mask.bool()
        except AttributeError as e:
            #do nothing, means it's an older pytorch version
            _ = 0
        #My assumption is this will usually just sum the loss so we need to average it ourselves
        with poutine.scale(scale=self.get_reconstruction_const(scale=kl)): 
            pyro.sample('preds', Categorical(logits=logits).mask(mask),
                obs=obs)

    def getVariationalDistribution(self, src, trg, src_mask, trg_mask, src_lengths, trg_lengths, y_trgs):
        #prettys sure I could abstract the guide stuff into this...
        mu_x, sig_x, latent_input = self.inference_network(self.src_embed(src).detach(), src_mask, src_lengths, pad_pack_x=True)
        self.mu_x, self.sig_x = mu_x, sig_x
        dist = self.getDistribution(mu_x, sig_x, cond_input=latent_input)
        return dist

    def guide(self, src, trg, src_mask, trg_mask, src_lengths, trg_lengths, y_trg, kl=1.0):
        #TODO it may be a good idea to specify which parts I am registering....
        pyro.module("GNMT", self)
        #VAENMT paper says they detach the embeddings

        self.batch_size = len(trg_lengths) 
        #according to a colleague the technical correct way to scale KL is 1 . / size_of_data FYI
        dist = self.getVariationalDistribution(src, trg, src_mask, trg_mask, src_lengths, trg_lengths, y_trg)
        with pyro.plate('data'):
            with poutine.scale(scale=self.get_guide_kl_const(scale=kl)):
                z = pyro.sample('z', dist) 
    def setTrainMT(self, val):
        self.train_mt = val

    def setTrainLM(self, val):
        self.train_lm = val

    def model(self, src, trg, src_mask, trg_mask, src_lengths, trg_lengths,y_trg, kl=1.0):
        #TODO again, maaaaaaybe a good idea to specify which parts I need to update...
        pyro.module("GNMT", self)

        with pyro.plate('data'):

            with poutine.scale(scale=self.get_model_kl_const(scale=kl)): 
                use_flows = False
                dist = self.getDistribution(torch.zeros_like(self.mu_x), torch.ones_like(self.sig_x), cond_input=None, extra_cond=use_flows)
                z = pyro.sample('z', dist)

            z_hid = self.project(z)
            
            #Calculations for translation
            if self.train_mt: 
                self.translationModelOptimization(z, z_hid, src, src_mask, src_lengths, trg, trg_mask, trg_lengths, y_trg, kl)

            #Calculations for language modeling
            #mask not passed in because it's handled differently for the lang modeling
            if self.train_lm:
                self.languageModelOptimization(z, z_hid, src, src_lengths, src_mask, kl)


    
    def aux_model(self, src, trg, src_mask, trg_mask, src_lengths, trg_lengths,y_trg, kl=1.0):
        print("aux model not supported presently")
        #pyro.module("GNMT", self)
        #with pyro.plate('data'):
        #    use_flows = False
        #    dist = self.getDistribution(self.mu_x, self.sig_x, extra_cond=use_flows, cond_input=None)
        #    surrogate = pyro.sample('surrogate', dist)
        
    def aux_guide(self, src, trg, src_mask, trg_mask, src_lengths, trg_lengths, y_trgs, kl=1.0):
        print("aux guide not supported presently")
        #mu_xy, sig_xy, mu_x, sig_x, latent_hidden = self.inference_network(self.src_embed(src).detach(), src_mask, src_lengths,
        #    self.trg_embed(trg).detach(), trg_mask, trg_lengths,pad_pack_x=True, pad_pack_y=False)

        #self.mu_xy, self.sig_xy = mu_xy, sig_xy 
        #self.mu_x, self.sig_x = mu_x, sig_x

        #with pyro.plate('data'):
            #idea of this is we just want the base distribution to match
        #    use_flows = False
        #    dist = self.getDistribution(self.mu_xy.detach(), self.sig_x.detach(), cond_input=latent_hidden, extra_cond=use_flows)
        #    surrogate = pyro.sample('surrogate', dist)
    
    def setTRGSentence(self, trg, trg_mask, trg_lengths):
        self.trg = trg
        self.trg_mask = trg_mask
        self.trg_lengths = trg_lengths

    def get_batch_params(self, ret_posterior=True):
        #this is for evaluation purposes, which is why we clone and detach the tensor
        mu = self.mu_x if ret_posterior else  torch.zeros_like(self.mu_x)
        sig = self.sig_x if ret_posterior else torch.ones_like(self.sig_x) 
        return mu.clone().detach(), sig.clone().detach()