import torch
import torch.nn as nn
from models.NormalizingFlowsEncoderDecoder import NormalizingFlowsEncoderDecoder
from models.layers.GaussianLayer import GaussLayer
import pyro
from pyro.distributions import Normal, Categorical
from pyro.nn import AutoRegressiveNN
from pyro import poutine

class VanillaJointEncoderDecoder(NormalizingFlowsEncoderDecoder):
    def __init__(self, encoder, decoder, src_embed, trg_embed, generator,
     lang_model, lm_generator, num_layers, train_lm=True, train_mt=True):
        #this will set parameters for Neural Translation Modeling: P(Y | X, Z , theta)
        super(VanillaJointEncoderDecoder, self).__init__(encoder, decoder, src_embed, trg_embed, generator)

        #Neural Language Modeling: P(X | Z, theta)
        self.lang_model = lang_model
        self.lm_generator = lm_generator
        self.projection_z = None

        #misc things I need to keep track of 
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
        hidden_states, encoder_final = super(VanillaJointEncoderDecoder, self).encode(src, src_mask, src_lengths,
         pad_pack=pad_pack)
        return hidden_states, encoder_final

    def decode(self, encoder_hidden, encoder_final, src_mask, trg, trg_mask,
               decoder_hidden=None): 
        if self.decodeTarget:
            inputs = self.trg_embed(trg)       
            return self.decoder(inputs, encoder_hidden, encoder_final,
                                src_mask, trg_mask, hidden=decoder_hidden)
        else:
            #in this case src = trg
            inputs = self.src_embed(trg)
            return self.lang_model(inputs, src_mask, trg_mask)

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

    def languageModelOptimization(self, src, src_lengths, src_input_mask, kl):
        src = src.clone()  #pretty sure that's a bug anyways...
        #need to redo src side as batch doesn't handle it
        #TODO...probably should be handled in rebatch
        src_indx = src[:,:-1] 
        src_trgs = src[:,1:]
        self.src_tok_count = (src_trgs != self.pad_index).data.sum().item()
        src_output_mask = (src_trgs != self.pad_index) #similar to what is done in Batch class for trg

        inputs = self.getWordEmbeddingsWithWordDropout(self.src_embed, src_indx, src_output_mask)
        _, _, pre_output = self.lang_model(inputs, src_input_mask, src_output_mask)
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

    def translationModelOptimization(self, src, src_mask, src_lengths, trg, trg_mask, trg_lengths, y_trg, kl):
        encoder_hidden, encoder_final = self.encoder(self.src_embed(src), src_mask, src_lengths)
        inputs = self.getWordEmbeddingsWithWordDropout(self.trg_embed, trg, trg_mask)
        #key thing is HERE, i am directly calling our decoder
        _, _, pre_output = self.decoder(inputs, encoder_hidden, encoder_final,
                        src_mask, trg_mask)
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
        self.mu_x = torch.zeros_like(src).float()
        self.sig_x = torch.ones_like(src).float()

        dist = self.getDistribution(self.mu_x, self.sig_x, extra_cond=False, cond_input=self.mu_x)
        return dist

    def guide(self, src, trg, src_mask, trg_mask, src_lengths, trg_lengths, y_trg, kl=1.0):
        pyro.module("GNMT", self)
        self.mu_x = torch.zeros_like(src).float()
        self.sig_x = torch.ones_like(src).float()
        self.batch_size = len(trg_lengths) 

    def setTrainMT(self, val):
        self.train_mt = val

    def setTrainLM(self, val):
        self.train_lm = val

    def model(self, src, trg, src_mask, trg_mask, src_lengths, trg_lengths,y_trg, kl=1.0):
        pyro.module("GNMT", self)

        with pyro.plate('data'):
            #Calculations for translation
            if self.train_mt: 
                self.translationModelOptimization(src, src_mask, src_lengths, trg, trg_mask, trg_lengths, y_trg, kl)

            #Calculations for language modeling
            if self.train_lm:
                self.languageModelOptimization(src, src_lengths, src_mask, kl)


    
    def aux_model(self, src, trg, src_mask, trg_mask, src_lengths, trg_lengths,y_trg, kl=1.0):
        print("aux model not supported presently")
        
    def aux_guide(self, src, trg, src_mask, trg_mask, src_lengths, trg_lengths, y_trgs, kl=1.0):
        print("aux guide not supported presently")
    
    def setTRGSentence(self, trg, trg_mask, trg_lengths):
        self.trg = trg
        self.trg_mask = trg_mask
        self.trg_lengths = trg_lengths

    def get_batch_params(self, ret_posterior=True):
        #this is for evaluation purposes, which is why we clone and detach the tensor
        mu = self.mu_x if ret_posterior else  torch.zeros_like(self.mu_x)
        sig = self.sig_x if ret_posterior else torch.ones_like(self.sig_x) 
        return mu.clone().detach(), sig.clone().detach()