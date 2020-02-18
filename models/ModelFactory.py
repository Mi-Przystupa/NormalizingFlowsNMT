from models.layers.BahdanauAttention import BahdanauAttention
from models.layers.Decoder import Decoder
from models.layers.Encoder import Encoder
from models.layers.Generator import Generator
from models.layers.GaussianLayer import GaussLayer
from models.layers.LangModelDecoder import LangModelDecoder
from models.layers.InferenceNetwork import SourceTargetInferenceNetwork, SourceInferenceNetwork 
from models.EncoderDecoder import EncoderDecoder
from models.VariationalEncoderDecoder import VariationalEncoderDecoder
from models.GenerativeEncoderDecoder import GenerativeEncoderDecoder
from models.VanillaEncoderDecoder import VanillaEncoderDecoder
from models.VanillaJointEncoderDecoder import VanillaJointEncoderDecoder
import torch
import torch.nn as nn
import numpy as np

class ModelFactory:
    def __init__(self, src_vocab, tgt_vocab, type='nmt', emb_size=256, hidden_size=512, 
            num_layers=1, dropout=0.1,z_layer=100, pool_size=2, use_cuda=False, use_projection=True):
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_cuda = use_cuda
        self.z_layer = z_layer
        self.pool_size = pool_size
        self.use_projection = use_projection

    def getModel(self, type, use_attention=True):
        if use_attention:
            attention = BahdanauAttention(self.hidden_size)
        else:
            #this...may not actually work...
            attention = None
        if type == 'nmt':
           model = self._createNMT(attention) 
        if type == 'vanilla_nmt':
            model = self._createVanillaNMT(attention) #same as nmt, but with pyro
        elif type == 'vanilla_joint_nmt':
            model = self._createVanillaJointNMT(attention)
        elif type == 'vnmt':
            model = self._createOriginalVNMT(attention)
        elif type == 'mod_vnmt':
            model = self._createModVNMT(attention)
        elif type == 'simple_mod_vnmt':
            model = self._createSimpleModVNMT(attention)
        elif type == 'vaenmt':
            model = self._createVAENMT(attention)
        elif type =='vaenmt_no_lm':
            model = self._createVAENMT(attention)
            model.setTrainLM(False)
        elif type == 'mod_vaenmt':
            model = self._createModVAENMT(attention)
        elif type == 'mod_vaenmt_no_lm':
            model = self._createModVAENMT(attention)
            model.setTrainLM(False)
        elif type == 'gnmt':
            model = self._createGNMT(attention, None)
        else:
            assert False, "Invalid model choice {}".format(type)

        return model.cuda() if self.use_cuda else model

    def getGenerator(self, vocab_size = -1, use_maxout=True):
        if vocab_size == -1:
            vocab_size = self.tgt_vocab
        return Generator(self.hidden_size, vocab_size, pool_size=self.pool_size, use_maxout=use_maxout)

    def getProjection(self, out_dim=-1):
        #if out dim < 0, assume we are just doing a "change of basis" of z
        out_dim = self.z_layer if out_dim <= 0 else out_dim
        return nn.Sequential(nn.Linear(self.z_layer, out_dim), nn.Tanh()) if self.use_projection else None

    def _createVanillaNMT(self, attention):
        decoder = Decoder(self.emb_size, self.hidden_size, attention,
         num_layers=self.num_layers, dropout=self.dropout)

        return VanillaEncoderDecoder(
                Encoder(self.emb_size, self.hidden_size, num_layers=self.num_layers, dropout=self.dropout),
                decoder,
                nn.Embedding(self.src_vocab, self.emb_size),
                nn.Embedding(self.tgt_vocab, self.emb_size),
                self.getGenerator(),
                )

    def _createVanillaJointNMT(self, attention):
        src_embeds = nn.Sequential(
                nn.Embedding(self.src_vocab, self.emb_size),
                nn.Dropout(self.dropout)
            )
        trg_embeds = nn.Sequential(
                nn.Embedding(self.tgt_vocab, self.emb_size),
                nn.Dropout(self.dropout)
            )

        decoder = Decoder(self.emb_size, self.hidden_size, attention, num_layers=self.num_layers,
            dropout=self.dropout)
        lang_decoder = LangModelDecoder(self.emb_size, self.hidden_size, num_layers=self.num_layers, \
             dropout=self.dropout, z_dim=0)
        return VanillaJointEncoderDecoder(
                Encoder(self.emb_size, self.hidden_size, num_layers=self.num_layers, dropout=self.dropout),
                decoder,
                src_embeds,
                trg_embeds,
                self.getGenerator(), #translation generator
                lang_decoder,
                self.getGenerator(self.src_vocab, use_maxout=False), #Language modeling generator
                self.num_layers,)

    def _createVNMT(self, attention, embed_concat, rnn_input_concat, use_posterior):
        projection = self.getProjection()
        decoder = Decoder(self.emb_size, self.hidden_size, attention,
         num_layers=self.num_layers, dropout=self.dropout, additional_input_dim=self.z_layer,
          embed_concat=embed_concat, rnn_input_concat=rnn_input_concat)

        posterior = GaussLayer(4*self.hidden_size, self.hidden_size, self.z_layer) if use_posterior else None
        return VariationalEncoderDecoder(
                Encoder(self.emb_size, self.hidden_size, num_layers=self.num_layers, dropout=self.dropout),
                decoder,
                nn.Embedding(self.src_vocab, self.emb_size),
                nn.Embedding(self.tgt_vocab, self.emb_size),
                self.getGenerator(),
                GaussLayer(2*self.hidden_size, self.hidden_size, self.z_layer), #prior 
                posterior,  #posterior
                projection # transforms samples from gaussian (normalizing flow??)
                )

    def _createSimpleModVNMT(self, attention):
        return self._createVNMT(attention, embed_concat=True, rnn_input_concat=False, use_posterior=False)

    def _createModVNMT(self, attention):
        #here we include z in both pre_output and as input to rnn
        return self._createVNMT(attention, embed_concat=True, rnn_input_concat=False, use_posterior=True) 
    
    def _createOriginalVNMT(self, attention):
        #in original VNMT they only concat as part of input to RNN
        return self._createVNMT(attention, embed_concat=False, rnn_input_concat=True, use_posterior=True)

    def _createVAENMT(self, attention):
        projection = self.getProjection(out_dim=self.hidden_size)
        if not self.use_projection:
            #vaenmt uses z to initialize the hidden states, so project allows this
            assert self.hidden_size == self.z_layer, "If not using projection layer set z_dim === hidden_dim. "
        return self._createGNMT(attention, projection, share_params=False, do_projection=True, embed_concat=False)

    def _createModVAENMT(self, attention):
        projection = self.getProjection(out_dim=self.hidden_size)
        if not self.use_projection:
            #vaenmt uses z to initialize the hidden states, so project allows this so otherwise they must be equal
            assert self.hidden_size == self.z_layer, "If not using projection layer set z_dim === hidden_dim. "
        return self._createGNMT(attention, projection, share_params=False, do_projection=True, embed_concat=True, langmodel_preconcat=True)

    def _createGNMT(self, attention, projection, share_params=True, do_projection=False, embed_concat=False, langmodel_preconcat=False):
        #do projection in gauss layers is not same as the projection layer
        #q_z_x_y = GaussLayer(4* self.hidden_size, self.hidden_size, self.z_layer,
        # do_projection=do_projection, sigma_act=nn.Softplus(), activation=nn.ReLU())

        q_z_x = GaussLayer(2* self.hidden_size, self.hidden_size,self.z_layer,
         do_projection=do_projection, sigma_act=nn.Softplus(), activation=nn.ReLU())

        inf_network = SourceInferenceNetwork(self.emb_size, self.hidden_size, dist_x=q_z_x,
          num_layers=self.num_layers, dropout=self.dropout)

        src_embeds = nn.Sequential(
                nn.Embedding(self.src_vocab, self.emb_size),
                nn.Dropout(self.dropout)
            )
        trg_embeds = nn.Sequential(
                nn.Embedding(self.tgt_vocab, self.emb_size),
                nn.Dropout(self.dropout)
            )

        decoder = Decoder(self.emb_size, self.hidden_size, attention, num_layers=self.num_layers,
            dropout=self.dropout,additional_input_dim=self.z_layer, embed_concat=embed_concat)
        lang_decoder = LangModelDecoder(self.emb_size, self.hidden_size, num_layers=self.num_layers, \
             dropout=self.dropout, z_dim=self.z_layer, preoutput_concat=langmodel_preconcat)
        return GenerativeEncoderDecoder(
                Encoder(self.emb_size, self.hidden_size, num_layers=self.num_layers, dropout=self.dropout),
                decoder,
                src_embeds,
                trg_embeds,
                self.getGenerator(), #translation generator
                lang_decoder,
                self.getGenerator(self.src_vocab, use_maxout=False), #Language modeling generator
                inf_network, 
                projection,
                self.num_layers,
                z_dim=self.z_layer ) #from paper
    
    def _createNMT(self, attention):
        return EncoderDecoder(
                Encoder(self.emb_size, self.hidden_size, num_layers=self.num_layers, dropout=self.dropout),
                Decoder(self.emb_size, self.hidden_size, attention, num_layers=self.num_layers, dropout=self.dropout),
                nn.Embedding(self.src_vocab, self.emb_size),
                nn.Embedding(self.tgt_vocab, self.emb_size),
                self.getGenerator())


