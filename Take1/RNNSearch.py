import torch
import torch.nn as nn
import torch.nn.functional as F
import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from GaussianLayer import GaussLayer
from BatchWordEmbeddings import BatchWordEmbeddings
from  AttentionLayers import BahdanauAttention, GlobalAttention
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

class RNNSearch(nn.Module):

    def __init__(self ,x_w2i, y_w2i, input_dim=300, hidden_dim=200, num_layers=1,
                 enc_bi=True, enc_dropout=0.0, dec_dropout=0.0, b_f=True, use_cuda=False):
        super(RNNSearch, self).__init__()
        #seq2seq model
        self.x_embed = BatchWordEmbeddings(x_w2i,use_cuda=use_cuda)
        self.y_embed = BatchWordEmbeddings(y_w2i, use_cuda=use_cuda)
        self.encoder = nn.GRU(input_dim, hidden_dim, num_layers,
                              batch_first=b_f, dropout=enc_dropout,
                              bidirectional=enc_bi)

        self.bridge = nn.Linear(hidden_dim * 2 if enc_bi else 1, hidden_dim)
        context_dim = hidden_dim * 2 if enc_bi else 1
        self.global_attention = GlobalAttention(hidden_dim, context_dim)

        self.decoder = nn.GRU(input_dim + context_dim, hidden_dim, num_layers,
                              batch_first=b_f, dropout=dec_dropout)
        self.emitter = nn.Linear(hidden_dim, len(y_w2i))

        
        #misc parameters
        self.tgt_vocab_size = len(self.y_embed.getVocab())
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.b_f = b_f
        self.enc_bi = enc_bi
        self.use_cuda = use_cuda
        if self.use_cuda:
            self.cuda()

    
    def guide(self, x_sent, y_sent):
        pass

    def model(self, x_sent, y_sent):
        pyro.module("seq2seq", self)
        x_embeds, x_len, x_mask, y_sent = self.x_embed(x_sent, y_sent)
        x_out, s_0 = self.encoder(x_embeds)
        h_j, _ = pad_packed_sequence(x_out, batch_first=self.b_f)

        #TODO technically this is done in forward call....
        y_labels = self.y_embed.sentences2IndexesAndLens(y_sent)

        T_max = max([y[1] for y in y_labels])
        y_labels, y_mask = self.y_embed.padAndMask(y_labels, batch_first=self.b_f)
        if self.use_cuda:
            y_labels = y_labels.cuda()

        with pyro.plate('z_minibatch'):
            #sample from model prior P(Z | X)
            #Generate sequences
            #TODO probably need to verify this, supposed to be H_e' = g(Wh_z + b_z) in paper eq 11

            #TODO verify that this makes sense to do
            #TODO so...based on paper graphic the init hidden state is the last part of the RNN run in reverse on seq
            #TODO i added a "bridge" to take in the final context and convert to a proper hidden state size...ned to put it in
            #s_0 = s_0.view(self.num_layers,2 if self.enc_bi else 1, len(y_len), self.hidden_dim)
            s_0 = torch.cat([s_0[0], s_0[1]],dim=1).unsqueeze(0)
            #s_t = s_0[:, 1 if self.enc_bi else 0, :, :]
            s_t = self.bridge(s_0)
            for t in range(0, T_max):
                #TODO atm we are teacher forcing the model (i.e. using labeledinputs)
                #TODO in future may want to use generative samples to improve robustness
                #probably need to figure out a more eloquent solution...
                inputs = self.y_embed.getBatchEmbeddings(y_labels[:, t]).unsqueeze(1) 
                #attention mechanism
                context_vector = self.global_attention(s_t[0], h_j, batch_first=self.b_f)

                inputs = torch.cat([inputs, context_vector.unsqueeze(1)], dim=2)
                #decoding
                output, s_t = self.decoder(inputs, s_t)
                output = self.emitter(output.squeeze())
                l = y_labels[:,t]
                entry = pyro.sample('y_{}'.format(t),
                            dist.Categorical(probs=F.softmax(output, dim=1)).mask(y_mask[:, t:t+1].squeeze()),
                            obs=l)

    def forward(self, x_sent, y_sent):
        print('hello world')
        return x_sent, y_sent
 
if __name__ == "__main__":
    sentences = ["I am an apple", "There is a dog", "look a tree", 'here is a realy long sentence just for verifying']
    words = []
    for s in sentences:
        for w in s.split():
            words.append(w)
    words = set(words)
    word2idx = {w:i for i, w in enumerate(list(words))}
    x_words = word2idx
    seq2seq = RNNSearch(x_words, x_words.copy())
    seq2seq.model(sentences, sentences)
    seq2seq.guide(sentences, sentences)

