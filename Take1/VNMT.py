import torch
import torch.nn as nn
import torch.nn.functional as F
import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from GaussianLayer import GaussLayer
from BatchWordEmbeddings import BatchWordEmbeddings
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

class VNMT(nn.Module):

    def __init__(self,x_w2i, y_w2i, input_dim=300, z_dim=100, hidden_dim=200,gauss_dim=200, num_layers=1,
                 c_dim = 300,enc_bi=True, enc_dropout=0.0, dec_dropout=0.0, b_f=True, use_cuda=False):
        super(VNMT, self).__init__()
        self.x_embed = BatchWordEmbeddings(x_w2i,use_cuda=use_cuda)
        self.y_embed = BatchWordEmbeddings(y_w2i, use_cuda=use_cuda)

        #your typical seq2seq RNNs
        self.encoder = nn.GRU(input_dim, hidden_dim, num_layers,
                              batch_first=b_f, dropout=enc_dropout,
                              bidirectional=enc_bi)

        
        #use to actually you know, decode sentences
        self.emitter = nn.Linear(hidden_dim, len(y_w2i))
        posterior_dim = hidden_dim * 4 if enc_bi else 2
        prior_dim = hidden_dim * 2 if enc_bi else 1
        #TODO add attention
        self.posterior = GaussLayer(posterior_dim, gauss_dim, z_dim)
        self.prior = GaussLayer(prior_dim, gauss_dim, z_dim)
        self.h_e_p = nn.Linear(z_dim, c_dim)

        #TODO I was wondering how they go from the biRNN to single RNN...it's a bridge :O
        self.bridge = nn.Linear(hidden_dim * 2 if enc_bi else 1, hidden_dim)
        self.decoder = nn.GRU(input_dim + c_dim, hidden_dim, num_layers,
                              batch_first=b_f, dropout=dec_dropout)


        #misc stuff
        self.tgt_vocab_size = len(self.y_embed.getVocab())
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.b_f = b_f
        self.enc_bi = enc_bi
        self.use_cuda = use_cuda
        if self.use_cuda:
            self.cuda()

    def guide(self, x_sent, y_sent, decode=False, kl_annealing=1.0):
        pyro.module("vnmt", self)
        #transform sentences into embeddings
        x_embeds, x_len, x_mask, y_sent = self.x_embed(x_sent, y=y_sent)
        y_embeds, y_len, y_mask, y_indices = self.y_embed(y_sent, y=range(len(y_sent)), pack_seq=True)

        #run our sequences through rnn
        #2nd output is just last hidden state where as I want ALL hidden states which are output

        x_out, _ = self.encoder(x_embeds)
        X, x_len = pad_packed_sequence(x_out, batch_first=self.b_f)
        #TODO one problem with this is that the sequences are varying lengths
        #TODO i.e. even if the rest of the dims are 0 vectors the strength of
        #TODO each of existing entries might be lessened depending on how long longest sequence is
        if self.use_cuda:
            x_len = x_len.cuda()
        X = torch.sum(X, dim=1) / x_len.unsqueeze(1).float()

        y_out, _ = self.encoder(y_embeds)
        Y, y_len = pad_packed_sequence(y_out, batch_first=self.b_f)
        if self.use_cuda:
            y_len = y_len.cuda()
        Y = torch.sum(Y, dim=1) / y_len.unsqueeze(1).float()

        Y = zip([r for r in Y], y_indices)
        Y = sorted(Y, key=lambda x: x[1], reverse=False)
        Y = torch.stack([y[0] for y in Y])


        #Mean pool over the hidden states to
        z_input = torch.cat([X, Y],dim=1)
        z_mean, z_sig = self.posterior(z_input)

        #sample from our variational distribution P(Z | X, Y)
        with pyro.plate('z_minibatch'):
            with poutine.scale(scale=kl_annealing):
                semantics = pyro.sample('z_semantics', dist.Normal(z_mean, z_sig))


    def model(self, x_sent, y_sent, decode=False, kl_annealing=1.0):
        pyro.module("vnmt", self)
        #Produce our prior parameters
        x_embeds, x_len, x_mask, y_sent = self.x_embed(x_sent, y_sent)
        x_out, s_0 = self.encoder(x_embeds)

        X, x_len = pad_packed_sequence(x_out, batch_first=self.b_f)
        if self.use_cuda:
            x_len = x_len.cuda()

        z_input = torch.sum(X, dim=1) / x_len.unsqueeze(1).float()
        z_mean, z_sig = self.prior(z_input)

        #TODO technically this is done in forward call....
        y_labels = self.y_embed.sentences2IndexesAndLens(y_sent)
        

        T_max = max([y[1] for y in y_labels])
        y_labels, y_mask = self.y_embed.padAndMask(y_labels, batch_first=self.b_f)

        with pyro.plate('z_minibatch'):
            #sample from model prior P(Z | X)
            with poutine.scale(scale=kl_annealing):
                semantics = pyro.sample('z_semantics', dist.Normal(z_mean, z_sig))
            #Generate sequences
            #TODO probably need to verify this, supposed to be H_e' = g(Wh_z + b_z) in paper eq 11
            semantics = F.relu(self.h_e_p(semantics))

            #TODO verify that this makes sense to do
            #TODO so...based on paper graphic the init hidden state is the last part of the RNN run in reverse on seq
            #TODO i added a "bridge" to take in the final context and convert to a proper hidden state size...ned to put it in
            #s_0 = s_0.view(self.num_layers,2 if self.enc_bi else 1, len(y_len), self.hidden_dim)
            s_t = s_0[1].unsqueeze(0) #s_0[:, 1 if self.enc_bi else 0, :, :]
            for t in range(0, T_max):
                #TODO atm we are teacher forcing the model (i.e. using labeledinputs)
                #TODO in future may want to use generative samples to improve robustness
                #probably need to figure out a more eloquent solution...
                l = y_labels[:,t]
                inputs = torch.cat([F.relu(self.y_embed.getBatchEmbeddings(l).unsqueeze(1)), semantics.unsqueeze(1)], dim=2)
                output, s_t = self.decoder(inputs, s_t)
                output = self.emitter(output).squeeze(1)
                entry = pyro.sample('y_{}'.format(t),
                            dist.Categorical(probs=F.softmax(output, dim=1)).mask(y_mask[:, t:t+1].squeeze()),
                            obs=l)

    def forward(self, sentences):
        print('probably can use this for decoding')
        return sentences

    def usePretrained(self, path, emb, is_x=True):
        if is_x:
            words = self.x_embed.getVocab()
            del self.x_embed
        else:
            words = self.y_embed.getVocab()
            del self.y_embed

        new_emb = BatchWordEmbeddings({'dummy': 0})
        if emb.lower() == 'fasttext':
            new_emb.loadFastText('../data/cc.en.300.bin', word2idx.keys())
        else:
            print('only fast text supported right now')

        if is_x:
            self.x_embed = new_emb
        else:
            self.y_embed = new_emb

    def initHidden(self, batch_size, bidir):
        num_dir = 2 if bidir else 1
        if self.b_f:
            dims = (batch_size,num_dir * self.num_layers, self.hidden_dim)
        else:
            dims = (num_dir * self.num_layers, batch_size, self.hidden_dim)

        return torch.zeros(dims)


if __name__ == "__main__":
    sentences = ["I am an apple", "There is a dog", "look a tree", 'here is a realy long sentence just for verifying']
    words = []
    for s in sentences:
        for w in s.split():
            words.append(w)
    words = set(words)
    word2idx = {w:i for i, w in enumerate(list(words))}
    x_words = word2idx
    vnmt = VNMT(x_words, x_words.copy())
    vnmt.model(sentences, sentences)
    #vnmt.guide(sentences, sentences)