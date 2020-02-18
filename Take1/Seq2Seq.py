import torch
import torch.nn as nn
import torch.nn.functional as F
from BatchWordEmbeddings import BatchWordEmbeddings
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

class Seq2Seq(nn.Module):

    def __init__(self ,x_w2i, y_w2i, input_dim=300, hidden_dim=200, num_layers=1,
                 enc_bi=True, enc_dropout=0.0, dec_dropout=0.0, b_f=True, use_cuda=False):
        super(Seq2Seq, self).__init__()
        #seq2seq model
        self.x_embed = BatchWordEmbeddings(x_w2i,use_cuda=use_cuda)
        self.y_embed = BatchWordEmbeddings(y_w2i, use_cuda=use_cuda)
        self.encoder = nn.GRU(input_dim, hidden_dim, num_layers,
                              batch_first=b_f, dropout=enc_dropout,
                              bidirectional=enc_bi)

        self.bridge = nn.Linear(hidden_dim * 2 if enc_bi else 1, hidden_dim)
        self.decoder = nn.GRU(input_dim, hidden_dim, num_layers,
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

    def forward(self, x_sent, y_sent, loss_fn=None):
        #handles getting word embeddings and keeping pair data aligned
        if loss_fn is None:
            loss_fn = nn.CrossEntropyLoss(reduction='sum',ignore_index=self.y_embed.getWord2Index('<PAD>'))
        x_embeds, _, _, y_sent = self.x_embed(x_sent, y_sent)
        #s_0 is presumably the final hidden states of our encoder
        x_out, s_0 = self.encoder(x_embeds)

        y_labels = self.y_embed.sentences2IndexesAndLens(y_sent)

        T_max = max([y[1] for y in y_labels])
        y_labels, y_mask = self.y_embed.padAndMask(y_labels, batch_first=self.b_f)
        if self.use_cuda:
            y_labels = y_labels.cuda()
        #Presumably this line make each data point independent of others
            
        s_0 = torch.cat([s_0[0], s_0[1]],dim=1)
        #The bridge is supposedly a trick to 
        s_t = torch.tanh(self.bridge(s_0).unsqueeze(0))
        loss = 0.0
        total_loss = 0.0
        for t in range(0, T_max):
            #TODO atm we are teacher forcing the model (i.e. using labeledinputs)
            #TODO in future may want to use generated samples to improve robustness
            inputs = self.y_embed.getBatchEmbeddings(y_labels[:, t]).unsqueeze(1) 
            output, s_t = self.decoder(inputs, s_t)
            output = self.emitter(output.squeeze(1))
            l = y_labels[:,t]
            seq_mask = y_mask[:, t].float()
            #loss_val = torch.dot(loss_fn(output, l), seq_mask) / torch.sum(y_mask)
            loss_val = loss_fn(output, l)
            loss = loss + loss_val
        return loss / len(x_sent)

    def encode(self, sentence):
        #encode input sentences with src_embeddings and return final hidden state
        assert isinstance(sentence, str), 'sentence must be string, given {}'.format(type(sentence))
        embedding, _, _, _ = self.x_embed([sentence])
        x_out, s_t = self.encoder(embedding)
        s_0 = torch.cat([s_t[0], s_t[1]], dim=1)
        s_0 = torch.tanh(self.bridge(s_0).unsqueeze(0))
        return s_0

    def decode(self, hidden_state, curr_token):
        #assert isinstance(curr_token, str), 'curr_token must be string, given {}'.format(type(curr_token))
        #given an hidden_state and current token, get token embedding and pass through decoder
        if isinstance(curr_token, str):
            index = self.y_embed.getWord2Index(curr_token)
        else:
            index = curr_token
        index = torch.Tensor([index]).long() 
        if self.use_cuda:
            index = index.cuda()
        inputs = self.y_embed.getBatchEmbeddings(index)
        output, s_t = self.decoder(inputs.unsqueeze(1), hidden_state)
        #return log likelyhood
        return F.log_softmax(self.emitter(output.squeeze(1)),dim=1), s_t

    def getTGTSOS(self):
        return self.y_embed.getSOS()

    def getTGTEOS(self):
        return self.y_embed.getEOS()

    def getTGTEOSIndx(self):
        return self.y_embed.getEOSIndex()
    def getTGTSOSIndx(self):
        return self.y_embed.getSOSIndex()
    def getSRCSOSIndx(self):
        return self.x_embed.getSOSIndex()
    def getSRCSOS(self):
        return self.x_embed.getSOS()
 
if __name__ == "__main__":
    sentences = ["I am an apple", "There is a dog", "look a tree", 'here is a realy long sentence just for verifying']
    words = []
    for s in sentences:
        for w in s.split():
            words.append(w)
    words = set(words)
    word2idx = {w:i for i, w in enumerate(list(words))}
    x_words = word2idx
    seq2seq = Seq2Seq(x_words, x_words.copy())
    seq2seq.model(sentences, sentences)
    seq2seq.guide(sentences, sentences)

