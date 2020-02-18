from gensim.models import FastText, KeyedVectors
import torch
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

EMBED_PATH='../cc.en.300.bin'

class BatchWordEmbeddings(nn.Module):
    #Module to load batches of embeddings
    def __init__(self, word2ind, emb_dim=300, freeze=True, pad='<PAD>', unk_tok='<UNK>',
                 sos='<SOS>', eos='<EOS>', pretrain_pth=None, saveModel=None,use_cuda=False):

        super(BatchWordEmbeddings, self).__init__()
        used_pretrain = False
        if pretrain_pth is not None:
            try:
                embed_info = torch.load(pretrain_pth)
                used_pretrain = True
            except Exception as e:
                print('could not load from pretrained path')
                print('{}'.format(str(e)))

        if not used_pretrain:
            word2ind[unk_tok] = len(word2ind)
            word2ind[pad] = len(word2ind)
            word2ind[sos] = len(word2ind)
            word2ind[eos] = len(word2ind)
            embeddings = nn.Embedding(len(word2ind), emb_dim, padding_idx=word2ind[pad])
            embed_info = {'matrix': embeddings, 'word2indx': word2ind}


        self.embeddings = embed_info['matrix']
        self.word2indx = embed_info['word2indx']
        self.indx2word = {v: k for k, v in self.word2indx.items()} 

        #these are parameters mostly for loading new pretrained models
        self.freeze = freeze
        self.pad = pad
        self.unk_tok = unk_tok
        self.sos = sos
        self.eos = eos
        self.saveModel = saveModel
        self.use_cuda = use_cuda
        if use_cuda:
            self.cuda()
    def sortAndPad(self, indicesListAndLens, y=None, batch_first=True):
        #input is a list of indices that are unsorted and a list of their lengths
        #y is a list of labels who pairing with the input is maintained
        sequences = indicesListAndLens
        if y is not None:
            assert len(indicesListAndLens) == len(y) , 'num entries {}  not equal y {}'.format(len(indicesListAndLens), len(y))
            #this produces a tuple of 3 entries: (indices, length, y)
            for i in range(len(sequences)):
                sequences[i].append(y[i])

        sorted_seq = sorted(sequences, key= lambda x: x[1], reverse=True)
        sequences = [ent[0] for ent in sorted_seq]
        lens = [ent[1] for ent in sorted_seq]
        y = y if y is None else [ ent[2] for ent in sorted_seq]
        sequences = pad_sequence(sequences, batch_first=batch_first, padding_value=self.word2indx['<PAD>'])
        if self.use_cuda:
            sequences = sequences.cuda()
        return sequences, lens, y
    def padAndMask(self, indicesListAndLens, batch_first=True):
        sequences = [indices[0] for indices in indicesListAndLens]
        pad = pad_sequence(sequences, batch_first=batch_first, padding_value=self.word2indx['<PAD>'])
        mask = pad.new_zeros(pad.shape)
        for i, e in enumerate(indicesListAndLens):
            #TODO for some reason...using e[1] caused memory problems
            mask[i,0:len(e[0])] = 1.0
        if self.use_cuda:
            pad = pad.cuda()
            mask = mask.cuda()            

        return pad, mask
    def getBatchEmbeddings(self, batch):
        #given a batch of indices in embedding return them
        return self.embeddings(batch)

    def sentences2IndexesAndLens(self, sentences):
        #List of Str -> List of List  (tensorofIndices length )
        sequences = [[None]]*len(sentences)
        for i, s in enumerate(sentences):
            indices = [self.word2indx[t] if t in self.word2indx else self.word2indx['<UNK>'] for t in s.split() ]
            indices = [self.word2indx[self.sos]] + indices + [self.word2indx[self.eos]]
            sequences[i] = [torch.tensor(indices).long(), len(indices)]

        return sequences 

    def forward(self, sentences, y=None, batch_first=True, pack_seq=True):
        #sentences is a list of strings (our data)
        # y is list of labels which we just sort (do not mess with otherwise)
        sequences = self.sentences2IndexesAndLens(sentences)

        padded, lengths, y = self.sortAndPad(sequences, y, batch_first)
        mask = torch.ones((len(lengths), max(lengths)))
        for i in range(len(mask)):
            l = lengths[i]
            mask[i][l:] = 0

        if self.use_cuda:
            padded = padded.cuda()
            mask = mask.cuda()
        embeddings = self.embeddings(padded)
        if pack_seq:
            embeddings = pack_padded_sequence(embeddings, lengths, batch_first=batch_first)
        return embeddings, lengths, mask, y 

    def loadFastText(self, model_pth, vocab):
        if '.bin' in model_pth:
            model = FastText.load_fasttext_format(model_pth)
        else:
            model = FastText.load(model_pth)
        self.create_embeddings(model, vocab)

    def create_embeddings(self, model, vocab):
        embeddings = []
        word2indx = {}
        for i, word in enumerate(vocab) :
            word = word.strip()
            try:
                embeddings.append(torch.from_numpy(model.wv[word]).float())
                word2indx[word] = i
            except:
                embeddings.append(torch.randn(model.wv.vector_size))
                word2indx[word] = i
        embeddings.append(torch.zeros(model.wv.vector_size))
        word2indx[self.pad] = len(word2indx)
        embeddings.append(torch.rand(model.wv.vector_size))
        word2indx[self.unk_tok] = len(word2indx)

        embeddings = torch.stack(embeddings)
        embed_info = {'matrix': embeddings, 'word2indx': word2indx}
        self.embeddings = nn.Embedding.from_pretrained(embeddings,freeze=self.freeze)
        self.word2indx = word2indx

        if self.saveModel is not None:
            torch.save(embed_info, self.saveModel)
    def getVocab(self):
        return self.word2indx.keys()
    def getWord2Index(self, word):
        return self.word2indx[word]
    def getIndex2Word(self, index):
        return self.indx2word[index]
    def getSOS(self):
        return self.sos
    def getSOSIndex(self):
        return self.word2indx[self.sos]
    def getEOS(self):
        return self.eos
    def getEOSIndex(self):
        return self.word2indx[self.eos]

if __name__ == "__main__":

    sentences = ["I am an apple", "There is a dog", "look a tree", 'here is a realy long sentence just for verifying']
    words = []
    for s in sentences:
        for w in s.split():
            words.append(w)
    words = set(words)
    word2idx = {w:i for i, w in enumerate(list(words))}
    m = BatchWordEmbeddings(word2idx)
    #m.loadFastText('../data/cc.en.300.bin', word2idx.keys())
    #print(m)
    sentences = [s for s in sentences]
    y = list(range(len(sentences)))
    emb, lengths, mask, y = m(sentences, y)
    gru = nn.GRU(300, 300)
    packe_emb, pack_hid = gru(emb)
    unpacked_packe_emb = pad_packed_sequence(packe_emb)

    unpacked, lengths = pad_packed_sequence(emb)
    unpack_emb, unpack_hid = gru(unpacked)

    emb2, lengths, mask, _ = m(sentences)

    print(m(sentences, y))
    print(m(sentences))
                    


                    




