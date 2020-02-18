import torch.utils.data
from torch.utils.data import Dataset
import torch
import logging

class BitextDataSet(Dataset):

    def __init__(self, L1, L2, sentence_limit=50, transform=None, savePath='./'):
        filename = savePath + L1.split('/')[-1].split('.')[0] + '_' + L2.split('/')[-1].split('.')[0] + '.pth'
        self.sent_lim = sentence_limit
        try:
            self.dataset = torch.load(filename)
            print('found prev bitext pair. if not desired delete {}'.format(filename))
        except Exception as e:
            print('new data pair, creating file and saving')   
            self.CreateBitextPairs(L1, L2, filename)
        self.transform = transform
        self.src_Word2Indx = self.createWord2IndexDicts([s[0] for s in self.dataset])
        self.tgt_Word2Indx = self.createWord2IndexDicts([s[1] for s in self.dataset])


    def withinLimit(self, sen1, sen2):
        s1 = sen1.split()
        s2 = sen2.split()
        return len(s1) <= self.sent_lim and len(s2) <= self.sent_lim

    def CreateBitextPairs(self, L1, L2, savename):
        with open(L1) as l1, open(L2) as l2:
            pairs = list(zip(l1, l2))
            self.dataset = [[p[0],p[1]] for p in pairs if self.withinLimit(p[0], p[1])]
        torch.save(self.dataset, savename)

    def __getitem__(self, idx):
        entry = self.dataset[idx]
        if self.transform is not None:
           entry = self.transform(entry)
        return entry[0], entry[1]

    def __len__(self):
        return len(self.dataset)

    def createWord2IndexDicts(self, sentences):
        #Given a list of sentences, create a word2indx dictionary
        #assumes words have been processed to be separated by spaces
        word2index = {}
        index = 0
        for s in sentences:
            tokens = s.split()
            for t in tokens:
                if t in word2index:
                    continue
                else:
                    word2index[t] = index
                    index += 1
        return word2index

    def getSrcWord2Index(self):
        return self.src_Word2Indx.copy()
    def getTgtWord2Index(self):
        return self.tgt_Word2Indx.copy()



class Word2IndxTransform:
    #will transform a string sentence into list of indices
    #word2indx dictionary of words & associate index
    #unk_tok: default token to use on unsceen words
    def __init__(self, word2indx, unk_tok):
        self.word2indx = word2indx
        self.unk_tok = unk_tok
        if unk_tok not in self.word2indx:
            self.word2indx[unk_tok] = len(self.word2indx)
    def transform(self, sentence):
        words = sentence.split()
        return [self.word2indx[w] if w in self.word2indx else self.word2indx[self.unk_tok] for w in words]

def NoiseModel(indices,pw=.9, k=3, a=None):
    # currently this shuffles words then drops them, i guess...either way this is fine
    if len(indices) <= 1:
        return indices

    indices += 1
    a = (k + 1) if a is None else a
    noise = torch.zeros(indices.size()).uniform_(0, a)
    q = indices + noise
    q_sort, noise_ind = torch.sort(q)
    mask = torch.bernoulli(torch.ones(noise_ind.size()).float() * pw)
    mask = mask.long()
    
    return torch.tensor([noise_ind[m] for m in torch.nonzero(mask)]).long()

def strNoise(str_list, C=NoiseModel):
    if len(str_list) == 1:
        return str_list[0]

    noise_ind = C(torch.arange(len(str_list)).float())
    retry = 0
    while len(noise_ind) < 1 :
        noise_ind = C(torch.arange(len(str_list)).float())
        retry += 1
        if retry > 3:
            #if we retry 4 times and still don't get a good noise model, just return original sentence
            noise_ind = torch.arange(len(str_list)).float()

    noise_sent = str_list[noise_ind[0]]
    for i in noise_ind[1:]:
        noise_sent = noise_sent + ' ' + str_list[i] 
    return noise_sent

def SentenceNoise(sent, C=NoiseModel):
    words = sent.split()
    return strNoise(words, C)

def WordNoise(word, C=NoiseModel):
    chars = list(word) 
    return strNoise(chars, C=NoiseModel).replace(" ", "")


if __name__ == "__main__":
    sent = "There was a friendly man who did his best"
    word = "because"
    print(sent)
    print(SentenceNoise(sent))
    print(SentenceNoise(sent))
    print(word)
    print(WordNoise(word))
    print(WordNoise(word))
    with open('en.txt') as f, open('noiseEn.txt', 'w+') as c:
        for l in f:
            noise = SentenceNoise(l)
            c.write(noise + '\n')




            
            



