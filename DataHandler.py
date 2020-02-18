from torchtext import data
import torch.nn as nn
import torch.nn.functional as F
import json

class DataHandler:
    def __init__(self, src_tokenizer, trg_tokenizer, to_lower, eos_tok, sos_tok, pad_tok, unk_tok, min_freq, device):
        self.src_field = data.Field(tokenize=src_tokenizer, 
                     batch_first=True, lower=to_lower, include_lengths=True,
                     unk_token=unk_tok, pad_token=pad_tok, init_token=sos_tok, eos_token=eos_tok)
        self.trg_field = data.Field(tokenize=trg_tokenizer, 
                     batch_first=True, lower=to_lower, include_lengths=True,
                     unk_token=unk_tok, pad_token=pad_tok, init_token=sos_tok, eos_token=eos_tok)
        self.min_freq = min_freq

        self.pad_tok = pad_tok
        self.pad_indx = -1 # we don't know till you build the vocab

        self.device = device
        self.min_freq = min_freq

    def build_vocab_(self, field, data, custom_vocab=None):
        if custom_vocab == None:
            field.build_vocab(data, min_freq=self.min_freq)
        else: 
            with open(custom_vocab, 'r') as f:
                custom_vocab = json.load(f)
                if isinstance(custom_vocab, dict):
                    custom_vocab = [[w] for w in list(custom_vocab.keys())]
                field.build_vocab(custom_vocab, min_freq=self.min_freq)

    def build_vocabs(self, train_data, custom_vocab_src=None, custom_vocab_trg=None):
        self.build_vocab_(self.src_field, train_data.src, custom_vocab_src)
        self.build_vocab_(self.trg_field, train_data.trg, custom_vocab_trg)
        self.pad_indx = self.trg_field.vocab.stoi[self.pad_tok]

    def load_vocabs(self, src_vocab, trg_vocab):
        self.src_field.vocab = src_vocab
        self.trg_field.vocab = trg_vocab
        self.pad_indx = self.trg_field.vocab.stoi[self.pad_tok]

    def getPadIndex(self):
        return self.pad_indx

    def getBucketIter(self, dataset, **kwargs):
        if 'device' not in kwargs:
            kwargs = dict(kwargs, device=self.device)
        else:
            kwargs = dict(kwargs)
        return data.BucketIterator(dataset, **kwargs)

    def getIter(self, dataset, **kwargs):
        if 'device' not in kwargs:
            kwargs = dict(kwargs, device=self.device)
        else:
            kwargs = dict(kwargs) # just in case
        return data.Iterator(dataset, **kwargs)

    def getSRCField(self):
        return self.src_field

    def getTRGField(self):
        return self.trg_field
    
    def getSRCVocab(self):
        return self.src_field.vocab

    def getTRGVocab(self):
        return self.trg_field.vocab



    