import torch.nn as nn
import torch
from pyro.distributions import Bernoulli
#Code is from following: https://colab.research.google.com/github/bastings/annotated_encoder_decoder/blob/master/annotated_encoder_decoder.ipynb
class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many 
    other models.
    """
    def __init__(self, encoder, decoder, src_embed, trg_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.trg_embed = trg_embed
        self.generator = generator

        self.unk_tok_indx = 0
        self.sos_tok_indx = 2
        self.pad_index = 1
        self.train_size = 1.
        #TODO: technically only matters for variational models...is a hack so i don't hav eto change main
        self.use_mean_field = False
        self.to_anneal = 'q_p'
        
    def forward(self, src, trg, src_mask, trg_mask, src_lengths, trg_lengths):
        """Take in and process masked src and target sequences."""
        encoder_hidden, encoder_final = self.encode(src, src_mask, src_lengths)
        return self.decode(encoder_hidden, encoder_final, src_mask, trg, trg_mask)
    
    def encode(self, src, src_mask, src_lengths, pad_pack=True, hidden=None):
        return self.encoder(self.src_embed(src), src_mask, src_lengths, pad_pack, hidden)
    
    def decode(self, encoder_hidden, encoder_final, src_mask, trg, trg_mask,
               decoder_hidden=None):
        return self.decoder(self.trg_embed(trg), encoder_hidden, encoder_final,
                            src_mask, trg_mask, hidden=decoder_hidden)
    def callGenerator(self, pre_outputs):
        return self.generator(pre_outputs)

    def save(self, pth):
        torch.save(self.state_dict(), pth)

    def load(self, pth_or_dict):
        if isinstance(pth_or_dict, str):
            self.load_state_dict(torch.load(pth_or_dict))
        else:
            self.load_state_dict(pth_or_dict)
    
    def setTrainDataSize(self, train_size):
        #this is very specific to VNMT
        self.train_size = train_size

    def setUnkTokenIndex(self, unk_tok_indx):
        self.unk_tok_indx = unk_tok_indx

    def setSOSTokenIndex(self, sos_tok_indx):
        self.sos_tok_indx = sos_tok_indx

    def setWordDropout(self, word_drop):
        self.word_drop = word_drop

    def setPadIndex(self, pad_index):
        self.pad_index = pad_index

    def setUseMeanField(self, is_meanfield):
        self.use_mean_field = is_meanfield

    def setToAnneal(self, to_anneal):
        self.to_anneal = to_anneal

    def getWordEmbeddingsWithWordDropout(self, embeddings, indexes, pad_mask ):
        #word drop out approach proposed in bowman et. al 2016
        if len(pad_mask.size()) > 2:
            pad_mask = pad_mask.squeeze()
        indexes = indexes.clone() #clone tensor just in case...don't want any mutation
        mask = torch.zeros_like(indexes).float().fill_(self.word_drop)
        mask = Bernoulli(mask).sample().byte()
        mask = mask * pad_mask.byte() # don't word drop things passed sentence length
        mask[0,:] = 0 #do not mask out sos token
        try:
            mask = mask.bool()
        except:
            #do nothing
            i = 0
        indexes.masked_fill_(mask, self.unk_tok_indx)

        return embeddings(indexes)
    
    def getWordDropout(self):
        return self.word_drop

    def getUnkTokenIndex(self):
        return self.unk_tok_indx
    
    def initParameters(self, type):
        for n, p in self.named_parameters():
            if 'rnn' in n and p.dim() > 1:
                nn.init.orthogonal_(p)
            elif type == 'normal':
                nn.init.normal_(p, mean=0, std=0.001)
            elif type == 'xavier_uniform':
                nn.init.xavier_uniform_(p)
            else:
                raise ValueError("Unsupported Initialization type {}".format(type))
