print('change use_cuda to false at some point and set it correctly (in main)')
from pyro.distributions import Bernoulli
class Batch:
    """Object for holding a batch of data with mask during training.
    Input is a batch from a torch text iterator.
    """
    def __init__(self, src, trg, pad_index=0, word_drop=0.0, unk_indx=0, use_cuda=False):
        
        src, src_lengths = src
        
        self.src = src
        self.src_lengths = src_lengths
        self.src_mask = (src != pad_index).unsqueeze(-2)
        self.nseqs = src.size(0)
        
        self.trg = None
        self.trg_y = None
        self.trg_mask = None
        self.trg_lengths = None
        self.ntokens = None

        if trg is not None:
            trg, trg_lengths = trg
            self.trg = trg[:, :-1]
            #word drop out approach proposed in bowman et. al 2016
            mask = trg.new_zeros(self.trg.size(0), self.trg.size(1)).float().fill_(word_drop)
            mask = Bernoulli(mask).sample().byte()
            try:
                mask = mask.bool()
            except AttributeError as e:
                #just means your using an older pytorch version... 
                _ = 0
            self.trg.masked_fill_(mask, unk_indx)
            self.trg_lengths = trg_lengths
            self.trg_y = trg[:, 1:]
            self.trg_mask = (self.trg_y != pad_index)
            self.ntokens = (self.trg_y != pad_index).data.sum().item()
        
        if use_cuda:
            self.src = self.src.cuda()
            self.src_mask = self.src_mask.cuda()

            if trg is not None:
                self.trg = self.trg.cuda()
                self.trg_y = self.trg_y.cuda()
                self.trg_mask = self.trg_mask.cuda()
        else:
            self.src = self.src.cpu()
            self.src_mask = self.src_mask.cpu()
            self.src_lengths = self.src_lengths.cpu()

            if trg is not None:
                self.trg = self.trg.cpu()
                self.trg_y = self.trg_y.cpu()
                self.trg_mask = self.trg_mask.cpu()
                self.trg_lengths = self.trg_lengths.cpu()