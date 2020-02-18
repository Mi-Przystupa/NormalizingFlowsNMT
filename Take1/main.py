import torch
import torch.nn as nn
import argparse
import logging
import pyro
from pyro.infer import SVI, Trace_ELBO, JitTrace_ELBO
from pyro.optim import ClippedAdam
from torch.utils.data import DataLoader
from VNMT import VNMT
from Seq2Seq import Seq2Seq
from PyroSeq2Seq import PyroSeq2Seq
from RNNSearch import RNNSearch
from Decoders import GreedyDecodingTranslation, BeamDecodingTranslation, SimpleGreedyDecodingTranslation
from ModelFactory import make_model
from LossFuncs import SimpleLossCompute
import math, time

from BitextDataSet import BitextDataSet

def train(dataloader, model, optimizer, loss, trainOp=None):

    model.train()
    for b in dataloader:
        if trainOp is None:
            optimizer.zero_grad()
            outputs = model(b[0][0])
            l = loss(outputs, b[1][0])
            l.backward()
            optimizer.step()
        else:
            trainOp(b, model, optimizer, loss)
    return model

def run_epoch(data_iter, model, loss_compute, print_every=50):
    """Standard Training and Logging Function"""

    start = time.time()
    total_tokens = 0
    total_loss = 0
    print_tokens = 0
    for i, batch in enumerate(data_iter, 1):
#        out, _, pre_output = model.forward(batch.src, batch.trg,
#                                           batch.src_mask, batch.trg_mask,
#                                           batch.src_lengths, batch.trg_lengths)

        out, _, pre_output,y_sent =model.forward(batch[0], batch[1])

        #loss = loss_compute(pre_output, batch.trg_y, batch.nseqs)
        targets =  model.getTargets(y_sent)
        loss = loss_compute(pre_output, targets, len(batch[0]))
        total_loss += loss
        total_tokens += len(y_sent)#batch.ntokens
        print_tokens += len(y_sent)#batch.ntokens
        
        if model.training and i % print_every == 0:
            elapsed = time.time() - start
            print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
                    (i, loss / len(y_sent), print_tokens /elapsed))#batch.nseqs, print_tokens / elapsed))
            start = time.time()
            print_tokens = 0

    return math.exp(total_loss / float(total_tokens))

def custom_collator(batch):
    sentences1 = [s[0] for s in batch]
    sentences2 = [s[1] for s in batch]
    return sentences1, sentences2



def main(args):
    pyro.enable_validation(is_validate=True)
    dataset = BitextDataSet(args.L1, args.L2,sentence_limit=5)
    dataloader = DataLoader(dataset, batch_size=64, collate_fn=custom_collator, shuffle=True)
    #pyro fun, all of it is still broken :P 
    #vnmt = VNMT(dataset.getSrcWord2Index(), dataset.getTgtWord2Index(),use_cuda=True)
    #pyroseq2seq = Seq2Seq(dataset.getSrcWord2Index(), dataset.getTgtWord2Index(), use_cuda=True)
    #rnnsearch = RNNSearch(dataset.getSrcWord2Index(), dataset.getTgtWord2Index(), use_cuda=True)

    #classical approach
    #seq2seq = Seq2Seq(dataset.getSrcWord2Index(), dataset.getTgtWord2Index(), use_cuda=True)
    model = make_model(dataset.getSrcWord2Index(), dataset.getTgtWord2Index())

    #print(vnmt)
    #vnmt = vnmt.cuda()
    # setup optimizer
    adam_params = {"lr": 0.003, #"betas": (args.beta1, args.beta2),
                   "clip_norm": 20.0, "lrd": .99996,
                   "weight_decay": 0.0}
    adam = ClippedAdam(adam_params)
    # setup inference algorithm
    elbo = JitTrace_ELBO() if False else Trace_ELBO()
    pad_indx = model.trg_embed.getWord2Index('<PAD>')
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_compute = SimpleLossCompute(model.generator, nn.NLLLoss(reduction="sum",ignore_index=pad_indx), opt=optim) 
    #svi = SVI(vnmt.model, vnmt.guide, adam, loss=elbo)
    #svi = SVI(seq2seq.model, seq2seq.guide, adam, loss=elbo)
    #svi = SVI(rnnsearch.model, rnnsearch.guide, adam, loss=elbo)

    #Classical pytorch approach 

    epochs = 2 
    print(len(dataloader))
    print(len(dataset))
    #seq2seq.train()
    for e in range(0, epochs):
        tot_loss = 0.0
        run_epoch(dataloader, model, loss_compute)
        #for i, b in enumerate(dataloader):
            #print(b)
            #l = svi.step(b[0], b[1])
            #optim.zero_grad()
            #loss = seq2seq( b[0], b[1]) 
            #loss.backward()
            #optim.step()
            #tot_loss += loss.item() * len(b[0])
            
            #print('batch {}, loss {}'.format(i, l))
            #loss += l
        print(tot_loss / len(dataset))
    
    to_translate = [dataset[i][0] for i in range(2) ]
    print('Original Sentences')
    for sent in to_translate:
        print(sent)
    print('Greedy Decoding Translation')
    greedy_trans = GreedyDecodingTranslation(model, to_translate)
    for sent in greedy_trans:
        print(' '.join([model.y_embed.getIndex2Word(s) for s in sent]))
    print('Simple Greedy Decoding Translation')
    greedy_trans = SimpleGreedyDecodingTranslation(model, to_translate)
    for sent in greedy_trans:
        print(' '.join([model.y_embed.getIndex2Word(s) for s in sent]))
    #print('Presumably Beam Search Decoding Translation')
    #beam_trans = BeamDecodingTranslation(seq2seq, to_translate, 3)
    #for sent in beam_trans:
    #    print(' '.join([seq2seq.y_embed.getIndex2Word(s) for s in sent]))




    model = None
    optimizer = None
    loss = None
    #train(dataloader, model, optimizer, loss)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--L1', dest='L1', default='en.txt',
            help="Path to file of first language")
    parser.add_argument('--L2', dest='L2', default='noiseEn.txt',
            help="Path to file of 2nd language")
    
    args = parser.parse_args()
    main(args)

