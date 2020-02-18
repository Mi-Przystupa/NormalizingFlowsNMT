import torch
import torch.nn as nn
import numpy as np
import glob
import logging
from pyro.distributions import TransformedDistribution
from models.layers.CondIAF import CondInverseAutoregressiveFlowStable
from Translator import lookup_words


def init_logger(fn=None, cont=False):

    # !!! here
    from imp import reload # python 2.x don't need to import reload, use it directly
    reload(logging)
    filemode = 'a' if cont else 'w' 
    logging_params = {
        'level': logging.INFO,
        'format': '%(asctime)s %(name)-12s %(levelname)-8s %(message)s',#'%(asctime)s__[%(levelname)s, %(module)s.%(funcName)s](%(name)s)__[L%(lineno)d] %(message)s',
        'datefmt': '%m-%d %H:%M', 
        'filemode': filemode
    }

    if fn is not None:
        logging_params['filename'] = fn

    logging.basicConfig(**logging_params)
    logging.error('init basic configure of logging success')
    

def generate_flow_samples(model, data_iter,src_vocab, trg_vocab, n=10000, src_bpe=None, trg_bpe=None, num_sents=10):
    #model: trained NormalizingFlowsEncoderDecoder model
    #data_iter: list of Batch(...) entries (lookg at trainer code for examples)
    #TODO: this... actually has to be 1 sentence per "batch" to work
    results = {}

    for indx, batch in enumerate(data_iter, 1):
        with torch.no_grad():
            distribution = model.getVariationalDistribution(
                batch.src, batch.trg,
                batch.src_mask, batch.trg_mask,
                batch.src_lengths, batch.trg_lengths,
                batch.trg_y
            )
            if isinstance(distribution, TransformedDistribution):
                base_dist = distribution.base_dist
                z_k = base_dist.sample((n,)).squeeze(1).detach() # B x event_shape => n x B x event_shape, so we assume batch = 1 so just get rid of it
                samples = {'z_0': z_k.cpu().numpy()}
                i = 1 
                for  t in distribution.transforms:
                    if isinstance(t, CondInverseAutoregressiveFlowStable):
                        #TODO this is a hack because the context dims != sample dims ...another fix would be you know, repeat the batch or something
                        t.context = t.context.repeat(n, 1)

                    z_k = t(z_k)
                    samples['z_{}'.format(i)] = z_k.detach().cpu().detach()
                    i += 1
            else:
                base_dist = distribution
                samples = {'z_0': base_dist.sample((n,)).squeeze(1).detach().cpu().numpy()}

        src = lookup_words(batch.src.squeeze()[1:], src_vocab)
        src = src if src_bpe is None else src_bpe.DecodePieces(src)
        trg = lookup_words(batch.trg.squeeze()[1:], trg_vocab)
        trg = trg if trg_bpe is None else trg_bpe.DecodePieces(trg)
        results[indx] = {'samples': samples, 'src': src, 'trg': trg}
        if indx > num_sents:
            break
    return results 

def combineResults(dataset):
    pth = './.results/{}/'.format(dataset)
    experiments = [f for f in glob.glob(pth + '*') if (not '.json' in f)]
    results = {}
    for exp in experiments:
        name = exp.split('/')[-1]
        results[name] = {}
        results[name]['perplexities'] = torch.load(exp + '/perplexities.pth')
        with open(exp + '/bleus.json', 'r') as f:
            bleus = json.load(f)
            for k, v in bleus.items():
                results[name][k] = v
        
        with open(exp + '/arguments.json', 'r') as f:
            arguments = json.load(f)
            for k, v in arguments.items():
                results[name][k] = v
    with open(pth + '{}_results.json'.format(dataset), 'w') as f:
        json.dump(results, f, indent=4)


def filter_fn(x, max_len):
    return len(vars(x)['src']) <= max_len and len(vars(x)['trg']) <= max_len

def greedy_decode(model, src, src_mask, src_lengths, max_len=100, sos_index=1, eos_index=None):
    """Greedily decode a sentence."""

    with torch.no_grad():
        encoder_hidden, encoder_final = model.encode(src, src_mask, src_lengths)
        prev_y = torch.ones(1, 1).fill_(sos_index).type_as(src)
        trg_mask = torch.ones_like(prev_y)
    output = []
    attention_scores = []
    hidden = None

    for i in range(max_len):
        with torch.no_grad():
            out, hidden, pre_output = model.decode(
              encoder_hidden, encoder_final, src_mask,
              prev_y, trg_mask, hidden)
            # we predict from the pre-output layer, which is
            # a combination of Decoder state, prev emb, and context
            prob = model.callGenerator(pre_output[:, -1])

        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data.item()
        output.append(next_word)
        prev_y = torch.ones(1, 1).type_as(src).fill_(next_word)
        attention_scores.append(model.decoder.attention.alphas.cpu().numpy())
    
    output = np.array(output)
        
    # cut off everything starting from </s> 
    # (only when eos_index provided)
    if eos_index is not None:
        first_eos = np.where(output==eos_index)[0]
        if len(first_eos) > 0:
            output = output[:first_eos[0]]      
    
    return output, np.concatenate(attention_scores, axis=1)


def lookup_words(x, vocab=None):
    if vocab is not None:
        x = [vocab.itos[i] for i in x]

    return [str(t) for t in x]

def print_examples(example_iter, model, n=2, max_len=100, 
                   sos_index=1, 
                   src_eos_index=None, 
                   trg_eos_index=None, 
                   src_vocab=None, trg_vocab=None, EOS_TOKEN='</s>', SOS_TOKEN="<s>"):
    """Prints N examples. Assumes batch size of 1."""

    model.eval()
    count = 0
    print()
    
    if src_vocab is not None and trg_vocab is not None:
        src_eos_index = src_vocab.stoi[EOS_TOKEN]
        trg_sos_index = trg_vocab.stoi[SOS_TOKEN]
        trg_eos_index = trg_vocab.stoi[EOS_TOKEN]
    else:
        src_eos_index = None
        trg_sos_index = 1
        trg_eos_index = None
        
    for i, batch in enumerate(example_iter):
      
        src = batch.src.cpu().numpy()[0, :]
        trg = batch.trg_y.cpu().numpy()[0, :]

        # remove </s> (if it is there)
        src = src[:-1] if src[-1] == src_eos_index else src
        trg = trg[:-1] if trg[-1] == trg_eos_index else trg      
      
        result, _ = greedy_decode(
          model, batch.src, batch.src_mask, batch.src_lengths,
          max_len=max_len, sos_index=trg_sos_index, eos_index=trg_eos_index)
        print("Example #%d" % (i+1))
        print("Src : ", " ".join(lookup_words(src, vocab=src_vocab)))
        print("Trg : ", " ".join(lookup_words(trg, vocab=trg_vocab)))
        print("Pred: ", " ".join(lookup_words(result, vocab=trg_vocab)))
        print()
        
        count += 1
        if count == n:
            break