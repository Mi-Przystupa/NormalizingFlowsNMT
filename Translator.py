
import logging
import torch
import numpy as np
from Trainer import rebatch
from pyro.distributions import Categorical
from collections import namedtuple

def lookup_words(x, vocab=None):
    if vocab is not None:
        x = [vocab.itos[i] for i in x]

    return [str(t) for t in x]

def write_translations(name, hypotheses, references):
    with open(name, mode='w', encoding='utf-8') as f:
        f.write('Hypothesis' + '\t' + 'Reference\n')
        for pair in zip(hypotheses, references):
            f.write(pair[0].strip() + ' \t '  + pair[1].strip() + '\n')
    logging.info('Wrote translation to :{}'.format(name))

class Translator:
    def __init__(self, dataset, iterator, model, max_len=100, sos_index=1, eos_index=None,
        pad_index=None, use_cuda=False, k = 10, length_norm=False, do_lang_model=False):
        self.dataset = dataset
        self.references = [" ".join(example.src if do_lang_model else example.trg) for example in self.dataset]

        self.iterator = iterator
        self.hypotheses = []
        self.model = model
        self.max_len = max_len
        self.sos_index = sos_index
        self.eos_index = eos_index
        self.pad_index = pad_index
        self.do_lang_model = do_lang_model
        #beam search things
        self.k = k #number of beams to consider
        self.length_norm = length_norm
        self.trg_vocab_size = -1

        self.use_cuda = use_cuda

    def encodeSent(self, src, src_mask, src_lengths):
        self.model.eval()
        with torch.no_grad():
            encoder_hidden, encoder_final = self.model.encode(src, src_mask, src_lengths)
        return encoder_hidden, encoder_final #, prev_y, trg_mask

    def decodeSent(self, encoder_hidden, encoder_final, src_mask, prev_y, trg_mask, hidden):
        with torch.no_grad():
            _ , hidden, pre_output = self.model.decode(
            encoder_hidden, encoder_final, src_mask,
            prev_y, trg_mask, hidden)
        return hidden, pre_output
    
    def getTokRepresentation(self, tok, type_as, ret_mask=False):
        tok = torch.ones(1, 1).fill_(tok).type_as(type_as)
        if ret_mask:
            mask = torch.ones_like(tok)
            return tok, mask
        else:
            return tok

    def beam_search(self, src, src_mask, src_lengths):
        #encoder_hidden, encoder_final, prev_y, trg_mask = self.encodeSent(src, src_mask, src_lengths)       
        encoder_hidden, encoder_final = self.encodeSent(src, src_mask, src_lengths)
        prev_y, trg_mask  = self.getTokRepresentation(self.sos_index, src, ret_mask=True)

        eosindx = self.eos_index
        k = self.k
        #Create beams
        Beam = namedtuple('Beam', 'seq mask score hidden attn_list')
        beams = [Beam([prev_y], trg_mask, 0.0, None, [])]
        translations = []
        #hackery to track vocabularly for each beam as it is expanded
        voc_ind = None
        beam_inds = None #this is supposed to track the beams indices across expanding search tree
        while k > 0:
            scores = []
            attns = []
            hiddens = []
            # for each candidate beam
            for i, b in enumerate(beams):
                seq, mask, score, hidden, attn_list = b #unroll Beam parameters 
                trg_mask = mask
                prev_y = seq[-1] #current word is at end of sequence

                #decode last word in sequence
                hidden, pre_output = self.decodeSent( encoder_hidden, encoder_final, src_mask,
                    prev_y, trg_mask, hidden)

                #get probability of all word in vocabularly, internally is log probability
                prob = self.model.callGenerator(pre_output[:, -1])

                #new score is logp(x_1,x_2,...,x_t) + log p(x_t+1 | x<t) ...
                new_score = score + prob 
                scores.append(new_score)
                #Reallly should be a tuple...anyways initialize the vocab indices to align with expanded nodes for all beams
                if voc_ind is None:
                    voc_ind = torch.arange(0, new_score.size(-1) , out=torch.zeros_like(new_score)).repeat(1, len(beams)).view(-1).long()
                if beam_inds is None:
                    beam_inds = torch.zeros_like(new_score).repeat(len(beams), 1).long()

                beam_inds[i] = i #set row of expansions to be the current beam
                if self.model.decoder.attention is not None and not self.do_lang_model:
                    attns.append(self.model.decoder.attention.alphas) #track attention for this beam
                else:
                    attns.append(torch.zeros_like(hidden))
                hiddens.append(hidden.clone())
                
            #sort candidates by most likely
            #flatten all things so the indices all align...hopefully
            scores = torch.cat(scores).view(-1)
            top_scores, indices = torch.topk(scores, k)
            beam_inds = beam_inds.view(-1)
            new_beams = []
            for i,new_score in zip(indices, top_scores):
                #get the...hopefully corresponding indices from the top scoring expansions
                word_ind = voc_ind[i]
                beam_ind = beam_inds[i]
                next_y = self.getTokRepresentation(word_ind, src)

                seq, mask, _, _, attn_list = beams[beam_ind.item()]
                new_seq = seq + [next_y]
                hidden = hiddens[beam_ind].clone()
                attn_list.append(attns[beam_ind])

                #first token in sequences is sos, which we do not include in final translation so don't count it
                if word_ind == eosindx or (len(new_seq) - 1) >= self.max_len:
                    #exclude eos tokens in translation
                    to_add = seq if word_ind == eosindx else new_seq
                    candidate = Beam(to_add, mask, new_score, hidden, attn_list)
                    k = k - 1
                    translations.append(candidate)
                else:
                    candidate = Beam(new_seq, mask, new_score, hidden, attn_list)
                    new_beams.append(candidate)
            """ 
            if k != len(new_beams):
                #if k != len(new_beams), means the dimensions of our arrays have changed so we need to recalculate them
                voc_ind = None
                beam_ind = None
            else:
                try:
                    beam_inds.view(k, self.trg_vocab_size)
                except RuntimeError as e:
                    beam_inds = None
                    voc_ind = None
                    #means we couldn't resize beam_inds
            """
            beam_inds, voc_ind = None, None
            #set new beams
            beams = new_beams

        #pick most likely translation
        if self.length_norm:
            criteria = lambda x: x.score / (len(x.seq) - 1) #normalize by length, supposedly helps with translations
        else:
            criteria = lambda x: x.score

        #TODO: I need to change this some how....
        output = max(translations,key= criteria)
        #first token is the sos tok so do not include it
        translation = np.array([o.item() for o in output.seq][1:])
        try:
            attention = torch.cat(output.attn_list, dim=1)
            attention = attention.cpu().numpy()
        except TypeError as e:
            #means we weren't collecting attention ouptuts for one reason or another
            attention = output.attn_list
        return translation, attention

    def greedy_decode(self, src, src_mask, src_lengths ):
        """Greedily decode a sentence."""
        encoder_hidden, encoder_final = self.encodeSent(src, src_mask, src_lengths)
        prev_y, trg_mask = self.getTokRepresentation(self.sos_index, src, ret_mask=True)

        output = []
        attention_scores = []
        hidden = None

        for i in range(self.max_len):
            hidden, pre_output = self.decodeSent(encoder_hidden, encoder_final, src_mask,
                    prev_y, trg_mask, hidden)

            # we predict from the pre-output layer, which is
            # a combination of Decoder state, prev emb, and context
            prob = self.model.callGenerator(pre_output[:, -1])

            _, next_word = torch.max(prob, dim=1)
            next_word = next_word.data.item()
            output.append(next_word)
            prev_y = torch.ones(1, 1).type_as(src).fill_(next_word)
            if self.model.decoder.attention is not None and not self.do_lang_model:
                attention_scores.append(self.model.decoder.attention.alphas.cpu().numpy())
            else:
                #no attention do not store
                attention_scores.append(torch.zeros_like(hidden).cpu().numpy())
        
        output = np.array(output)
            
        # cut off everything starting from </s> 
        # (only when eos_index provided)
        if self.eos_index is not None:
            first_eos = np.where(output==self.eos_index)[0]
            if len(first_eos) > 0:
                output = output[:first_eos[0]]      
        
        return output, np.concatenate(attention_scores, axis=1)
    def sample_TRG_sentence(self, src_lengths):
        num_words = self.model.trg_embed.weight.size(0)
        prob = 1. / num_words 
        distr = Categorical(probs=torch.tensor([prob for _ in range(num_words)]))
        trgs = [[self.sos_index] for _ in src_lengths ]
        for i, s in enumerate(src_lengths):
            for _ in range(s):
                trgs[i] = trgs[i] + [distr.sample().item()]

        return src_lengths.new_tensor(trgs).long()
    def ExpectationMaximization_decoding(self, src, src_mask, src_lengths, decoding_alg,
        trg=None, trg_mask=None, trg_lengths=None):
        if trg is not None and trg_mask is not None and trg_lengths is not None:
            self.model.setTRGSentence(trg)
            return decoding_alg(src, src_mask, src_lengths) 
        else:
            init = self.sample_TRG_sentence(src_lengths)
            self.model.setTRGSentence(init, src_mask, src_lengths)

        for i in range(0, 20):
            pred, attention = decoding_alg(src, src_mask, src_lengths)
            assert len(src) == 1, "this only works for translating 1 word at a time"
            pred = src_lengths.new_tensor(torch.from_numpy(pred)).long().view(1, -1)
            trg_lengths = src_lengths.new_tensor([pred.size(1)])
            trg_mask = src_mask.new_ones(src_mask.size(0), src_mask.size(1), pred.size(1))
            self.model.setTRGSentence(pred, trg_mask, trg_lengths)

        return pred[0], attention
    
    def translate(self, decodefn='greedy', do_em=False):
        if decodefn == 'greedy': 
            decoding_fn = self.greedy_decode
            msg = 'Doing Greedy Search'
        elif decodefn == 'beamsearch':
            msg = 'Doing Beam Search with k = {}'.format(self.k)
            decoding_fn = self.beam_search
        else:
            msg = 'sorry, {} not supported, switching to greedy. choices: greedy or beamsearch'.format(decodefn)
            decodefn = 'greedy'
            decoding_fn = self.greedy_decode
        logging.info(msg)
        print(msg)

        if do_em:
            decoding_fn = lambda s, s_m, s_l: self.ExpectationMaximization_decoding(s, s_m, s_l, self.greedy_decode)

        self.hypotheses = []
        alphas = []  # save the last attention scores
        #otherwise dropout messes up translation process
        self.model.eval()
        for batch in self.iterator:
            batch = rebatch(self.pad_index, batch, use_cuda=self.use_cuda)
            pred, attention = decoding_fn(batch.src, batch.src_mask, batch.src_lengths) 
            self.hypotheses.append(pred)
            alphas.append(attention)
        self.model.train()
        return self.hypotheses

    def FullEvalTranslate(self, trg_vocab, bleu_func,  decodefn='greedy', bpe_model=None):
        #these other functions set self.hypotheses right now so no need to do it here 
        self.trg_vocab_size = len(trg_vocab) #track actual size of trg vocabularly
        hypotheses = self.translate(decodefn=decodefn, do_em=False)
        hypotheses = self.convertHypothesisToSentences(trg_vocab)
        if bpe_model is not None:
            references = [bpe_model.DecodePieces(r.split()) for r in self.references]
            hypotheses = [bpe_model.DecodePieces(h.split()) for h in hypotheses] 
        references = references.copy()
        bleu_score = bleu_func(hypotheses, references)
        return bleu_score, hypotheses, references
    
    def convertHypothesisToSentences(self, vocab,  hypotheses=None):
        if hypotheses is None:
            hypotheses = self.hypotheses

        hypotheses = [lookup_words(x, vocab) for x in hypotheses]
        
        return [" ".join(h) for h in hypotheses ]



    def getHypothesis(self):
        return self.hypotheses


