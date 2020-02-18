import torch
import torch.nn
import copy

def BeamSearch(model, sent, k=5, max_length=20):
    model.eval()
    with torch.no_grad():
        #TODO I'm not sure if this will actually work on the model....needs to be tested
        encoding = model.encode(sent)
    eosindx = model.getTGTEOSIndx()
    sosindx = model.getTGTSOSIndx()

    #TODO one worry is whether or not i'm passing aorund th esame hidden state every where...
    sequences = [[[sosindx], 1.0, encoding.detach()]] 
    k = k
    translations = []
    while k > 0:
        all_candidates = list()
        for s in sequences:
            seq, score, hidden = s[0], s[1], s[2]
            #use last token in sequence
            with torch.no_grad():
                row, hidden = model.decode(hidden, seq[-1])
            for i, r in enumerate(row[0]):
                #we see <SOS> pretty frequently so just ignore it
                if i == sosindx or seq[-1] == i:
                    continue
                #i genuinley do not know if i need to deep copy...but i also won't want these things messing with each other
                candidate = [seq + [i], score * -row[0,i].item(), hidden.detach()]
                if len(candidate[0]) > max_length:
                    k = k - 1
                    translations.append(candidate)
                else:
                    #don't add candidate if we're done with it
                    all_candidates.append(candidate)
        ordered = sorted(all_candidates, key=lambda x: x[1])
        sequences = ordered[:k]
        toremove = []
        for i, s in enumerate(sequences):
            if eosindx == s[0][-1]:
                toremove.append(i)
                translations.append(s)
                k = k - 1

        for i in sorted(toremove, reverse=True):
            del sequences[i]
        
    return max(translations,key=lambda x: x[1])[0]


def BeamDecodingTranslation(model, sentences, k, m_l=8):
    translations = [None] * len(sentences)
    for i, s in enumerate(sentences):
        translations[i] = BeamSearch(model, s, k=k, max_length=m_l)
    return translations

def GreedyDecodingTranslation(model, sentences, m_l=20):
    return BeamDecodingTranslation(model, sentences, k=1,m_l=m_l)

def SimpleGreedyDecodingTranslation(model, sentences, m_l=20):
    model.eval()
    eosindx = model.getTGTEOSIndx()
    sosindx = model.getTGTSOSIndx()
    translations = [None] * len(sentences)
    for i, s in enumerate(sentences):

        with torch.no_grad():
            #TODO I'm not sure if this will actually work on the model....needs to be tested
            encoding = model.encode(s)
        translation = [sosindx]
        log_prob, hidden = model.decode(encoding, curr_token=translation[-1])
        for _ in range(m_l):
            next_tok = torch.argmax(log_prob)
            translation = translation + [next_tok.item()]
            if next_tok == eosindx:
                break
            log_prob, hidden  = model.decode(hidden, next_tok.item())
        translations[i] = translation
    return translations




if __name__ == "__main__":
    from Seq2Seq import Seq2Seq
    sentences = ["I am an apple", "There is a dog", "look a tree", 'here is a realy long sentence just for verifying']
    words = []
    for s in sentences:
        for w in s.split():
            words.append(w)
    words = set(words)
    word2idx = {w:i for i, w in enumerate(list(words))}
    x_words = word2idx
    seq2seq = Seq2Seq(word2idx.copy(), word2idx.copy())
    translation = SimpleGreedyDecodingTranslation(seq2seq, sentences)
    #translation = GreedyDecodingTranslation(seq2seq, sentences)
    print(translation)
    print(len(translation))
    #translation = BeamDecodingTranslation(seq2seq, sentences, 2)
    print(len(translation))
