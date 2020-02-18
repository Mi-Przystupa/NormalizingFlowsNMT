import spacy
from nltk.tokenize import ToktokTokenizer
from araNorm import araNorm
import sentencepiece as spm

def tokenize_spacy(text, lang):
    #assumed to be a spacy model
    return [tok.text for tok in lang.tokenizer(text)]

def toktok_tokenize(text, lang):
    return lang.tokenize(text)

def bpe_tokenize(text, model, transform=None):
    if transform is not None:
        text = transform(text)
    return model.EncodeAsPieces(text.strip())

def pickTokenizer(lang, on_whitespace):
    #it's axtually redundant  condition
    if on_whitespace:
        return lambda x: x.split()
    elif (lang == 'en' or lang == 'de') and not on_whitespace:
        tokenizer = spacy.load(lang)
        return lambda x: tokenize_spacy(x, tokenizer)
    else:
        tokenizer = ToktokTokenizer()
        return lambda x: toktok_tokenize(x, tokenizer)

def getTokenizer(source, target, on_whitespace):
    return pickTokenizer(source, on_whitespace), pickTokenizer(target, on_whitespace)

def getBPE(src_pth, trg_pth):
    src_bpe = spm.SentencePieceProcessor()
    src_bpe.Load(src_pth)
    trg_bpe = spm.SentencePieceProcessor()
    trg_bpe.Load(trg_pth)

    src_transform = None
    trg_transform = None

    src_m = src_pth.split('/')[-1]
    trg_m = trg_pth.split('/')[-1]
    if 'arabic' in src_m:
        arab_norm = araNorm()
        src_transform = lambda x: arab_norm.run(x) 

    if 'arabic' in trg_m:
        arab_norm = araNorm()
        trg_transform = lambda x: arab_norm.run(x)

    src_tokenizer = lambda x: bpe_tokenize(x, src_bpe, src_transform)
    trg_tokenizer = lambda x: bpe_tokenize(x, trg_bpe, trg_transform)
    return src_tokenizer, trg_tokenizer