import torch
import torch.nn
from BatchWordEmbeddings import BatchWordEmbeddings
from Layers import Generator, Decoder, Encoder
from AttentionLayers import BahdanauAttention
from EncoderDecoder import EncoderDecoder

def make_model(src_vocab, tgt_vocab, emb_size=300, hidden_size=512, num_layers=1, dropout=0.1):
    "Helper: Construct a model from hyperparameters."

    attention = BahdanauAttention(hidden_size)

    use_cuda = torch.cuda.is_available()
    model = EncoderDecoder(
        Encoder(emb_size, hidden_size, num_layers=num_layers, dropout=dropout),
        Decoder(emb_size, hidden_size, attention, num_layers=num_layers, dropout=dropout),
        BatchWordEmbeddings(src_vocab, emb_dim=emb_size, use_cuda=use_cuda),
        BatchWordEmbeddings(tgt_vocab, emb_dim=emb_size,use_cuda=use_cuda),
        Generator(hidden_size, len(tgt_vocab)))

    return model.cuda() if use_cuda else model