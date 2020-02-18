import torch.nn as nn
import torch

#Modified code from this tutorial:
#https://colab.research.google.com/github/bastings/annotated_encoder_decoder/blob/master/annotated_encoder_decoder.ipynb#scrollTo=yygWWAJ9oBsT
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
        
    #def forward(self, src, trg, src_mask, trg_mask, src_lengths, trg_lengths):
    def forward(self, src_sent, y_sent):
        """Take in and process masked src and target sequences."""
        #encoder_hidden, encoder_final = self.encode(src, src_mask, src_lengths)
        src_embeds, src_lengths, src_mask, y = self.src_embed(src_sent, y_sent)
        src_mask = src_mask.unsqueeze(1)
        encoder_hidden, encoder_final, = self.encode(src_embeds, src_mask, src_lengths)
        indexesAndLens = self.trg_embed.sentences2IndexesAndLens(y_sent)
        pad_seq, trg_mask = self.trg_embed.padAndMask(indexesAndLens)
        trg = self.trg_embed.getBatchEmbeddings(pad_seq)
        out, _, pre_output =  self.decode(encoder_hidden, encoder_final, src_mask, trg, trg_mask.unsqueeze(1))
        return out, _, pre_output, y_sent
        #return self.decode(encoder_hidden, encoder_final, src_mask, trg, trg_mask)
    
    def encode(self, src_embeds, src_mask, src_lengths):
        return self.encoder(src_embeds, src_mask, src_lengths)
        #src_embed, src_mask, src_lengths, _, y
        #return self.encoder(self.src_embed(src), src_mask, src_lengths)
    
    def decode(self, encoder_hidden, encoder_final, src_mask, trg, trg_mask,
               decoder_hidden=None):
        return self.decoder(trg, encoder_hidden, encoder_final,
                            src_mask, trg_mask, hidden=decoder_hidden)

        #return self.decoder(self.trg_embed(trg), encoder_hidden, encoder_final,
        #                    src_mask, trg_mask, hidden=decoder_hidden)
    def getTargets(self, y_sent):
        sequences = self.trg_embed.sentences2IndexesAndLens(y_sent)
        sequences, _ = self.trg_embed.padAndMask(sequences)
        return sequences

    def getTGTSOS(self):
        return self.trg_embed.getSOS()
    def getTGTEOS(self):
        return self.trg_embed.getEOS()
    def getTGTEOSIndx(self):
        return self.trg_embed.getEOSIndex()
    def getTGTSOSIndx(self):
        return self.trg_embed.getSOSIndex()
    def getSRCSOSIndx(self):
        return self.src_embed.getSOSIndex()
    def getSRCSOS(self):
        return self.src_embed.getSOS()