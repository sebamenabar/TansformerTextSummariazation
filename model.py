import math
import torch
import torch.nn as nn
from torch.functional import F


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = nn.Parameter(torch.zeros(max_len, d_model), requires_grad=False)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class CustomEncoderLayer(nn.TransformerEncoderLayer):
    def forward(self, src, src_mask=None):
        r"""Pass the input through the endocder layer.

        Args:
            src: the sequnce to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).

        Shape:
            see the docs in Transformer class.
        """
        src2 = self.self_attn(src, src, src, key_padding_mask=src_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class CustomDecoder(nn.TransformerEncoder):
    def forward(self, tgt, memory, tgt_padding_mask=None, tgt_attn_mask=None, memory_mask=None):
        r"""Pass the inputs (and mask) through the decoder layer in turn.

        Args:
            tgt: the sequence to the decoder (required).
            memory: the sequnce from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = tgt

        for i in range(self.num_layers):
            output = self.layers[i](output, memory, tgt_attn_mask=tgt_attn_mask,
                                    tgt_padding_mask=tgt_padding_mask, memory_mask=memory_mask)

        if self.norm:
            output = self.norm(output)

        return output


class CustomDecoderLayer(nn.TransformerDecoderLayer):
    def forward(self, tgt, memory, tgt_attn_mask=None,
                tgt_padding_mask=None, memory_mask=None):
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequnce from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).

        Shape:
            see the docs in Transformer class.
        """
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_attn_mask,
                              key_padding_mask=tgt_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(
            tgt, memory, memory, key_padding_mask=memory_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


class CustomTransformer(nn.Transformer):
    def forward(self, src, tgt, src_mask=None, tgt_attn_mask=None,
                tgt_padding_mask=None, memory_mask=None):

        if src.size(1) != tgt.size(1):
            raise RuntimeError("the batch number of src and tgt must be equal")

        if src.size(2) != self.d_model or tgt.size(2) != self.d_model:
            raise RuntimeError(
                "the feature number of src and tgt must be equal to d_model")

        memory = self.encoder(src, src_mask)
        output = self.decoder(tgt, memory, tgt_attn_mask=tgt_attn_mask,
                              tgt_padding_mask=tgt_padding_mask, memory_mask=memory_mask)
        return output


class WTransformer(nn.Module):
    def __init__(self, src_emb, tgt_emb, tgt_vocab_size, d_model=300, nhead=10, num_encoder_layers=6, num_decoder_layers=6,
                 dim_feedforward=1024, dropout=0.1,):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward

        encoder = nn.TransformerEncoder(
            CustomEncoderLayer(d_model, nhead, dim_feedforward, dropout), num_encoder_layers)
        decoder = CustomDecoder(
            CustomDecoderLayer(d_model, nhead, dim_feedforward, dropout), num_decoder_layers)
        self.transformer = CustomTransformer(
            d_model=d_model,
            nhead=nhead,
            dropout=0.1,
            custom_encoder=encoder,
            custom_decoder=decoder,
        )
        self.pe = PositionalEncoding(d_model, dropout)
        self.src_emb = src_emb
        self.tgt_emb = tgt_emb
        self.vocab_ff = nn.Linear(d_model, tgt_vocab_size)

    def generate_square_subsequent_mask(self, sz):
        return self.transformer.generate_square_subsequent_mask(sz)

    def encode(self, src, src_mask=None):
        src = self.src_emb(src) * math.sqrt(self.d_model)
        src = self.pe(src)
        return self.transformer.encoder(src, src_mask)

    def decode(self, tgt, memory, tgt_attn_mask=None,
               tgt_padding_mask=None, memory_mask=None):
        tgt = self.tgt_emb(tgt) * math.sqrt(self.d_model)
        tgt = self.pe(tgt)
        return self.transformer.decoder(tgt, memory, tgt_attn_mask=tgt_attn_mask,
                                        tgt_padding_mask=tgt_padding_mask, memory_mask=memory_mask)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        src = self.src_emb(src) * math.sqrt(self.d_model)
        src = self.pe(src)
        tgt = self.tgt_emb(tgt) * math.sqrt(self.d_model)
        tgt = self.pe(tgt)

        return self.vocab_ff(self.transformer(src, tgt, src_mask=src_mask, tgt_attn_mask=tgt_mask, memory_mask=src_mask))
