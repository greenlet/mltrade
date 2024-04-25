
import torch
from torch import nn

from mlt.model.transformer.layers import EncoderLayer, DecoderLayer
from mlt.model.transformer.models import PositionalEncoding


class FinEncoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self, inp_dim: int, out_dim: int, n_layers: int, n_head: int, d_k: int, d_v: int,
            d_model: int, d_inner: int, dropout: float = 0.1,
            n_position: int = 15000):

        super().__init__()

        # self.position_enc = PositionalEncoding(inp_dim, n_position=n_position)

        encoders = []
        for i in range(n_layers):
            inp_size = inp_dim if i == 0 else d_model
            out_size = d_model if i < n_layers - 1 else out_dim
            enc = EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout, inp_size=inp_size, out_size=out_size)
            encoders.append(enc)
        self.layer_stack = nn.ModuleList(encoders)
        self.d_model = d_model

    def forward(self, src_seq, src_mask=None, return_attns=False):
        enc_slf_attn_list = []

        # -- Forward
        # enc_output = self.position_enc(src_seq)
        enc_output = src_seq

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=src_mask)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,

