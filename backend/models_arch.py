"""
Model architecture definitions — must be identical to training notebooks.
"""

import torch
import torch.nn as nn
import math

PAD_IDX = 0

# ══════════════════════════════════════════════════════════════════════════════
# Seq2Seq (No Attention)
# ══════════════════════════════════════════════════════════════════════════════

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.rnn = nn.LSTM(embed_dim, hidden_dim, num_layers,
                           dropout=dropout, batch_first=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        _, (hidden, cell) = self.rnn(embedded)
        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.rnn = nn.LSTM(embed_dim, hidden_dim, num_layers,
                           dropout=dropout, batch_first=False)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt_token, hidden, cell):
        tgt_token = tgt_token.unsqueeze(0)
        embedded  = self.dropout(self.embedding(tgt_token))
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        prediction = self.fc_out(output.squeeze(0))
        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device  = device


# ══════════════════════════════════════════════════════════════════════════════
# Transformer
# ══════════════════════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=512):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TranslatorTransformer(nn.Module):
    def __init__(self, src_vocab_sz, tgt_vocab_sz, d_model=256, nhead=8,
                 num_enc_layers=3, num_dec_layers=3, dim_ff=512, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.src_embedding = nn.Embedding(src_vocab_sz, d_model, padding_idx=0)
        self.tgt_embedding = nn.Embedding(tgt_vocab_sz, d_model, padding_idx=0)
        self.pos_enc = PositionalEncoding(d_model, dropout)
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead,
            num_encoder_layers=num_enc_layers,
            num_decoder_layers=num_dec_layers,
            dim_feedforward=dim_ff, dropout=dropout, batch_first=True
        )
        self.fc_out = nn.Linear(d_model, tgt_vocab_sz)

    def make_src_key_padding_mask(self, src):
        return src == 0

    def make_tgt_mask(self, tgt):
        tgt_len = tgt.shape[1]
        return torch.triu(torch.ones(tgt_len, tgt_len, device=tgt.device), diagonal=1).bool()

    def encode(self, src):
        mask = self.make_src_key_padding_mask(src)
        emb  = self.pos_enc(self.src_embedding(src) * math.sqrt(self.d_model))
        return self.transformer.encoder(emb, src_key_padding_mask=mask), mask

    def decode_step(self, tgt, memory, mem_mask):
        tgt_mask = self.make_tgt_mask(tgt)
        emb = self.pos_enc(self.tgt_embedding(tgt) * math.sqrt(self.d_model))
        out = self.transformer.decoder(emb, memory,
                                       tgt_mask=tgt_mask,
                                       memory_key_padding_mask=mem_mask)
        return self.fc_out(out)