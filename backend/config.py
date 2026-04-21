import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CONFIG = {
    "en_vocab_path":          os.path.join(BASE_DIR, "data", "processed", "en_vocab.pkl"),
    "de_vocab_path":          os.path.join(BASE_DIR, "data", "processed", "de_vocab.pkl"),
    "seq2seq_en_de_path":     os.path.join(BASE_DIR, "models", "seq2seq_no_attn_en_de.pt"),
    "seq2seq_de_en_path":     os.path.join(BASE_DIR, "models", "seq2seq_no_attn_de_en.pt"),
    "transformer_en_de_path": os.path.join(BASE_DIR, "models", "transformer_en_de.pt"),
    "transformer_de_en_path": os.path.join(BASE_DIR, "models", "transformer_de_en.pt"),
    "seq2seq": {
        "embed_dim":  256,
        "hidden_dim": 512,
        "num_layers": 2,
        "dropout":    0.5,
    },
    "transformer": {
        "d_model":    256,
        "nhead":      8,
        "num_layers": 3,
        "dim_ff":     512,
        "dropout":    0.1,
    },
    "max_len":   50,
    "beam_size": 4,
    "PAD_IDX": 0,
    "SOS_IDX": 1,
    "EOS_IDX": 2,
    "UNK_IDX": 3,
}