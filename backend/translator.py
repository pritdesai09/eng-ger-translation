import torch
import pickle
import spacy
from backend.models_arch import Encoder, Decoder, Seq2Seq, TranslatorTransformer
from backend.config import CONFIG

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Load spaCy tokenizers ─────────────────────────────────────────────────────
nlp_en = spacy.load("en_core_web_sm", disable=["parser", "ner"])
nlp_de = spacy.load("de_core_news_sm", disable=["parser", "ner"])

def tokenize(text, lang):
    nlp = nlp_en if lang == 'en' else nlp_de
    return [tok.text.lower() for tok in nlp.tokenizer(str(text).strip())]

# ── Load vocabularies ─────────────────────────────────────────────────────────
def load_vocab(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data['word2idx'], data['idx2word']

en_w2i, en_i2w = load_vocab(CONFIG['en_vocab_path'])
de_w2i, de_i2w = load_vocab(CONFIG['de_vocab_path'])

# ── Build and load Seq2Seq ────────────────────────────────────────────────────
def _build_seq2seq(src_vocab_sz, tgt_vocab_sz):
    cfg = CONFIG['seq2seq']
    enc = Encoder(src_vocab_sz, cfg['embed_dim'], cfg['hidden_dim'],
                  cfg['num_layers'], cfg['dropout'])
    dec = Decoder(tgt_vocab_sz, cfg['embed_dim'], cfg['hidden_dim'],
                  cfg['num_layers'], cfg['dropout'])
    return Seq2Seq(enc, dec, DEVICE).to(DEVICE)

seq2seq_en_de = _build_seq2seq(len(en_w2i), len(de_w2i))
seq2seq_de_en = _build_seq2seq(len(de_w2i), len(en_w2i))
seq2seq_en_de.load_state_dict(torch.load(CONFIG['seq2seq_en_de_path'], map_location=DEVICE))
seq2seq_de_en.load_state_dict(torch.load(CONFIG['seq2seq_de_en_path'], map_location=DEVICE))
seq2seq_en_de.eval()
seq2seq_de_en.eval()

# ── Build and load Transformer ────────────────────────────────────────────────
def _build_transformer(src_vocab_sz, tgt_vocab_sz):
    cfg = CONFIG['transformer']
    return TranslatorTransformer(src_vocab_sz, tgt_vocab_sz,
                                 cfg['d_model'], cfg['nhead'],
                                 cfg['num_layers'], cfg['num_layers'],
                                 cfg['dim_ff'], cfg['dropout']).to(DEVICE)

transformer_en_de = _build_transformer(len(en_w2i), len(de_w2i))
transformer_de_en = _build_transformer(len(de_w2i), len(en_w2i))
transformer_en_de.load_state_dict(torch.load(CONFIG['transformer_en_de_path'], map_location=DEVICE))
transformer_de_en.load_state_dict(torch.load(CONFIG['transformer_de_en_path'], map_location=DEVICE))
transformer_en_de.eval()
transformer_de_en.eval()

print(f"✅ All models loaded on {DEVICE}")

# ── Inference: Seq2Seq ────────────────────────────────────────────────────────
SOS, EOS, UNK = 1, 2, 3

def _seq2seq_translate(model, tokens, src_w2i, tgt_i2w):
    ids = [SOS] + [src_w2i.get(t, UNK) for t in tokens] + [EOS]
    src = torch.tensor(ids, dtype=torch.long).unsqueeze(1).to(DEVICE)
    with torch.no_grad():
        hidden, cell = model.encoder(src)
        dec_in = torch.tensor([SOS], dtype=torch.long).to(DEVICE)
        result = []
        for _ in range(CONFIG['max_len']):
            out, hidden, cell = model.decoder(dec_in, hidden, cell)
            pred = out.argmax(1).item()
            if pred == EOS:
                break
            result.append(tgt_i2w.get(pred, '<unk>'))
            dec_in = torch.tensor([pred], dtype=torch.long).to(DEVICE)
    return ' '.join(result)

# ── Inference: Transformer (Beam Search) ─────────────────────────────────────
def _transformer_translate(model, tokens, src_w2i, tgt_i2w, beam_size=4):
    ids = [SOS] + [src_w2i.get(t, UNK) for t in tokens] + [EOS]
    src = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        memory, mem_mask = model.encode(src)
        beams     = [(0.0, [SOS])]
        completed = []
        import torch.nn.functional as F
        for _ in range(CONFIG['max_len']):
            new_beams = []
            for score, toks in beams:
                tgt_t  = torch.tensor(toks, dtype=torch.long).unsqueeze(0).to(DEVICE)
                logits = model.decode_step(tgt_t, memory, mem_mask)
                lp     = F.log_softmax(logits[0, -1], dim=-1)
                topk   = lp.topk(beam_size)
                for lp_val, idx in zip(topk.values, topk.indices):
                    new_beams.append((score + lp_val.item(), toks + [idx.item()]))
            new_beams.sort(key=lambda x: x[0], reverse=True)
            beams = []
            for s, toks in new_beams[:beam_size]:
                if toks[-1] == EOS:
                    completed.append((s / len(toks), toks))
                else:
                    beams.append((s, toks))
            if not beams:
                break
        best = max(completed if completed else beams, key=lambda x: x[0])[1]
    return ' '.join([tgt_i2w.get(i, '<unk>') for i in best[1:] if i not in (SOS, EOS)])

# ── Public interface ──────────────────────────────────────────────────────────
def translate(text: str, model_type: str, direction: str) -> dict:
    """
    model_type: 'seq2seq' | 'transformer'
    direction:  'en_de'   | 'de_en'
    """
    if direction == 'en_de':
        tokens, src_w2i, tgt_i2w = tokenize(text, 'en'), en_w2i, de_i2w
    else:
        tokens, src_w2i, tgt_i2w = tokenize(text, 'de'), de_w2i, en_i2w

    if model_type == 'seq2seq':
        model  = seq2seq_en_de if direction == 'en_de' else seq2seq_de_en
        result = _seq2seq_translate(model, tokens, src_w2i, tgt_i2w)
    else:
        model  = transformer_en_de if direction == 'en_de' else transformer_de_en
        result = _transformer_translate(model, tokens, src_w2i, tgt_i2w)

    return {
        "translation": result,
        "source_tokens": tokens,
        "model": model_type,
        "direction": direction
    }