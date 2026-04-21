import torch
import pickle
import spacy
import re
import torch.nn.functional as F
from backend.models_arch import Encoder, Decoder, Seq2Seq, TranslatorTransformer
from backend.config import CONFIG

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[Translator] Device: {DEVICE}")

print("[Translator] Loading spaCy...")
nlp_en = spacy.load("en_core_web_sm", disable=["parser", "ner"])
nlp_de = spacy.load("de_core_news_sm", disable=["parser", "ner"])

def tokenize(text, lang):
    nlp = nlp_en if lang == 'en' else nlp_de
    return [tok.text.lower() for tok in nlp.tokenizer(str(text).strip())]

def load_vocab(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data['word2idx'], data['idx2word']

print("[Translator] Loading vocabularies...")
en_w2i, en_i2w = load_vocab(CONFIG['en_vocab_path'])
de_w2i, de_i2w = load_vocab(CONFIG['de_vocab_path'])
print(f"[Translator] EN: {len(en_w2i)} | DE: {len(de_w2i)}")

_models = {}

def _get_model(key):
    if key in _models:
        return _models[key]
    print(f"[Translator] Loading model: {key}")
    cfg_s = CONFIG['seq2seq']
    cfg_t = CONFIG['transformer']

    if key == 'seq2seq_en_de':
        enc   = Encoder(len(en_w2i), cfg_s['embed_dim'], cfg_s['hidden_dim'], cfg_s['num_layers'], cfg_s['dropout'])
        dec   = Decoder(len(de_w2i), cfg_s['embed_dim'], cfg_s['hidden_dim'], cfg_s['num_layers'], cfg_s['dropout'])
        model = Seq2Seq(enc, dec, DEVICE).to(DEVICE)
        model.load_state_dict(torch.load(CONFIG['seq2seq_en_de_path'], map_location=DEVICE))
    elif key == 'seq2seq_de_en':
        enc   = Encoder(len(de_w2i), cfg_s['embed_dim'], cfg_s['hidden_dim'], cfg_s['num_layers'], cfg_s['dropout'])
        dec   = Decoder(len(en_w2i), cfg_s['embed_dim'], cfg_s['hidden_dim'], cfg_s['num_layers'], cfg_s['dropout'])
        model = Seq2Seq(enc, dec, DEVICE).to(DEVICE)
        model.load_state_dict(torch.load(CONFIG['seq2seq_de_en_path'], map_location=DEVICE))
    elif key == 'transformer_en_de':
        model = TranslatorTransformer(len(en_w2i), len(de_w2i),
                    cfg_t['d_model'], cfg_t['nhead'], cfg_t['num_layers'],
                    cfg_t['num_layers'], cfg_t['dim_ff'], cfg_t['dropout']).to(DEVICE)
        model.load_state_dict(torch.load(CONFIG['transformer_en_de_path'], map_location=DEVICE))
    elif key == 'transformer_de_en':
        model = TranslatorTransformer(len(de_w2i), len(en_w2i),
                    cfg_t['d_model'], cfg_t['nhead'], cfg_t['num_layers'],
                    cfg_t['num_layers'], cfg_t['dim_ff'], cfg_t['dropout']).to(DEVICE)
        model.load_state_dict(torch.load(CONFIG['transformer_de_en_path'], map_location=DEVICE))

    model.eval()
    _models[key] = model
    print(f"[Translator] ✅ {key} ready.")
    return model

SOS, EOS, UNK = 1, 2, 3

def detokenize(tokens):
    text = ' '.join(tokens)
    text = re.sub(r'\s([.,!?;:\'])', r'\1', text)
    text = re.sub(r'\(\s', '(', text)
    text = re.sub(r'\s\)', ')', text)
    return text.strip()

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
            word = tgt_i2w.get(pred, '<unk>')
            if word not in ('<unk>', '<pad>'):
                result.append(word)
            dec_in = torch.tensor([pred], dtype=torch.long).to(DEVICE)
    return detokenize(result)

def _transformer_translate(model, tokens, src_w2i, tgt_i2w, beam_size=4):
    ids = [SOS] + [src_w2i.get(t, UNK) for t in tokens] + [EOS]
    src = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        memory, mem_mask = model.encode(src)
        beams, completed = [(0.0, [SOS])], []
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
    result = [tgt_i2w.get(i, '<unk>') for i in best[1:]
              if i not in (SOS, EOS) and tgt_i2w.get(i, '') not in ('<unk>', '<pad>')]
    return detokenize(result)

def translate(text: str, model_type: str, direction: str) -> dict:
    if direction == 'en_de':
        tokens, src_w2i, tgt_i2w = tokenize(text, 'en'), en_w2i, de_i2w
        model_key = f"{model_type}_en_de"
    else:
        tokens, src_w2i, tgt_i2w = tokenize(text, 'de'), de_w2i, en_i2w
        model_key = f"{model_type}_de_en"

    model = _get_model(model_key)

    if model_type == 'seq2seq':
        result = _seq2seq_translate(model, tokens, src_w2i, tgt_i2w)
    else:
        result = _transformer_translate(model, tokens, src_w2i, tgt_i2w)

    return {"translation": result, "source_tokens": tokens,
            "model": model_type, "direction": direction}