import pandas as pd
import spacy
import pickle
import os
import re
from collections import Counter
from sklearn.model_selection import train_test_split

# ── Config ────────────────────────────────────────────────────────────────────
RAW_PATH       = "data/raw/english-to-german/deu.txt"
PROCESSED_PATH = "data/processed/"
MIN_FREQ       = 2
MAX_LEN        = 50
MIN_LEN        = 3

os.makedirs(PROCESSED_PATH, exist_ok=True)

# ── Load spaCy models ─────────────────────────────────────────────────────────
print("Loading spaCy models...")
nlp_en = spacy.load("en_core_web_sm", disable=["parser", "ner"])
nlp_de = spacy.load("de_core_news_sm", disable=["parser", "ner"])

# ── Tokenizers ────────────────────────────────────────────────────────────────
def tokenize_en(text):
    return [tok.text.lower() for tok in nlp_en.tokenizer(str(text))]

def tokenize_de(text):
    return [tok.text.lower() for tok in nlp_de.tokenizer(str(text))]

# ── Clean text ────────────────────────────────────────────────────────────────
def clean_text(text):
    text = str(text).strip()
    text = re.sub(r"[^\w\s\.\,\!\?\-\'\"]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text

# ── Load dataset ──────────────────────────────────────────────────────────────
print("Loading dataset...")
df = pd.read_csv(
    RAW_PATH,
    sep='\t',
    header=None,
    usecols=[0, 1],
    names=['english', 'german'],
    on_bad_lines='skip',
    encoding='utf-8'
)
print(f"Loaded: {len(df)} rows")

# ── Clean ─────────────────────────────────────────────────────────────────────
print("Cleaning...")
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)
df['english'] = df['english'].apply(clean_text)
df['german']  = df['german'].apply(clean_text)

# Drop rows where cleaning left empty strings
df = df[(df['english'].str.strip() != '') & (df['german'].str.strip() != '')]
df.reset_index(drop=True, inplace=True)
print(f"After cleaning: {len(df)} rows")

# ── Tokenize ──────────────────────────────────────────────────────────────────
print("Tokenizing English... (may take a few minutes)")
df['en_tokens'] = df['english'].apply(tokenize_en)

print("Tokenizing German...")
df['de_tokens'] = df['german'].apply(tokenize_de)

# ── Filter by length ──────────────────────────────────────────────────────────
mask = (
    df['en_tokens'].apply(len).between(MIN_LEN, MAX_LEN) &
    df['de_tokens'].apply(len).between(MIN_LEN, MAX_LEN)
)
df = df[mask].reset_index(drop=True)
print(f"After length filtering: {len(df)} rows")

# ── Build Vocabularies ────────────────────────────────────────────────────────
SPECIAL_TOKENS = ['<pad>', '<sos>', '<eos>', '<unk>']

def build_vocab(token_lists, min_freq=MIN_FREQ):
    counter = Counter()
    for tokens in token_lists:
        counter.update(tokens)
    vocab = {tok: idx for idx, tok in enumerate(SPECIAL_TOKENS)}
    for word, freq in counter.items():
        if freq >= min_freq and word not in vocab:
            vocab[word] = len(vocab)
    return vocab

print("Building vocabularies...")
en_vocab = build_vocab(df['en_tokens'])
de_vocab = build_vocab(df['de_tokens'])

en_idx2word = {v: k for k, v in en_vocab.items()}
de_idx2word = {v: k for k, v in de_vocab.items()}

print(f"English vocab size: {len(en_vocab)}")
print(f"German vocab size:  {len(de_vocab)}")

# ── Encode ────────────────────────────────────────────────────────────────────
def encode(tokens, vocab):
    unk = vocab['<unk>']
    return [vocab['<sos>']] + [vocab.get(t, unk) for t in tokens] + [vocab['<eos>']]

print("Encoding sequences...")
df['en_encoded'] = df['en_tokens'].apply(lambda t: encode(t, en_vocab))
df['de_encoded'] = df['de_tokens'].apply(lambda t: encode(t, de_vocab))

# ── Train / Val / Test split ──────────────────────────────────────────────────
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
val_df,   test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

print(f"\nTrain: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

# ── Save ──────────────────────────────────────────────────────────────────────
print("Saving processed files...")
train_df.to_pickle(os.path.join(PROCESSED_PATH, "train.pkl"))
val_df.to_pickle(os.path.join(PROCESSED_PATH,   "val.pkl"))
test_df.to_pickle(os.path.join(PROCESSED_PATH,  "test.pkl"))

with open(os.path.join(PROCESSED_PATH, "en_vocab.pkl"), "wb") as f:
    pickle.dump({'word2idx': en_vocab, 'idx2word': en_idx2word}, f)

with open(os.path.join(PROCESSED_PATH, "de_vocab.pkl"), "wb") as f:
    pickle.dump({'word2idx': de_vocab, 'idx2word': de_idx2word}, f)

print("\n✅ Preprocessing complete. All files saved to data/processed/")