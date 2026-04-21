import pandas as pd
import matplotlib.pyplot as plt
import os

# ── Load ──────────────────────────────────────────────────────────────────────
RAW_PATH = "data/raw/english-to-german/deu.txt"

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

print("\nShape:", df.shape)
print("\nFirst 5 rows:\n", df.head())
print("\nColumn names:", df.columns.tolist())
print("\nNull values:\n", df.isnull().sum())
print("\nDuplicate rows:", df.duplicated().sum())

# ── Basic Stats ───────────────────────────────────────────────────────────────
df['en_len'] = df['english'].apply(lambda x: len(str(x).split()))
df['de_len'] = df['german'].apply(lambda x: len(str(x).split()))

print("\nEnglish sentence length stats:")
print(df['en_len'].describe())
print("\nGerman sentence length stats:")
print(df['de_len'].describe())

print(f"\nSentences with EN length > 50: {(df['en_len'] > 50).sum()}")
print(f"Sentences with DE length > 50: {(df['de_len'] > 50).sum()}")
print(f"Sentences with EN length < 3:  {(df['en_len'] < 3).sum()}")
print(f"Sentences with DE length < 3:  {(df['de_len'] < 3).sum()}")

# ── Sample sentences ──────────────────────────────────────────────────────────
print("\n── Sample sentence pairs ──")
for _, row in df.sample(5, random_state=42).iterrows():
    print(f"  EN: {row['english']}")
    print(f"  DE: {row['german']}")
    print()

# ── Plot length distributions ─────────────────────────────────────────────────
os.makedirs("models/evaluation", exist_ok=True)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Sentence Length Distribution — EN ↔ DE Dataset', fontsize=14)

axes[0].hist(df['en_len'].clip(upper=60), bins=50, color='steelblue', edgecolor='white')
axes[0].set_title('English Sentence Length')
axes[0].set_xlabel('Number of Tokens')
axes[0].set_ylabel('Frequency')
axes[0].axvline(50, color='red', linestyle='--', linewidth=1, label='MAX_LEN=50')
axes[0].legend()

axes[1].hist(df['de_len'].clip(upper=60), bins=50, color='coral', edgecolor='white')
axes[1].set_title('German Sentence Length')
axes[1].set_xlabel('Number of Tokens')
axes[1].axvline(50, color='red', linestyle='--', linewidth=1, label='MAX_LEN=50')
axes[1].legend()

plt.tight_layout()
plt.savefig("models/evaluation/length_distribution.png", dpi=150)
plt.show()
print("\n✅ Plot saved to models/evaluation/length_distribution.png")