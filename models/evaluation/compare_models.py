"""
Run this AFTER downloading model weights from Google Drive.
This script generates the final comparison report.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Paste your actual BLEU scores here after training ────────────────────────
results = {
    "Seq2Seq (No Attention)": {"EN→DE": 8.5,  "DE→EN": 9.2},
    "Transformer (Attention)": {"EN→DE": 18.3, "DE→EN": 19.1},
}

param_counts = {
    "Seq2Seq (No Attention)": 14_000_000,
    "Transformer (Attention)": 18_000_000,
}

inference_speed = {
    "Seq2Seq (No Attention)": 12,   # ms per sentence
    "Transformer (Attention)": 28,
}

# ── Plot 1: BLEU Score Comparison ────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

models = list(results.keys())
en_de_scores = [results[m]["EN→DE"] for m in models]
de_en_scores = [results[m]["DE→EN"] for m in models]
x = np.arange(len(models))
w = 0.35

axes[0].bar(x - w/2, en_de_scores, w, label='EN→DE', color='steelblue')
axes[0].bar(x + w/2, de_en_scores, w, label='DE→EN', color='coral')
axes[0].set_title('BLEU Score Comparison', fontsize=13)
axes[0].set_xticks(x)
axes[0].set_xticklabels(['Seq2Seq', 'Transformer'], fontsize=10)
axes[0].set_ylabel('BLEU Score')
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)

# ── Plot 2: Parameter Count ───────────────────────────────────────────────────
params = [param_counts[m] / 1e6 for m in models]
axes[1].bar(models, params, color=['#4ECDC4', '#FF6B6B'])
axes[1].set_title('Model Size (Million Parameters)', fontsize=13)
axes[1].set_ylabel('Parameters (M)')
axes[1].set_xticklabels(['Seq2Seq', 'Transformer'], fontsize=10)
axes[1].grid(axis='y', alpha=0.3)

# ── Plot 3: Inference Speed ───────────────────────────────────────────────────
speeds = [inference_speed[m] for m in models]
axes[2].bar(models, speeds, color=['#95E1D3', '#F38181'])
axes[2].set_title('Inference Speed (ms / sentence)', fontsize=13)
axes[2].set_ylabel('Milliseconds')
axes[2].set_xticklabels(['Seq2Seq', 'Transformer'], fontsize=10)
axes[2].grid(axis='y', alpha=0.3)

plt.suptitle('Model Comparison: Seq2Seq vs Transformer', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig("models/evaluation/model_comparison.png", dpi=150)
plt.show()
print("✅ Comparison chart saved.")

# ── Print Summary Table ───────────────────────────────────────────────────────
print("\n" + "="*60)
print(f"{'Metric':<30} {'Seq2Seq':>12} {'Transformer':>12}")
print("="*60)
print(f"{'BLEU EN→DE':<30} {en_de_scores[0]:>12.2f} {en_de_scores[1]:>12.2f}")
print(f"{'BLEU DE→EN':<30} {de_en_scores[0]:>12.2f} {de_en_scores[1]:>12.2f}")
print(f"{'Parameters (M)':<30} {params[0]:>12.1f} {params[1]:>12.1f}")
print(f"{'Inference (ms)':<30} {speeds[0]:>12} {speeds[1]:>12}")
print("="*60)