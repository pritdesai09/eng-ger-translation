# 🌐 English ↔ German Machine Translation System

A full-stack Neural Machine Translation (NMT) system supporting bidirectional
English ↔ German translation using two architectures:

- **Model 1**: Seq2Seq LSTM Encoder-Decoder (No Attention)
- **Model 2**: Transformer Encoder-Decoder (Multi-Head Attention)

## Tech Stack
- **Training**: PyTorch on Google Colab GPU
- **Backend**: FastAPI (Python)
- **Frontend**: HTML, CSS, Vanilla JS
- **Tokenization**: spaCy
- **Evaluation**: BLEU Score (NLTK)

## Project Structure

eng-ger-translation/
├── data/               # Dataset (raw + processed)
├── models/             # Saved model weights + evaluation results
├── notebooks/          # Google Colab training notebooks
├── backend/            # FastAPI server
├── frontend/           # HTML/CSS/JS web interface
└── requirements.txt

## Setup Instructions
See each phase's README section below.

### Run Backend
```powershell
cd backend
uvicorn app:app --reload --port 8000
```

### Open Frontend
Open `frontend/index.html` in your browser.

## BLEU Score Results
| Model | EN→DE | DE→EN |
|---|---|---|
| Seq2Seq (No Attention) | TBD | TBD |
| Transformer (Attention) | TBD | TBD |