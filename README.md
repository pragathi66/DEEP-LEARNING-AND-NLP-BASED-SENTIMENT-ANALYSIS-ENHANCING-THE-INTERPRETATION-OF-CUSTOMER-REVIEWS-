# Sentiment Analysis on Fashion Reviews — LSTM

This repository contains a compact sentiment-classification pipeline using a Keras LSTM model.
It is suitable as a portfolio/demo project for analyzing product review sentiment.

---

## 📋 Dataset format

Your CSV should contain at least two columns:

- `review_text` — the review text (string)
- `label` — sentiment label: `positive` or `negative` (case-insensitive)

**Example:**
```csv
review_text,label
"I loved the fabric and fitting",positive
"Poor quality, faded after wash",negative
