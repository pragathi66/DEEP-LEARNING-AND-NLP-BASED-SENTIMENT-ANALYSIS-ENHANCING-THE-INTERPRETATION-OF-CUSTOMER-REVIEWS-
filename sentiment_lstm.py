#!/usr/bin/env python3
"""
Sentiment classification for fashion product reviews using an LSTM.
Expects a CSV with at least columns: 'review_text' and 'label' (values: 'positive'/'negative').

Usage:
    python sentiment_lstm.py               # will look for 'fashion_reviews.csv' in cwd
    python sentiment_lstm.py --input path/to/your.csv
"""

import os
import re
import argparse
import numpy as np
import pandas as pd
import nltk

from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# ---------------------------
# Configuration / hyperparams
# ---------------------------
SEED = 42
MAX_WORDS = 5000        # vocabulary size
MAX_LEN = 100           # max tokens per sample (padding/truncating)
EMBEDDING_DIM = 128
LSTM_UNITS = 64
BATCH_SIZE = 64
EPOCHS = 5
TEST_SIZE = 0.2
VALIDATION_SPLIT = 0.2

np.random.seed(SEED)


# ---------------------------
# Helpers / preprocessing
# ---------------------------
def ensure_nltk_stopwords():
    try:
        stopwords.words('english')
    except LookupError:
        print("Downloading NLTK stopwords...")
        nltk.download('stopwords')


def preprocess_text(text, remove_stopwords=True):
    """
    Lowercase, remove non-alphanumeric, collapse whitespace, remove stopwords.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    # Keep only letters and numbers and whitespace
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    if remove_stopwords:
        tokens = text.split()
        tokens = [t for t in tokens if t not in STOP_WORDS]
        return " ".join(tokens)
    return text


# ---------------------------
# Main pipeline
# ---------------------------
def main(csv_path="fashion_reviews.csv", plot_distribution=True):
    ensure_nltk_stopwords()
    global STOP_WORDS
    STOP_WORDS = set(stopwords.words('english'))

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at path: {csv_path}")

    # Load
    df = pd.read_csv(csv_path)
    print(f"Loaded dataset with {len(df):,} rows and columns: {list(df.columns)}")

    # Basic validation for required columns
    required_cols = ['review_text', 'label']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset must contain columns: {required_cols}. Missing: {missing}")

    # Drop NA rows in relevant columns
    df = df.dropna(subset=['review_text', 'label']).reset_index(drop=True)
    print(f"After dropping NA: {len(df):,} rows remain")

    # Preprocess text
    df['cleaned'] = df['review_text'].apply(lambda t: preprocess_text(t))

    # Normalize label values and filter
    df['label'] = df['label'].astype(str).str.lower().str.strip()
    df = df[df['label'].isin(['positive', 'negative'])].reset_index(drop=True)
    if df.empty:
        raise ValueError("No rows with labels 'positive' or 'negative' after filtering.")
    print(f"After label filtering: {len(df):,} rows with distribution:")
    print(df['label'].value_counts())

    # Tokenize and pad
    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(df['cleaned'].values)
    sequences = tokenizer.texts_to_sequences(df['cleaned'].values)
    X = pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')

    # Encode labels
    y = df['label'].map({'positive': 1, 'negative': 0}).astype(int).values

    # Train / test split
    stratify_arg = y if len(np.unique(y)) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=stratify_arg
    )
    print(f"Train samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

    # Build model
    model = Sequential([
        Embedding(input_dim=MAX_WORDS, output_dim=EMBEDDING_DIM, input_length=MAX_LEN),
        LSTM(LSTM_UNITS, dropout=0.2, recurrent_dropout=0.2),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    # Train
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        verbose=2
    )

    # Evaluate
    y_prob = model.predict(X_test, batch_size=BATCH_SIZE)
    y_pred = (y_prob > 0.5).astype(int).reshape(-1)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred, digits=4))

    # Optional: plot sentiment distribution and training curves
    if plot_distribution:
        try:
            import matplotlib.pyplot as plt
            # label distribution
            plt.figure(figsize=(6, 4))
            df['label'].value_counts().plot(kind='bar', title='Sentiment Distribution')
            plt.xlabel('Sentiment'); plt.ylabel('Count'); plt.tight_layout()
            plt.show()
            # training loss/acc
            plt.figure(figsize=(10, 4))
            plt.subplot(1, 2, 1)
            plt.plot(history.history['loss'], label='train_loss')
            plt.plot(history.history['val_loss'], label='val_loss')
            plt.legend(); plt.title('Loss')
            plt.subplot(1, 2, 2)
            plt.plot(history.history['accuracy'], label='train_acc')
            plt.plot(history.history['val_accuracy'], label='val_acc')
            plt.legend(); plt.title('Accuracy')
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print("Plotting skipped (matplotlib issue):", e)

    # Save tokenizer and model (optional)
    try:
        tokenizer_json = tokenizer.to_json()
        with open("tokenizer.json", "w", encoding="utf-8") as f:
            f.write(tokenizer_json)
        model.save("sentiment_lstm_model.h5")
        print("Saved tokenizer.json and sentiment_lstm_model.h5")
    except Exception as e:
        print("Warning: failed to save model/tokenizer:", e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LSTM sentiment classifier on fashion reviews.")
    parser.add_argument("--input", "-i", default="fashion_reviews.csv", help="Path to input CSV file (must include review_text,label)")
    parser.add_argument("--no-plot", action="store_true", help="Disable plotting of results")
    args = parser.parse_args()
    main(csv_path=args.input, plot_distribution=not args.no_plot)
