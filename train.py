import argparse
import ast
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
)


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):
    if text is None:
        return ""

    text = str(text)

    # Remove excessive whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# EXTRACT FACTS AND ARGUMENTS FROM YOUR DATASET
# ============================================================

def extract_facts(value):

    if pd.isna(value):
        return ""

    # The CSV stores the dictionary as a STRING.
    value = str(value)

    # Try to extract only the facts-and-arguments section.
    match = re.search(
        r"'facts-and-arguments':\s*array\((.*?)\),\s*'judge-opinion':",
        value,
        flags=re.DOTALL
    )

    if match:
        facts_section = match.group(1)

        # Extract quoted Hindi strings.
        texts = re.findall(
            r"'((?:\\.|[^'])*)'",
            facts_section,
            flags=re.DOTALL
        )

        if texts:
            return clean_text(" ".join(texts))

    # Backup method:
    # locate facts-and-arguments and stop before judge-opinion
    start = value.find("'facts-and-arguments':")

    if start != -1:
        end = value.find("'judge-opinion':")

        if end == -1:
            end = len(value)

        section = value[start:end]

        # Remove Python/NumPy representation
        section = re.sub(r"array\(", " ", section)
        section = section.replace("dtype=object", " ")

        # Extract quoted strings
        texts = re.findall(
            r"'((?:\\.|[^'])*)'",
            section,
            flags=re.DOTALL
        )

        if texts:
            return clean_text(" ".join(texts))

    return ""


# ============================================================
# LABEL
# ============================================================

def normalize_label(value):

    try:
        value = int(value)

        if value in [0, 1]:
            return value

    except Exception:
        pass

    text = str(value).strip().lower()

    if text in ["granted", "grant", "yes", "true"]:
        return 1

    if text in ["denied", "deny", "dismissed", "no", "false"]:
        return 0

    return np.nan


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset(path):

    print(f"\nLoading: {path}")

    # CSV support
    if str(path).lower().endswith(".csv"):
        df = pd.read_csv(path)

    # Parquet support
    else:
        df = pd.read_parquet(path)

    print("Original shape:", df.shape)
    print("Columns:", df.columns.tolist())

    if "text" not in df.columns:
        raise ValueError(
            "ERROR: 'text' column not found."
        )

    if "label" not in df.columns:
        raise ValueError(
            "ERROR: 'label' column not found."
        )

    print("Extracting facts-and-arguments...")

    df["model_text"] = df["text"].apply(extract_facts)

    df["y"] = df["label"].apply(normalize_label)

    # Remove empty text
    df = df[df["model_text"].str.len() > 20]

    # Keep only valid labels
    df = df[df["y"].isin([0, 1])]

    df = df.reset_index(drop=True)

    print("Usable rows:", len(df))

    print(
        "Labels:",
        df["y"].value_counts().to_dict()
    )

    return df


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(model, word_vectorizer, char_vectorizer, df, name):

    print(f"\n========== {name.upper()} ==========")

    word_features = word_vectorizer.transform(
        df["model_text"]
    )

    char_features = char_vectorizer.transform(
        df["model_text"]
    )

    X = hstack(
        [word_features, char_features]
    ).tocsr()

    predictions = model.predict(X)

    probabilities = model.predict_proba(X)[:, 1]

    accuracy = accuracy_score(
        df["y"],
        predictions
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            df["y"],
            predictions,
            average="binary",
            zero_division=0
        )
    )

    macro_f1 = precision_recall_fscore_support(
        df["y"],
        predictions,
        average="macro",
        zero_division=0
    )[2]

    cm = confusion_matrix(
        df["y"],
        predictions
    )

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"Macro F1 : {macro_f1:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "macro_f1": float(macro_f1),
        "confusion_matrix": cm.tolist()
    }


# ============================================================
# MAIN TRAINING
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train",
        required=True
    )

    parser.add_argument(
        "--valid",
        required=True
    )

    parser.add_argument(
        "--test",
        required=True
    )

    parser.add_argument(
        "--out",
        default="artifacts"
    )

    args = parser.parse_args()

    output_dir = Path(args.out)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    train = load_dataset(args.train)

    valid = load_dataset(args.valid)

    test = load_dataset(args.test)

    print("\n================================")
    print("FINAL DATASET SIZES")
    print("================================")

    print("Train:", len(train))
    print("Validation:", len(valid))
    print("Test:", len(test))

    # --------------------------------------------------------
    # TF-IDF WORD FEATURES
    # --------------------------------------------------------

    print("\nCreating word TF-IDF...")

    word_vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.98,
        max_features=180000,
        sublinear_tf=True
    )

    # --------------------------------------------------------
    # TF-IDF CHARACTER FEATURES
    # --------------------------------------------------------

    print("Creating character TF-IDF...")

    char_vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        min_df=2,
        max_features=120000,
        sublinear_tf=True
    )

    # --------------------------------------------------------
    # TRAIN FEATURES
    # --------------------------------------------------------

    print("\nTransforming training data...")

    train_word = word_vectorizer.fit_transform(
        train["model_text"]
    )

    train_char = char_vectorizer.fit_transform(
        train["model_text"]
    )

    X_train = hstack(
        [train_word, train_char]
    ).tocsr()

    print(
        "Training feature matrix:",
        X_train.shape
    )

    # --------------------------------------------------------
    # TRAIN CLASSIFIER
    # --------------------------------------------------------

    print("\nTraining Logistic Regression...")

    model = LogisticRegression(
        max_iter=300,
        C=2.0,
        class_weight="balanced",
        solver="liblinear"
    )

    model.fit(
        X_train,
        train["y"]
    )

    print("Training completed.")

    # --------------------------------------------------------
    # EVALUATE
    # --------------------------------------------------------

    train_metrics = evaluate_model(
        model,
        word_vectorizer,
        char_vectorizer,
        train,
        "Train"
    )

    validation_metrics = evaluate_model(
        model,
        word_vectorizer,
        char_vectorizer,
        valid,
        "Validation"
    )

    test_metrics = evaluate_model(
        model,
        word_vectorizer,
        char_vectorizer,
        test,
        "Test"
    )

    # --------------------------------------------------------
    # SAVE MODEL
    # --------------------------------------------------------

    print("\nSaving model...")

    joblib.dump(
        word_vectorizer,
        output_dir / "word_tfidf.joblib"
    )

    joblib.dump(
        char_vectorizer,
        output_dir / "char_tfidf.joblib"
    )

    joblib.dump(
        model,
        output_dir / "logistic_model.joblib"
    )

    # --------------------------------------------------------
    # SAVE RETRIEVAL DATA
    # --------------------------------------------------------

    print("Preparing historical case retrieval...")

    retrieval_columns = [
        "model_text",
        "y"
    ]

    if "id" in train.columns:
        retrieval_columns.insert(
            0,
            "id"
        )

    if "district" in train.columns:
        retrieval_columns.insert(
            1,
            "district"
        )

    retrieval = train[
        retrieval_columns
    ].copy()

    # Keep maximum 20,000 cases for fast demo retrieval
    if len(retrieval) > 20000:

        retrieval = retrieval.sample(
            20000,
            random_state=42
        )

    retrieval.to_parquet(
        output_dir / "retrieval_corpus.parquet",
        index=False
    )

    # --------------------------------------------------------
    # SAVE METRICS
    # --------------------------------------------------------

    metrics = {
        "train": train_metrics,
        "validation": validation_metrics,
        "test": test_metrics
    }

    with open(
        output_dir / "metrics.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metrics,
            f,
            indent=4
        )

    print("\n================================")
    print("MODEL TRAINING FINISHED")
    print("================================")

    print(
        "Artifacts saved in:",
        output_dir.resolve()
    )


if __name__ == "__main__":
    main()