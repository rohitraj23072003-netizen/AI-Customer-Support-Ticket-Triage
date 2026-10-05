import os
import re
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

DATA_PATH = "data/historical_tickets.csv"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def build_model():
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True, max_features=10000)),
        ("classifier", LogisticRegression(max_iter=2000))
    ])

def main():
    # STEP 1 + STEP 2
    df = pd.read_csv(DATA_PATH)
    required = {"ticket_text", "category", "priority", "resolution"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    df["clean_text"] = df["ticket_text"].map(clean_text)

    # STEP 3
    X = df["clean_text"]
    y_category = df["category"]
    y_urgency = df["priority"]
    X_train, X_test, ycat_train, ycat_test, yur_train, yur_test = train_test_split(
        X, y_category, y_urgency, test_size=0.25, random_state=42, stratify=y_category
    )

    # STEP 4
    category_model = build_model()
    urgency_model = build_model()
    category_model.fit(X_train, ycat_train)
    urgency_model.fit(X_train, yur_train)

    # STEP 5
    cat_pred = category_model.predict(X_test)
    urg_pred = urgency_model.predict(X_test)
    print("\n=== CATEGORY MODEL ===")
    print("F1 (macro):", round(f1_score(ycat_test, cat_pred, average="macro"), 4))
    print(classification_report(ycat_test, cat_pred, zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(ycat_test, cat_pred, labels=category_model.classes_))
    print("\n=== URGENCY MODEL ===")
    print("F1 (macro):", round(f1_score(yur_test, urg_pred, average="macro"), 4))
    print(classification_report(yur_test, urg_pred, zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(yur_test, urg_pred, labels=urgency_model.classes_))

    joblib.dump(category_model, f"{MODEL_DIR}/category_model.joblib")
    joblib.dump(urgency_model, f"{MODEL_DIR}/urgency_model.joblib")
    print("\nModels saved in models/")

if __name__ == "__main__":
    main()
