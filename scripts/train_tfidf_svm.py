import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, classification_report

def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def build_examples(records):
    texts, labels = [], []
    for r in records:
        if r["relations"]:
            for rel in r["relations"]:
                texts.append(r["text"])
                labels.append(rel["type"])
        else:
            if r["entities"]:
                texts.append(r["text"])
                labels.append("NONE")
    return texts, labels

if __name__ == "__main__":
    BASE = "/workspace/data/processed"

    train_records = load_jsonl(f"{BASE}/ner_re_train.jsonl")
    val_records = load_jsonl(f"{BASE}/ner_re_val.jsonl")
    test_records = load_jsonl(f"{BASE}/ner_re_test.jsonl")

    X_train_text, y_train = build_examples(train_records)
    X_val_text, y_val = build_examples(val_records)
    X_test_text, y_test = build_examples(test_records)

    print(f"Train examples: {len(X_train_text)}")
    print(f"Val examples: {len(X_val_text)}")
    print(f"Test examples: {len(X_test_text)}")

    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(X_train_text)
    X_val = vectorizer.transform(X_val_text)
    X_test = vectorizer.transform(X_test_text)

    clf = LinearSVC(class_weight="balanced", max_iter=5000)
    clf.fit(X_train, y_train)

    for split_name, X, y_true in [("Validation", X_val, y_val), ("Test", X_test, y_test)]:
        y_pred = clf.predict(X)
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="weighted")
        print(f"\n=== {split_name} ===")
        print(f"Accuracy: {acc:.3f}")
        print(f"Weighted F1: {f1:.3f}")
        print(classification_report(y_true, y_pred, zero_division=0))
