"""Classical baseline: TF-IDF + Linear SVM relation classifier.

It is scored on exactly the same candidate entity pairs as the transformer RE head
(gold entities, same type filter, same labels), so the comparison is like-for-like.
Features per pair: both entity texts, both entity types and the words between them.
"""
import json
import sys
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.svm import LinearSVC

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "models"))
import pv_config as cfg  # noqa: E402
from pair_utils import candidate_pairs, NONE_LABEL  # noqa: E402

MAX_BETWEEN = 40


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def pair_features(record, a, b):
    toks = record["tokens"]
    between = toks[a["token_end"]:b["token_start"]][:MAX_BETWEEN]
    return " ".join([
        f"TYPE1_{a['type']}", f"TYPE2_{b['type']}", f"PAIR_{a['type']}_{b['type']}",
        "E1", *toks[a["token_start"]:a["token_end"]],
        "E2", *toks[b["token_start"]:b["token_end"]],
        "BETWEEN", *between,
    ]).lower()


def build(records, type_pairs):
    X, y = [], []
    for r in records:
        for a, b, label in candidate_pairs(r, type_pairs):
            X.append(pair_features(r, a, b))
            y.append(label)
    return X, y


def score(y_true, y_pred, labels):
    positive = [l for l in labels if l != NONE_LABEL]
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=positive, average="micro", zero_division=0)
    _, _, macro, _ = precision_recall_fscore_support(y_true, y_pred, labels=positive, average="macro", zero_division=0)
    _, _, weighted, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    return {"re_micro_precision": p, "re_micro_recall": r, "re_micro_f1": f, "re_macro_f1": macro,
            "re_weighted_f1": weighted, "re_accuracy": accuracy_score(y_true, y_pred), "re_num_pairs": len(y_true),
            "re_per_class": {k: v for k, v in report.items() if isinstance(v, dict)}}


if __name__ == "__main__":
    base = cfg.PROCESSED_DIR
    label_maps = json.load(open(base / "label_maps.json"))
    type_pairs = label_maps["re_type_pairs"]
    labels = list(label_maps["re_label2id"])

    splits = {s: build(load_jsonl(base / f"ner_re_{s}.jsonl"), type_pairs) for s in ["train", "val", "test"]}
    for s, (X, y) in splits.items():
        print(f"{s}: {len(X)} candidate pairs")

    vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), sublinear_tf=True, token_pattern=r"\S+")
    X_train = vectorizer.fit_transform(splits["train"][0])
    clf = LinearSVC(class_weight="balanced", max_iter=10000)
    clf.fit(X_train, splits["train"][1])

    results = {}
    for s in ["val", "test"]:
        X, y = splits[s]
        pred = clf.predict(vectorizer.transform(X))
        m = score(y, list(pred), labels)
        results[s] = m
        print(f"\n=== {s} ===")
        print(f"micro-P {m['re_micro_precision']:.4f}  micro-R {m['re_micro_recall']:.4f}  "
              f"micro-F1 {m['re_micro_f1']:.4f}  macro-F1 {m['re_macro_f1']:.4f}  Acc {m['re_accuracy']:.4f}")
        print(classification_report(y, pred, labels=labels, zero_division=0))

    cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(cfg.RESULTS_DIR / "baseline_tfidf_svm.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"Saved {cfg.RESULTS_DIR / 'baseline_tfidf_svm.json'}")
