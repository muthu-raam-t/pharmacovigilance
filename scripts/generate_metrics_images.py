"""Builds the comparison tables and charts from results/model_comparison.csv
(and the TF-IDF baseline, if present) into results/figures/."""
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "models"))
import pv_config as cfg  # noqa: E402

CSV_PATH = cfg.RESULTS_DIR / "model_comparison.csv"
OUT_DIR = cfg.RESULTS_DIR / "figures"
ORDER = ["BERT-base", "RoBERTa", "SciBERT", "BioBERT", "PubMedBERT"]


def load_rows():
    with open(CSV_PATH, newline="") as f:
        rows = list(csv.DictReader(f))
    return sorted(rows, key=lambda r: ORDER.index(r["model_name"]) if r["model_name"] in ORDER else 99)


def table_image(rows, columns, title, accent, out_name, highlight_col):
    header = ["Model"] + [c[0] for c in columns]
    data = [[r["model_name"]] + [f"{float(r.get(c[1])):.3f}" if _isnum(r.get(c[1])) else "-"
                                 for c in columns] for r in rows]
    n = len(rows)
    fig, ax = plt.subplots(figsize=(1.6 + 1.5 * len(columns), 0.55 * n + 1.3))
    ax.axis("off")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12, loc="left")
    table = ax.table(cellText=data, colLabels=header, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.8)
    for c in range(len(header)):
        cell = table[0, c]
        cell.set_facecolor(accent)
        cell.set_text_props(color="white", fontweight="bold")
        cell.set_edgecolor("white")
    hi = 1 + [c[1] for c in columns].index(highlight_col)
    vals = [float(r.get(highlight_col)) if _isnum(r.get(highlight_col)) else -1 for r in rows]
    best = vals.index(max(vals))
    for i in range(n):
        bg = "#F5F5F5" if i % 2 == 0 else "white"
        for c in range(len(header)):
            table[i + 1, c].set_facecolor(bg)
            table[i + 1, c].set_edgecolor("#DDDDDD")
        if i == best:
            table[i + 1, hi].set_facecolor("#C6E0B4")
            table[i + 1, hi].set_text_props(fontweight="bold")
    out = OUT_DIR / out_name
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def _isnum(v):
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


def f1_bars(rows):
    names = [r["model_name"] for r in rows]
    series = [("NER F1", "ner_f1", "#2E5C8A"), ("RE micro-F1 (gold entities)", "re_micro_f1", "#8A4C2E"),
              ("End-to-end RE F1", "e2e_f1", "#4C8A2E")]
    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.26
    for k, (label, col, color) in enumerate(series):
        xs = [i + (k - 1) * width for i in range(len(names))]
        vals = [float(r[col]) for r in rows]
        bars = ax.bar(xs, vals, width, label=label, color=color)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1")
    ax.set_title("Backbone comparison")
    ax.legend(loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    out = OUT_DIR / "f1_comparison_bars.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out}")


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_rows()

    table_image(rows, [("Accuracy", "ner_accuracy"), ("Precision", "ner_precision"),
                       ("Recall", "ner_recall"), ("F1", "ner_f1")],
                "NER - Named Entity Recognition (test set)", "#2E5C8A", "comparison_table_ner.png", "ner_f1")

    re_rows = [dict(r) for r in rows]
    baseline_path = cfg.RESULTS_DIR / "baseline_tfidf_svm.json"
    if baseline_path.exists():
        b = json.load(open(baseline_path))["test"]
        re_rows = [{"model_name": "TF-IDF + SVM", **{k: str(v) for k, v in b.items() if not isinstance(v, dict)}}] + re_rows
    table_image(re_rows, [("Precision", "re_micro_precision"), ("Recall", "re_micro_recall"),
                          ("Micro-F1", "re_micro_f1"), ("Macro-F1", "re_macro_f1")],
                "RE - Relation Extraction on gold entities (test set)", "#8A4C2E",
                "comparison_table_re.png", "re_micro_f1")

    table_image(rows, [("Precision", "e2e_precision"), ("Recall", "e2e_recall"), ("F1", "e2e_f1")],
                "End-to-end RE - predicted entities (test set)", "#4C8A2E", "comparison_table_e2e.png", "e2e_f1")

    f1_bars(rows)
