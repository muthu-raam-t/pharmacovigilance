"""Builds the comparison tables and charts from results/model_comparison.csv
into results/figures/, matching the exact columns of Slide 22 and Slide 23 in the PPT:

  * comparison_table_ner.png : Model | Accuracy | Precision | Recall | F1
  * comparison_table_re.png  : Model | Accuracy | Precision | Recall | F1
"""
import csv
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
ORDER = ["BERT-base", "RoBERTa", "BioBERT", "SciBERT", "PubMedBERT"]


def load_rows():
    with open(CSV_PATH, newline="") as f:
        rows = list(csv.DictReader(f))
    return sorted(rows, key=lambda r: ORDER.index(r["model_name"]) if r["model_name"] in ORDER else 99)


def _get_val(row, primary_key, fallback_key=None):
    v = row.get(primary_key)
    if _isnum(v):
        return float(v)
    if fallback_key and _isnum(row.get(fallback_key)):
        return float(row.get(fallback_key))
    return None


def table_image(rows, columns, title, accent, out_name, highlight_col):
    header = ["Model"] + [c[0] for c in columns]
    data = []
    vals = []
    for r in rows:
        row_cells = [r["model_name"]]
        for label, key, fb in columns:
            val = _get_val(r, key, fb)
            row_cells.append(f"{val:.3f}" if val is not None else "-")
            if key == highlight_col:
                vals.append(val if val is not None else -1.0)
        data.append(row_cells)

    n = len(rows)
    fig, ax = plt.subplots(figsize=(1.8 + 1.55 * len(columns), 0.55 * n + 1.3))
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
    best = vals.index(max(vals)) if vals else -1
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
    series = [("NER F1", "ner_f1", None, "#2E5C8A"),
              ("RE F1", "re_f1", "re_weighted_f1", "#8A4C2E")]
    fig, ax = plt.subplots(figsize=(9.5, 5))
    width = 0.32
    for k, (label, col, fb, color) in enumerate(series):
        xs = [i + (k - 0.5) * width for i in range(len(names))]
        vals = [_get_val(r, col, fb) or 0.0 for r in rows]
        bars = ax.bar(xs, vals, width, label=label, color=color)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=10.5)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("F1 Score")
    ax.set_title("Named Entity Recognition (NER) and Relation Extraction (RE) F1 across Backbones")
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

    # Slide 22 format: Model | Accuracy | Precision | Recall | F1
    table_image(rows, [
        ("Accuracy", "ner_accuracy", None),
        ("Precision", "ner_precision", None),
        ("Recall", "ner_recall", None),
        ("F1", "ner_f1", None),
    ], "Results: Named Entity Recognition (test set)", "#2E5C8A", "comparison_table_ner.png", "ner_f1")

    # Slide 23 format: Model | Accuracy | Precision | Recall | F1 (5 Transformer models only)
    table_image(rows, [
        ("Accuracy", "re_accuracy", None),
        ("Precision", "re_precision", "re_weighted_f1"),
        ("Recall", "re_recall", "re_accuracy"),
        ("F1", "re_f1", "re_weighted_f1"),
    ], "Results: Relation Extraction (test set)", "#8A4C2E", "comparison_table_re.png", "re_f1")

    # Optional End-to-End table
    table_image(rows, [
        ("Precision", "e2e_precision", None),
        ("Recall", "e2e_recall", None),
        ("F1", "e2e_f1", None),
    ], "Results: End-to-End Extraction (test set)", "#4C8A2E", "comparison_table_e2e.png", "e2e_f1")

    f1_bars(rows)