import csv
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV_PATH = "/workspace/data/processed/model_comparison.csv"
OUT_DIR = "/workspace/data/processed/metrics_images"

def load_rows(path):
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)

def plot_model_metrics(row, out_dir):
    model_name = row["model_name"]
    metrics = {
        "NER Precision": float(row["ner_precision"]),
        "NER Recall": float(row["ner_recall"]),
        "NER F1": float(row["ner_f1"]),
        "RE Precision": float(row["re_precision"]),
        "RE Recall": float(row["re_recall"]),
        "RE F1": float(row["re_f1"]),
    }

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(metrics.keys(), metrics.values(), color=["#4C72B0"]*3 + ["#DD8452"]*3)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title(f"{model_name} — Evaluation Metrics")
    ax.axhline(y=1.0, color="gray", linewidth=0.5, linestyle="--")

    for bar, val in zip(bars, metrics.values()):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.02, f"{val:.3f}", ha="center", fontsize=9)

    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()

    safe_name = model_name.replace(" ", "_").replace("/", "_")
    out_path = os.path.join(out_dir, f"{safe_name}_metrics.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")

def plot_comparison_bars(rows, out_dir):
    fig, ax = plt.subplots(figsize=(9, 5))
    model_names = [r["model_name"] for r in rows]
    ner_f1 = [float(r["ner_f1"]) for r in rows]
    re_f1 = [float(r["re_f1"]) for r in rows]

    x = range(len(model_names))
    width = 0.35
    ax.bar([i - width/2 for i in x], ner_f1, width, label="NER F1", color="#4C72B0")
    ax.bar([i + width/2 for i in x], re_f1, width, label="RE F1", color="#DD8452")

    ax.set_xticks(list(x))
    ax.set_xticklabels(model_names, rotation=15, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("F1 Score")
    ax.set_title("Model Comparison — NER F1 vs RE F1")
    ax.legend()
    plt.tight_layout()

    out_path = os.path.join(out_dir, "all_models_comparison_bars.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")

def plot_comparison_table(rows, out_dir):
    model_names = [r["model_name"] for r in rows]
    n = len(rows)

    fig, ax = plt.subplots(figsize=(11, 1.1 * n + 3.5))
    ax.axis("off")

    header = ["Model", "Accuracy", "Precision", "Recall", "F1"]
    ner_data = [[r["model_name"], f"{float(r['ner_accuracy']):.3f}", f"{float(r['ner_precision']):.3f}",
                 f"{float(r['ner_recall']):.3f}", f"{float(r['ner_f1']):.3f}"] for r in rows]
    re_data = [[r["model_name"], f"{float(r['re_accuracy']):.3f}", f"{float(r['re_precision']):.3f}",
                f"{float(r['re_recall']):.3f}", f"{float(r['re_f1']):.3f}"] for r in rows]

    section_gap = [""] * len(header)
    section_label_ner = ["NER — Named Entity Recognition", "", "", "", ""]
    section_label_re = ["RE — Relation Extraction", "", "", "", ""]

    full_data = [section_label_ner] + [header] + ner_data + [section_gap] + [section_label_re] + [header] + re_data

    table = ax.table(cellText=full_data, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.7)

    ner_header_row = 1
    re_section_row = 2 + n + 1
    re_header_row = re_section_row + 1

    for col_idx in range(len(header)):
        table[0, col_idx].set_facecolor("#2E5C8A")
        table[0, col_idx].set_text_props(color="white", fontweight="bold")
        table[ner_header_row, col_idx].set_facecolor("#4C72B0")
        table[ner_header_row, col_idx].set_text_props(color="white", fontweight="bold")
        table[re_section_row, col_idx].set_facecolor("#8A4C2E")
        table[re_section_row, col_idx].set_text_props(color="white", fontweight="bold")
        table[re_header_row, col_idx].set_facecolor("#DD8452")
        table[re_header_row, col_idx].set_text_props(color="white", fontweight="bold")

    for i in range(0, len(header)):
        table[0, i].set_facecolor("#2E5C8A")

    ner_f1_vals = [float(r["ner_f1"]) for r in rows]
    re_f1_vals = [float(r["re_f1"]) for r in rows]
    best_ner_idx = ner_f1_vals.index(max(ner_f1_vals))
    best_re_idx = re_f1_vals.index(max(re_f1_vals))

    ner_data_start = ner_header_row + 1
    re_data_start = re_header_row + 1

    for row_offset in range(n):
        row_num = ner_data_start + row_offset
        bg = "#EAF2FB" if row_offset % 2 == 0 else "#FFFFFF"
        for col_idx in range(len(header)):
            table[row_num, col_idx].set_facecolor(bg)
        if row_offset == best_ner_idx:
            table[row_num, 4].set_facecolor("#C6E0B4")

    for row_offset in range(n):
        row_num = re_data_start + row_offset
        bg = "#FBEFE8" if row_offset % 2 == 0 else "#FFFFFF"
        for col_idx in range(len(header)):
            table[row_num, col_idx].set_facecolor(bg)
        if row_offset == best_re_idx:
            table[row_num, 4].set_facecolor("#C6E0B4")

    for col_idx in range(len(header)):
        table[2 + n, col_idx].set_facecolor("white")
        table[0, col_idx].set_height(table[0, col_idx].get_height() * 0.8)
        table[re_section_row, col_idx].set_height(table[re_section_row, col_idx].get_height() * 0.8)

    ax.set_title("5-Model Comparison — Evaluation Metrics", fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()

    out_path = os.path.join(out_dir, "all_models_comparison_table.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")
