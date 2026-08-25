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
    columns = ["Model", "NER Prec", "NER Rec", "NER F1", "RE Prec", "RE Rec", "RE F1"]
    table_data = []
    for r in rows:
        table_data.append([
            r["model_name"],
            f"{float(r['ner_precision']):.3f}",
            f"{float(r['ner_recall']):.3f}",
            f"{float(r['ner_f1']):.3f}",
            f"{float(r['re_precision']):.3f}",
            f"{float(r['re_recall']):.3f}",
            f"{float(r['re_f1']):.3f}",
        ])

    fig, ax = plt.subplots(figsize=(10, 0.6 * len(rows) + 1.5))
    ax.axis("off")

    table = ax.table(
        cellText=table_data,
        colLabels=columns,
        cellLoc="center",
        loc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)

    for col_idx in range(len(columns)):
        table[0, col_idx].set_facecolor("#4C72B0")
        table[0, col_idx].set_text_props(color="white", fontweight="bold")

    ner_f1_vals = [float(r["ner_f1"]) for r in rows]
    re_f1_vals = [float(r["re_f1"]) for r in rows]
    best_ner_idx = ner_f1_vals.index(max(ner_f1_vals))
    best_re_idx = re_f1_vals.index(max(re_f1_vals))

    table[best_ner_idx + 1, 3].set_facecolor("#C6E0B4")
    table[best_re_idx + 1, 6].set_facecolor("#C6E0B4")

    ax.set_title("5-Model Comparison — Evaluation Metrics Summary", fontsize=13, pad=20)
    plt.tight_layout()

    out_path = os.path.join(out_dir, "all_models_comparison_table.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")

if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = load_rows(CSV_PATH)

    for row in rows:
        plot_model_metrics(row, OUT_DIR)

    if len(rows) > 1:
        plot_comparison_bars(rows, OUT_DIR)
        plot_comparison_table(rows, OUT_DIR)
