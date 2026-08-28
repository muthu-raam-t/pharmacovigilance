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
    header = ["Model", "Accuracy", "Precision", "Recall", "F1"]

    def section_image(rows, metric_prefix, section_title, accent_color):
        data = [[r["model_name"],
                 f"{float(r[f'{metric_prefix}_accuracy']):.3f}",
                 f"{float(r[f'{metric_prefix}_precision']):.3f}",
                 f"{float(r[f'{metric_prefix}_recall']):.3f}",
                 f"{float(r[f'{metric_prefix}_f1']):.3f}"] for r in rows]

        n = len(rows)
        fig, ax = plt.subplots(figsize=(8, 0.7 * n + 1.8))
        ax.axis("off")
        ax.set_title(section_title, fontsize=15, fontweight="bold", pad=15, loc="left")

        table = ax.table(cellText=data, colLabels=header, cellLoc="center", loc="center",
                          colWidths=[0.30, 0.175, 0.175, 0.175, 0.175])
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2.0)

        for col_idx in range(len(header)):
            cell = table[0, col_idx]
            cell.set_facecolor(accent_color)
            cell.set_text_props(color="white", fontweight="bold")
            cell.set_edgecolor("white")

        f1_vals = [float(d[4]) for d in data]
        best_idx = f1_vals.index(max(f1_vals))

        for row_idx in range(n):
            bg = "#F5F5F5" if row_idx % 2 == 0 else "white"
            for col_idx in range(len(header)):
                cell = table[row_idx + 1, col_idx]
                cell.set_facecolor(bg)
                cell.set_edgecolor("#DDDDDD")
            table[row_idx + 1, 4].set_facecolor("#C6E0B4" if row_idx == best_idx else bg)

        plt.tight_layout()
        return fig

    fig_ner = section_image(rows, "ner", "NER — Named Entity Recognition", "#2E5C8A")
    ner_path = os.path.join(out_dir, "comparison_table_ner.png")
    fig_ner.savefig(ner_path, dpi=150, bbox_inches="tight")
    plt.close(fig_ner)
    print(f"Saved {ner_path}")

    fig_re = section_image(rows, "re", "RE — Relation Extraction", "#8A4C2E")
    re_path = os.path.join(out_dir, "comparison_table_re.png")
    fig_re.savefig(re_path, dpi=150, bbox_inches="tight")
    plt.close(fig_re)
    print(f"Saved {re_path}")


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = load_rows(CSV_PATH)

    for row in rows:
        plot_model_metrics(row, OUT_DIR)

    if len(rows) > 1:
        plot_comparison_bars(rows, OUT_DIR)
        plot_comparison_table(rows, OUT_DIR)
