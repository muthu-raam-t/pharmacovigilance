"""Collects dataset, knowledge-base and knowledge-graph statistics into results/kb_stats.json
and renders the dataset and knowledge-base tables into results/figures/."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import psycopg2  # noqa: E402
from neo4j import GraphDatabase  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "models"))
import pv_config as cfg  # noqa: E402

SOURCE_TABLES = [
    ("sider_drug_names", "SIDER"), ("sider_drug_atc", "SIDER"), ("sider_side_effects", "SIDER"),
    ("sider_side_effect_freq", "SIDER"), ("onsides_product_label", "OnSIDES"),
    ("onsides_product_adverse_effect", "OnSIDES"), ("onsides_product_to_rxnorm", "OnSIDES"),
    ("onsides_vocab_meddra", "OnSIDES"), ("onsides_vocab_rxnorm_ingredient", "OnSIDES"),
    ("onsides_vocab_rxnorm_ingredient_to_product", "OnSIDES"), ("onsides_vocab_rxnorm_product", "OnSIDES"),
    ("onsides_high_confidence", "OnSIDES"),
]
DERIVED_TABLES = ["drug_side_effects_clean", "onsides_single_ingredient_labels",
                  "onsides_drug_label_counts", "onsides_drug_effects"]


def draw_table(header, rows, title, out_name, total_row=None):
    data = rows + ([total_row] if total_row else [])
    fig, ax = plt.subplots(figsize=(max(7, 2.2 * len(header)), 0.5 * len(data) + 1.2))
    ax.axis("off")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12, loc="left")
    t = ax.table(cellText=data, colLabels=header, cellLoc="center", loc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(10.5)
    t.scale(1, 1.7)
    for c in range(len(header)):
        t[0, c].set_facecolor("#2E5C8A")
        t[0, c].set_text_props(color="white", fontweight="bold")
    for i in range(len(data)):
        for c in range(len(header)):
            t[i + 1, c].set_facecolor("#F5F5F5" if i % 2 == 0 else "white")
            t[i + 1, c].set_edgecolor("#DDDDDD")
            if total_row and i == len(data) - 1:
                t[i + 1, c].set_facecolor("#DCE6F0")
                t[i + 1, c].set_text_props(fontweight="bold")
    out = cfg.RESULTS_DIR / "figures" / out_name
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    (cfg.RESULTS_DIR / "figures").mkdir(parents=True, exist_ok=True)
    stats = {}

    # ---- dataset ----
    ds = json.load(open(cfg.RESULTS_DIR / "dataset_stats.json"))
    stats["dataset"] = ds
    sources = ["BC5CDR", "BioRED", "ADE"]
    rows, totals = [], [0, 0, 0, 0]
    for split, label in [("train", "Train"), ("val", "Validation"), ("test", "Test")]:
        vals = [ds[split]["by_source"].get(s, 0) for s in sources]
        rows.append([label] + [f"{v:,}" for v in vals] + [f"{sum(vals):,}"])
        totals = [t + v for t, v in zip(totals, vals + [sum(vals)])]
    draw_table(["Split / Source", "BC5CDR", "BioRED", "ADE Corpus", "Total"], rows,
               "Training data - records per split", "training_data_table.png",
               ["Total"] + [f"{t:,}" for t in totals])

    # ---- PostgreSQL ----
    pg = psycopg2.connect(**cfg.POSTGRES)
    cur = pg.cursor()
    kb = {}
    for table, _ in SOURCE_TABLES + [(t, "derived") for t in DERIVED_TABLES]:
        cur.execute("SELECT to_regclass(%s) IS NOT NULL", (f"public.{table}",))
        if cur.fetchone()[0]:
            cur.execute(f"SELECT COUNT(*) FROM {table}")
            kb[table] = cur.fetchone()[0]
    pg.close()
    stats["postgres"] = kb
    src_rows = [[t, s, f"{kb.get(t, 0):,}"] for t, s in SOURCE_TABLES]
    total = sum(kb.get(t, 0) for t, _ in SOURCE_TABLES)
    draw_table(["Table", "Source", "Rows"], src_rows, "Knowledge base - PostgreSQL source tables",
               "knowledge_base_table.png", ["Total (12 tables)", "SIDER + OnSIDES", f"{total:,}"])

    # ---- Neo4j ----
    driver = GraphDatabase.driver(cfg.NEO4J_URI, auth=cfg.NEO4J_AUTH)
    with driver.session() as s:
        nodes = {r["label"]: r["n"] for r in s.run("MATCH (n) RETURN labels(n)[0] AS label, count(*) AS n")}
        rels = {r["t"]: r["n"] for r in s.run("MATCH ()-[r]->() RETURN type(r) AS t, count(*) AS n")}
        model = {r["rel"]: {"count": r["n"], "mean_confidence": round(r["c"], 4)} for r in s.run(
            "MATCH ()-[r:MODEL_PREDICTED]->() RETURN r.relation AS rel, count(*) AS n, avg(r.confidence) AS c")}
    driver.close()
    stats["neo4j"] = {"nodes": nodes, "relationships": rels, "model_predicted": model}

    with open(cfg.RESULTS_DIR / "kb_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps({"postgres": kb, "neo4j": stats["neo4j"]}, indent=2))
    print(f"Saved {cfg.RESULTS_DIR / 'kb_stats.json'}")
