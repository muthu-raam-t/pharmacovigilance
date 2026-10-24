"""Extracts gold (human-annotated) drug-disease relations from the processed corpora.

One row per unique (drug, disease, relation) with its support (number of documents)."""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "models"))
import pv_config as cfg  # noqa: E402
from text_utils import normalize_name  # noqa: E402

SPLITS = ["ner_re_train.jsonl", "ner_re_val.jsonl", "ner_re_test.jsonl"]
OUT_PATH = cfg.PROCESSED_DIR / "drug_disease_associations.jsonl"

if __name__ == "__main__":
    agg = {}
    for split_file in SPLITS:
        with open(cfg.PROCESSED_DIR / split_file, encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                ents = {e["id"]: e for e in record["entities"]}
                for rel in record["relations"]:
                    a, b = ents.get(rel["arg1"]), ents.get(rel["arg2"])
                    if not a or not b or {a["type"], b["type"]} != {"Chemical", "Disease"}:
                        continue
                    chem, dis = (a, b) if a["type"] == "Chemical" else (b, a)
                    key = (normalize_name(chem["text"]), normalize_name(dis["text"]), rel["type"])
                    entry = agg.setdefault(key, {"docs": set(), "sources": set()})
                    entry["docs"].add(record["doc_id"])
                    entry["sources"].add(record["source_dataset"])

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for (drug, disease, relation), v in agg.items():
            f.write(json.dumps({"drug_name": drug, "disease_name": disease, "relation": relation,
                                "support": len(v["docs"]), "source_datasets": sorted(v["sources"])}) + "\n")

    counts = {}
    for (_, _, rel) in agg:
        counts[rel] = counts.get(rel, 0) + 1
    print(f"Unique gold drug-disease relations: {len(agg):,} {counts}")
    print(f"Saved to {OUT_PATH}")
