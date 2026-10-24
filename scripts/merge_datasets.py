"""Merges the three processed corpora, unifies labels and creates leakage-free splits.

Relation mapping (type-aware):
  * BC5CDR CID, ADE ADVERSE                    -> CAUSES
  * BioRED Chemical-Disease Positive_Correlation -> CAUSES   (chemical induces / worsens disease)
  * BioRED Chemical-Disease Negative_Correlation -> TREATS   (chemical treats / prevents disease)
  * every other BioRED relation                 -> ASSOCIATED

Splitting is done by PubMed id (group_id), so every sentence of an abstract, and
an abstract that appears in more than one corpus, always lands in one split.
"""
import json
import os
import random
from collections import Counter, defaultdict

from preprocessing_utils import load_jsonl, PROCESSED_DIR, RESULTS_DIR

IN_DIR = str(PROCESSED_DIR)
SEED = 42
TRAIN_FRAC, VAL_FRAC = 0.8, 0.1

LABEL_MAP = {
    "Chemical": "Chemical", "Drug": "Chemical", "ChemicalEntity": "Chemical",
    "Disease": "Disease", "DiseaseOrPhenotypicFeature": "Disease",
    "GeneOrGeneProduct": "Gene", "Gene": "Gene",
    "OrganismTaxon": "Organism",
    "SequenceVariant": "Variant",
    "CellLine": "CellLine",
}

ALWAYS_CAUSES = {"CID", "ADVERSE"}
CHEM_DISEASE = {"Chemical", "Disease"}


def map_relation(raw_type, type1, type2):
    if raw_type in ALWAYS_CAUSES:
        return "CAUSES"
    if {type1, type2} == CHEM_DISEASE:
        if raw_type == "Positive_Correlation":
            return "CAUSES"
        if raw_type == "Negative_Correlation":
            return "TREATS"
    return "ASSOCIATED"


def normalize_record(record):
    for ent in record["entities"]:
        ent["type"] = LABEL_MAP.get(ent["type"], ent["type"])
    types = {e["id"]: e["type"] for e in record["entities"]}

    relations = []
    for rel in record["relations"]:
        t1, t2 = types.get(rel["arg1"]), types.get(rel["arg2"])
        if t1 is None or t2 is None:
            continue
        rel["original_type"] = rel["type"]
        rel["type"] = map_relation(rel["type"], t1, t2)
        relations.append(rel)
    record["relations"] = relations

    new_tags = []
    for tag in record["ner_tags"]:
        if tag == "O":
            new_tags.append(tag)
        else:
            prefix, raw_type = tag.split("-", 1)
            new_tags.append(f"{prefix}-{LABEL_MAP.get(raw_type, raw_type)}")
    record["ner_tags"] = new_tags
    return record


def grouped_split(records):
    by_group = defaultdict(list)
    for r in records:
        by_group[str(r.get("group_id", r["doc_id"]))].append(r)

    group_ids = sorted(by_group)
    random.Random(SEED).shuffle(group_ids)

    n_total = len(records)
    train, val, test = [], [], []
    seen = 0
    for gid in group_ids:
        recs = by_group[gid]
        if seen < TRAIN_FRAC * n_total:
            train.extend(recs)
        elif seen < (TRAIN_FRAC + VAL_FRAC) * n_total:
            val.extend(recs)
        else:
            test.extend(recs)
        seen += len(recs)

    for split in (train, val, test):
        random.Random(SEED).shuffle(split)
    return train, val, test


def describe(split_records):
    return {
        "records": len(split_records),
        "by_source": dict(Counter(r["source_dataset"] for r in split_records)),
        "entities": dict(Counter(e["type"] for r in split_records for e in r["entities"])),
        "relations": dict(Counter(rel["type"] for r in split_records for rel in r["relations"])),
    }


if __name__ == "__main__":
    files = ["bc5cdr_processed.jsonl", "biored_processed.jsonl", "ade_processed.jsonl"]
    all_records = []
    for fname in files:
        recs = load_jsonl(os.path.join(IN_DIR, fname))
        all_records.extend(normalize_record(r) for r in recs)
        print(f"Loaded {len(recs)} records from {fname}")

    train, val, test = grouped_split(all_records)

    stats = {}
    for name, split in [("train", train), ("val", val), ("test", test)]:
        out_path = os.path.join(IN_DIR, f"ner_re_{name}.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for r in split:
                f.write(json.dumps(r) + "\n")
        stats[name] = describe(split)
        print(f"{name}: {len(split)} records -> {out_path}")
        print(f"   sources:   {stats[name]['by_source']}")
        print(f"   relations: {stats[name]['relations']}")

    groups = {name: {str(r.get('group_id')) for r in s} for name, s in [("train", train), ("val", val), ("test", test)]}
    overlap = (groups["train"] & groups["val"]) | (groups["train"] & groups["test"]) | (groups["val"] & groups["test"])
    print(f"PubMed ids shared between splits (should be 0): {len(overlap)}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "dataset_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved split statistics to {os.path.join(RESULTS_DIR, 'dataset_stats.json')}")
