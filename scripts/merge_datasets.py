import json
import random
import os

LABEL_MAP = {
    "Chemical": "Chemical", "Drug": "Chemical", "ChemicalEntity": "Chemical",
    "Disease": "Disease", "DiseaseOrPhenotypicFeature": "Disease",
    "GeneOrGeneProduct": "Gene", "Gene": "Gene",
    "OrganismTaxon": "Organism",
    "SequenceVariant": "Variant",
    "CellLine": "CellLine",
}

RELATION_MAP = {
    "CID": "CAUSES", "ADVERSE": "CAUSES",
    "Treats": "TREATS",
    "Positive_Correlation": "ASSOCIATED",
    "Negative_Correlation": "ASSOCIATED",
    "Association": "ASSOCIATED",
    "ASSOCIATED": "ASSOCIATED",
    "Bind": "ASSOCIATED",
    "Comparison": "ASSOCIATED",
    "Cotreatment": "ASSOCIATED",
    "Conversion": "ASSOCIATED",
    "Drug_Interaction": "ASSOCIATED",
}

def normalize_record(record):
    for ent in record["entities"]:
        ent["type"] = LABEL_MAP.get(ent["type"], ent["type"])
    for rel in record["relations"]:
        rel["type"] = RELATION_MAP.get(rel["type"], rel["type"])
    return record

def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]

if __name__ == "__main__":
    IN_DIR = "/workspace/data/processed"
    files = ["bc5cdr_processed.jsonl", "biored_processed.jsonl", "ade_processed.jsonl"]

    all_records = []
    for fname in files:
        path = os.path.join(IN_DIR, fname)
        recs = load_jsonl(path)
        all_records.extend([normalize_record(r) for r in recs])

    random.seed(42)
    random.shuffle(all_records)

    n = len(all_records)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)

    train = all_records[:train_end]
    val = all_records[train_end:val_end]
    test = all_records[val_end:]

    for name, split in [("train", train), ("val", val), ("test", test)]:
        out_path = os.path.join(IN_DIR, f"ner_re_{name}.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for r in split:
                f.write(json.dumps(r) + "\n")
        print(f"{name}: {len(split)} records -> {out_path}")
