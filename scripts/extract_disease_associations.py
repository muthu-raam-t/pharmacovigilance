import json
import os

BASE = "/workspace/data/processed"
SPLITS = ["ner_re_train.jsonl", "ner_re_val.jsonl", "ner_re_test.jsonl"]
OUT_PATH = os.path.join(BASE, "drug_disease_associations.jsonl")


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def extract_from_record(record):
    entity_map = {e["id"]: e for e in record["entities"]}
    pairs = []

    for rel in record["relations"]:
        e1 = entity_map.get(rel["arg1"])
        e2 = entity_map.get(rel["arg2"])
        if not e1 or not e2:
            continue

        drug_ent, disease_ent = None, None
        if e1["type"] == "Chemical" and e2["type"] == "Disease":
            drug_ent, disease_ent = e1, e2
        elif e2["type"] == "Chemical" and e1["type"] == "Disease":
            drug_ent, disease_ent = e2, e1
        else:
            continue

        pairs.append({
            "drug_name": drug_ent["text"].strip().lower(),
            "disease_name": disease_ent["text"].strip().lower(),
            "relation": rel["type"],
            "source_dataset": record["source_dataset"]
        })

    return pairs


if __name__ == "__main__":
    all_pairs = []
    for split_file in SPLITS:
        path = os.path.join(BASE, split_file)
        records = load_jsonl(path)
        for record in records:
            all_pairs.extend(extract_from_record(record))

    seen = set()
    unique_pairs = []
    for p in all_pairs:
        key = (p["drug_name"], p["disease_name"], p["relation"])
        if key not in seen:
            seen.add(key)
            unique_pairs.append(p)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for p in unique_pairs:
            f.write(json.dumps(p) + "\n")

    print(f"Extracted {len(all_pairs)} total pairs, {len(unique_pairs)} unique")
    print(f"Saved to {OUT_PATH}")
