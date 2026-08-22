import json
from collections import Counter

def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def run_sanity_check(path, split_name):
    records = load_jsonl(path)
    entity_type_counts = Counter()
    relation_type_counts = Counter()
    source_counts = Counter()
    zero_len_entities = 0
    total_tokens = 0

    for r in records:
        source_counts[r["source_dataset"]] += 1
        total_tokens += len(r["tokens"])
        for ent in r["entities"]:
            entity_type_counts[ent["type"]] += 1
            if ent["token_end"] <= ent["token_start"]:
                zero_len_entities += 1
        for rel in r["relations"]:
            relation_type_counts[rel["type"]] += 1

    print(f"\n=== {split_name} ===")
    print(f"Total records: {len(records)}")
    print(f"Avg tokens/record: {total_tokens / len(records):.1f}")
    print(f"Records by source dataset: {dict(source_counts)}")
    print(f"Entity type distribution: {dict(entity_type_counts)}")
    print(f"Relation type distribution: {dict(relation_type_counts)}")
    print(f"Zero-length entities (should be 0): {zero_len_entities}")

    for rtype, count in relation_type_counts.items():
        if count < 10:
            print(f"WARNING: relation type '{rtype}' has only {count} examples — severe imbalance risk")

if __name__ == "__main__":
    BASE = "/workspace/data/processed"
    for split in ["train", "val", "test"]:
        run_sanity_check(f"{BASE}/ner_re_{split}.jsonl", split)
