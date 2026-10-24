"""Split statistics and integrity checks (span validity, BIO validity, leakage)."""
from collections import Counter

from preprocessing_utils import load_jsonl, PROCESSED_DIR

BASE = str(PROCESSED_DIR)


def check_split(records, split_name):
    entity_counts, relation_counts, source_counts = Counter(), Counter(), Counter()
    zero_len, bad_bio, dangling_rel, total_tokens = 0, 0, 0, 0

    for r in records:
        source_counts[r["source_dataset"]] += 1
        total_tokens += len(r["tokens"])
        ids = set()
        for ent in r["entities"]:
            ids.add(ent["id"])
            entity_counts[ent["type"]] += 1
            if ent["token_end"] <= ent["token_start"]:
                zero_len += 1
            elif r["ner_tags"][ent["token_start"]] != f"B-{ent['type']}":
                bad_bio += 1
        for rel in r["relations"]:
            relation_counts[rel["type"]] += 1
            if rel["arg1"] not in ids or rel["arg2"] not in ids:
                dangling_rel += 1
        prev = "O"
        for tag in r["ner_tags"]:
            if tag.startswith("I-") and prev[2:] != tag[2:]:
                bad_bio += 1
            prev = tag

    print(f"\n=== {split_name} ===")
    print(f"Total records: {len(records)}")
    print(f"Avg tokens/record: {total_tokens / max(len(records), 1):.1f}")
    print(f"Records by source dataset: {dict(source_counts)}")
    print(f"Entity type distribution: {dict(entity_counts)}")
    print(f"Relation type distribution: {dict(relation_counts)}")
    print(f"Zero-length entities (should be 0): {zero_len}")
    print(f"Invalid BIO sequences (should be 0): {bad_bio}")
    print(f"Relations pointing to missing entities (should be 0): {dangling_rel}")
    for rtype, count in relation_counts.items():
        if count < 10:
            print(f"WARNING: relation type '{rtype}' has only {count} examples - severe imbalance risk")


if __name__ == "__main__":
    splits = {s: load_jsonl(f"{BASE}/ner_re_{s}.jsonl") for s in ["train", "val", "test"]}
    for name, recs in splits.items():
        check_split(recs, name)

    print("\n=== Leakage checks ===")
    texts = {s: {r["text"].strip().lower() for r in recs} for s, recs in splits.items()}
    groups = {s: {str(r.get("group_id")) for r in recs} for s, recs in splits.items()}
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        print(f"{a}/{b}: identical texts = {len(texts[a] & texts[b])}, shared PubMed ids = {len(groups[a] & groups[b])}  (both should be 0)")
