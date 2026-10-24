"""Builds NER / RE label maps and the set of entity-type pairs that can hold a relation."""
import json

from preprocessing_utils import load_jsonl, PROCESSED_DIR

BASE = str(PROCESSED_DIR)

if __name__ == "__main__":
    records = load_jsonl(f"{BASE}/ner_re_train.jsonl")

    ner_tags, re_labels, type_pairs = set(), {"NONE"}, set()
    for r in records:
        ner_tags.update(r["ner_tags"])
        types = {e["id"]: e["type"] for e in r["entities"]}
        for rel in r["relations"]:
            re_labels.add(rel["type"])
            t1, t2 = types.get(rel["arg1"]), types.get(rel["arg2"])
            if t1 and t2:
                type_pairs.add(tuple(sorted((t1, t2))))

    ner_tags.discard("O")
    ner_tags = ["O"] + sorted(ner_tags)
    re_labels.discard("NONE")
    re_labels = ["NONE"] + sorted(re_labels)

    label_maps = {
        "ner_label2id": {label: i for i, label in enumerate(ner_tags)},
        "ner_id2label": {str(i): label for i, label in enumerate(ner_tags)},
        "re_label2id": {label: i for i, label in enumerate(re_labels)},
        "re_id2label": {str(i): label for i, label in enumerate(re_labels)},
        "re_type_pairs": [list(p) for p in sorted(type_pairs)],
    }

    out_path = f"{BASE}/label_maps.json"
    with open(out_path, "w") as f:
        json.dump(label_maps, f, indent=2)

    print(f"NER labels ({len(ner_tags)}): {ner_tags}")
    print(f"RE labels ({len(re_labels)}): {re_labels}")
    print(f"Entity-type pairs that hold relations ({len(type_pairs)}): {sorted(type_pairs)}")
    print(f"Saved to {out_path}")
