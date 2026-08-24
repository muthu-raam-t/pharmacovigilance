import json
import os

def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]

if __name__ == "__main__":
    BASE = "/workspace/data/processed"
    records = load_jsonl(f"{BASE}/ner_re_train.jsonl")

    ner_tags = set()
    re_labels = {"NONE"}
    for r in records:
        for tag in r["ner_tags"]:
            ner_tags.add(tag)
        for rel in r["relations"]:
            re_labels.add(rel["type"])

    ner_tags = sorted(ner_tags)
    if "O" in ner_tags:
        ner_tags.remove("O")
        ner_tags = ["O"] + ner_tags

    re_labels = sorted(re_labels)
    if "NONE" in re_labels:
        re_labels.remove("NONE")
        re_labels = ["NONE"] + re_labels

    label_maps = {
        "ner_label2id": {label: i for i, label in enumerate(ner_tags)},
        "ner_id2label": {i: label for i, label in enumerate(ner_tags)},
        "re_label2id": {label: i for i, label in enumerate(re_labels)},
        "re_id2label": {i: label for i, label in enumerate(re_labels)},
    }

    out_path = f"{BASE}/label_maps.json"
    with open(out_path, "w") as f:
        json.dump(label_maps, f, indent=2)

    print(f"NER labels ({len(ner_tags)}): {ner_tags}")
    print(f"RE labels ({len(re_labels)}): {re_labels}")
    print(f"Saved to {out_path}")
