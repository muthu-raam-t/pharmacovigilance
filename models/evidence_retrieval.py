import json
import os

BASE = "/workspace/data/processed"
SPLITS = ["ner_re_train.jsonl", "ner_re_val.jsonl", "ner_re_test.jsonl"]


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def find_evidence_sentences(drug_name, disease_name, max_results=3):
    """
    Searches all preprocessed records for sentences where both the drug and
    disease appear as annotated entities, and returns the relation type
    found between them plus the source sentence text.
    """
    drug_name = drug_name.strip().lower()
    disease_name = disease_name.strip().lower()
    evidence = []

    for split_file in SPLITS:
        path = os.path.join(BASE, split_file)
        if not os.path.exists(path):
            continue
        records = load_jsonl(path)

        for record in records:
            entity_map = {e["id"]: e for e in record["entities"]}
            drug_ids = {e["id"] for e in record["entities"]
                        if e["type"] == "Chemical" and e["text"].strip().lower() == drug_name}
            disease_ids = {e["id"] for e in record["entities"]
                           if e["type"] == "Disease" and e["text"].strip().lower() == disease_name}

            if not drug_ids or not disease_ids:
                continue

            for rel in record["relations"]:
                a1, a2 = rel["arg1"], rel["arg2"]
                if (a1 in drug_ids and a2 in disease_ids) or (a2 in drug_ids and a1 in disease_ids):
                    evidence.append({
                        "sentence": record["text"],
                        "relation_type": rel["type"],
                        "source_dataset": record["source_dataset"],
                        "doc_id": record.get("doc_id", "unknown")
                    })
                    if len(evidence) >= max_results:
                        return evidence

    return evidence


if __name__ == "__main__":
    results = find_evidence_sentences("alpha-methyldopa", "hypotensive")
    print(f"Found {len(results)} evidence sentence(s)\n")
    for r in results:
        print(f"[{r['source_dataset']}] relation={r['relation_type']}")
        print(f"  \"{r['sentence']}\"\n")
