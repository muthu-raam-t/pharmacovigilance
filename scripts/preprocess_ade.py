import os
from preprocessing_utils import tokenize_with_spans, assign_bio_tags, write_jsonl

def parse_ade_file(filepath):
    records_raw = []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            parts = [p.strip() for p in line.strip().split("|")]
            if len(parts) == 8:
                pmid, sentence, effect, eff_start, eff_end, drug, drug_start, drug_end = parts
                records_raw.append({
                    "pmid": pmid, "sentence": sentence,
                    "effect": effect, "eff_start": int(eff_start), "eff_end": int(eff_end),
                    "drug": drug, "drug_start": int(drug_start), "drug_end": int(drug_end)
                })
            elif len(parts) == 4:
                pmid, sentence, effect, drug = parts
                eff_start = sentence.find(effect)
                drug_start = sentence.find(drug)
                if eff_start == -1 or drug_start == -1:
                    continue
                records_raw.append({
                    "pmid": pmid, "sentence": sentence,
                    "effect": effect, "eff_start": eff_start, "eff_end": eff_start + len(effect),
                    "drug": drug, "drug_start": drug_start, "drug_end": drug_start + len(drug)
                })
    return records_raw

def process_records(records_raw):
    records = []
    for i, r in enumerate(records_raw):
        text = r["sentence"]
        tokens_with_spans = tokenize_with_spans(text)
        tokens = [t[0] for t in tokens_with_spans]

        entities_raw = [
            {"id": "T0", "char_start": r["drug_start"], "char_end": r["drug_end"],
             "text": r["drug"], "type": "Chemical"},
            {"id": "T1", "char_start": r["eff_start"], "char_end": r["eff_end"],
             "text": r["effect"], "type": "Disease"}
        ]

        ner_tags, resolved_entities = assign_bio_tags(tokens_with_spans, entities_raw)

        if len(resolved_entities) < 2:
            continue

        relations = [{"type": "ADVERSE", "arg1": "T0", "arg2": "T1"}]

        records.append({
            "doc_id": f"ADE_{r['pmid']}_{i}",
            "text": text,
            "tokens": tokens,
            "ner_tags": ner_tags,
            "entities": resolved_entities,
            "relations": relations,
            "source_dataset": "ADE"
        })
    return records

if __name__ == "__main__":
    RAW_PATH = "/workspace/data/raw/ade_corpus/DRUG-AE.rel"
    OUT_DIR = "/workspace/data/processed"
    os.makedirs(OUT_DIR, exist_ok=True)

    raw = parse_ade_file(RAW_PATH)
    records = process_records(raw)
    write_jsonl(records, os.path.join(OUT_DIR, "ade_processed.jsonl"))

