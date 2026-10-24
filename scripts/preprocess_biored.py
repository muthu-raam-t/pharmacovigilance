"""BioRED (PubTator format) -> unified JSONL.

BioRED relations are document-level and undirected between normalised concept
ids. They are expanded to every mention pair of the two concepts. The original
relation type is kept here; merge_datasets.py maps it to the unified label set.
"""
import os
from preprocessing_utils import (tokenize_with_spans, assign_bio_tags, split_concept_ids,
                                 expand_concept_relations, write_jsonl,
                                 RAW_DIR as RAW_ROOT, PROCESSED_DIR)

RAW_DIR = str(RAW_ROOT / "biored")
OUT_DIR = str(PROCESSED_DIR)


def _new_doc():
    return {"title": "", "abstract": "", "entities": [], "relations": []}


def parse_biored_file(filepath):
    documents = {}
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            if "|t|" in line:
                doc_id, _, title = line.split("|", 2)
                documents.setdefault(doc_id, _new_doc())["title"] = title
            elif "|a|" in line:
                doc_id, _, abstract = line.split("|", 2)
                documents.setdefault(doc_id, _new_doc())["abstract"] = abstract
            else:
                parts = line.split("\t")
                if len(parts) >= 6 and parts[1].isdigit():
                    doc_id, start, end, text, etype, concept = parts[:6]
                    documents.setdefault(doc_id, _new_doc())["entities"].append({
                        "char_start": int(start), "char_end": int(end),
                        "text": text, "type": etype, "concept_ids": split_concept_ids(concept),
                    })
                elif len(parts) >= 4 and not parts[1].isdigit():
                    # doc_id  relation_type  concept_1  concept_2  [Novel|No]
                    doc_id, rel_type, id1, id2 = parts[:4]
                    documents.setdefault(doc_id, _new_doc())["relations"].append((id1, id2, rel_type))
    return documents


def process_documents(documents):
    records = []
    for doc_id, doc in documents.items():
        text = doc["title"] + " " + doc["abstract"]
        if not text.strip():
            continue
        tokens_with_spans = tokenize_with_spans(text)
        tokens = [t[0] for t in tokens_with_spans]

        for i, ent in enumerate(doc["entities"]):
            ent["id"] = f"T{i}"
        ner_tags, resolved = assign_bio_tags(tokens_with_spans, doc["entities"])
        relations = expand_concept_relations(resolved, doc["relations"])

        records.append({
            "doc_id": f"BioRED_{doc_id}",
            "group_id": doc_id,
            "text": text,
            "tokens": tokens,
            "ner_tags": ner_tags,
            "entities": resolved,
            "relations": relations,
            "source_dataset": "BioRED",
        })
    return records


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    all_records = []
    for split_file in sorted(os.listdir(RAW_DIR)):
        if split_file.endswith(".txt") or split_file.endswith(".PubTator"):
            docs = parse_biored_file(os.path.join(RAW_DIR, split_file))
            recs = process_documents(docs)
            print(f"{split_file}: {len(recs)} documents")
            all_records.extend(recs)
    write_jsonl(all_records, os.path.join(OUT_DIR, "biored_processed.jsonl"))
