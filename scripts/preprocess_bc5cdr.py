"""BC5CDR (PubTator format) -> unified JSONL.

Chemical-induced-disease (CID) relations are annotated between MeSH concepts at
document level. They are expanded to every mention pair of the two concepts,
so that no mention of a related concept pair is mislabelled as NONE.
"""
import os
from preprocessing_utils import (tokenize_with_spans, assign_bio_tags, split_concept_ids,
                                 expand_concept_relations, write_jsonl,
                                 RAW_DIR as RAW_ROOT, PROCESSED_DIR)

RAW_DIR = str(RAW_ROOT / "bc5cdr")
OUT_DIR = str(PROCESSED_DIR)


def _new_doc():
    return {"title": "", "abstract": "", "entities": [], "relations": []}


def parse_pubtator_file(filepath):
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
                    doc_id, start, end, text, etype, mesh = parts[:6]
                    documents.setdefault(doc_id, _new_doc())["entities"].append({
                        "char_start": int(start), "char_end": int(end),
                        "text": text, "type": etype, "concept_ids": split_concept_ids(mesh),
                    })
                elif len(parts) >= 4 and parts[1] == "CID":
                    doc_id, _, chem, dis = parts[:4]
                    documents.setdefault(doc_id, _new_doc())["relations"].append((chem, dis, "CID"))
    return documents


def process_documents(documents):
    records = []
    for doc_id, doc in documents.items():
        # PubTator offsets assume title + single space + abstract
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
            "doc_id": f"BC5CDR_{doc_id}",
            "group_id": doc_id,
            "text": text,
            "tokens": tokens,
            "ner_tags": ner_tags,
            "entities": resolved,
            "relations": relations,
            "source_dataset": "BC5CDR",
        })
    return records


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    all_records = []
    for split_file in sorted(os.listdir(RAW_DIR)):
        if split_file.endswith(".txt") or split_file.endswith(".PubTator"):
            docs = parse_pubtator_file(os.path.join(RAW_DIR, split_file))
            recs = process_documents(docs)
            print(f"{split_file}: {len(recs)} documents")
            all_records.extend(recs)
    write_jsonl(all_records, os.path.join(OUT_DIR, "bc5cdr_processed.jsonl"))
