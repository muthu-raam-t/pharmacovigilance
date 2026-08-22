import os
from preprocessing_utils import tokenize_with_spans, assign_bio_tags, write_jsonl

def parse_biored_file(filepath):
    documents = {}
    with open(filepath, encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        line = line.rstrip("\n")
        if not line.strip():
            continue

        if "|t|" in line:
            doc_id, _, title = line.split("|", 2)
            documents.setdefault(doc_id, {"title": "", "abstract": "", "entities": [], "relations": []})
            documents[doc_id]["title"] = title

        elif "|a|" in line:
            doc_id, _, abstract = line.split("|", 2)
            documents.setdefault(doc_id, {"title": "", "abstract": "", "entities": [], "relations": []})
            documents[doc_id]["abstract"] = abstract

        else:
            parts = line.split("\t")
            if len(parts) == 6:
                doc_id, start, end, text, etype, entity_id = parts
                documents.setdefault(doc_id, {"title": "", "abstract": "", "entities": [], "relations": []})
                documents[doc_id]["entities"].append({
                    "char_start": int(start), "char_end": int(end),
                    "text": text, "type": etype, "norm_id": entity_id
                })
            elif len(parts) >= 4 and not parts[1].isdigit():
                # relation line: doc_id  relation_type  entity1_id  entity2_id  [novel]
                doc_id, rel_type, id1, id2 = parts[0], parts[1], parts[2], parts[3]
                documents.setdefault(doc_id, {"title": "", "abstract": "", "entities": [], "relations": []})
                documents[doc_id]["relations"].append({
                    "type": rel_type, "norm_id_1": id1, "norm_id_2": id2
                })

    return documents

def process_documents(documents):
    records = []
    for doc_id, doc in documents.items():
        text = (doc["title"] + " " + doc["abstract"]).strip()
        if not text:
            continue

        tokens_with_spans = tokenize_with_spans(text)
        tokens = [t[0] for t in tokens_with_spans]

        for i, ent in enumerate(doc["entities"]):
            ent["id"] = f"T{i}"

        ner_tags, resolved_entities = assign_bio_tags(tokens_with_spans, doc["entities"])

        norm_to_entity_id = {}
        for ent in resolved_entities:
            src = next((e for e in doc["entities"] if e["id"] == ent["id"]), None)
            if src and "norm_id" in src:
                norm_to_entity_id[src["norm_id"]] = ent["id"]

        relations = []
        for rel in doc["relations"]:
            arg1 = norm_to_entity_id.get(rel["norm_id_1"])
            arg2 = norm_to_entity_id.get(rel["norm_id_2"])
            if arg1 and arg2:
                relations.append({"type": rel["type"], "arg1": arg1, "arg2": arg2})

        records.append({
            "doc_id": doc_id,
            "text": text,
            "tokens": tokens,
            "ner_tags": ner_tags,
            "entities": resolved_entities,
            "relations": relations,
            "source_dataset": "BioRED"
        })
    return records

if __name__ == "__main__":
    RAW_DIR = "/workspace/data/raw/biored"
    OUT_DIR = "/workspace/data/processed"
    os.makedirs(OUT_DIR, exist_ok=True)

    all_records = []
    for split_file in os.listdir(RAW_DIR):
        if split_file.endswith(".txt") or split_file.endswith(".PubTator"):
            docs = parse_biored_file(os.path.join(RAW_DIR, split_file))
            all_records.extend(process_documents(docs))

    write_jsonl(all_records, os.path.join(OUT_DIR, "biored_processed.jsonl"))
