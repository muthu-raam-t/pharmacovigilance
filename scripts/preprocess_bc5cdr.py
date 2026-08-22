import os
from preprocessing_utils import tokenize_with_spans, assign_bio_tags, write_jsonl

def parse_pubtator_file(filepath):
    """Parses a PubTator-format .txt file into a list of documents."""
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
                # entity annotation: doc_id start end text type mesh_id
                doc_id, start, end, text, etype, mesh_id = parts
                documents.setdefault(doc_id, {"title": "", "abstract": "", "entities": [], "relations": []})
                documents[doc_id]["entities"].append({
                    "char_start": int(start), "char_end": int(end),
                    "text": text, "type": etype, "mesh_id": mesh_id
                })
            elif len(parts) == 4 and parts[1] == "CID":
                # relation annotation: doc_id CID chemical_mesh disease_mesh
                doc_id, _, chem_id, dis_id = parts
                documents.setdefault(doc_id, {"title": "", "abstract": "", "entities": [], "relations": []})
                documents[doc_id]["relations"].append({
                    "type": "CID", "chemical_mesh": chem_id, "disease_mesh": dis_id
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

        mesh_to_entity_id = {}
        for ent in resolved_entities:
            src = next((e for e in doc["entities"] if e["id"] == ent["id"]), None)
            if src and "mesh_id" in src:
                mesh_to_entity_id[src["mesh_id"]] = ent["id"]

        relations = []
        for rel in doc["relations"]:
            arg1 = mesh_to_entity_id.get(rel["chemical_mesh"])
            arg2 = mesh_to_entity_id.get(rel["disease_mesh"])
            if arg1 and arg2:
                relations.append({"type": "CID", "arg1": arg1, "arg2": arg2})

        records.append({
            "doc_id": doc_id,
            "text": text,
            "tokens": tokens,
            "ner_tags": ner_tags,
            "entities": resolved_entities,
            "relations": relations,
            "source_dataset": "BC5CDR"
        })
    return records

if __name__ == "__main__":
    RAW_DIR = "/workspace/data/raw/bc5cdr"
    OUT_DIR = "/workspace/data/processed"
    os.makedirs(OUT_DIR, exist_ok=True)

    all_records = []
    for split_file in os.listdir(RAW_DIR):
        if split_file.endswith(".txt"):
            docs = parse_pubtator_file(os.path.join(RAW_DIR, split_file))
            all_records.extend(process_documents(docs))

    write_jsonl(all_records, os.path.join(OUT_DIR, "bc5cdr_processed.jsonl"))
