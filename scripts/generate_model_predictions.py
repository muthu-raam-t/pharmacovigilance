import json
import os
import sys
import torch
from transformers import AutoTokenizer

sys.path.append("/workspace")
from models.joint_model import JointNERREModel

BASE = "/workspace/data/processed"
SPLITS = ["ner_re_train.jsonl", "ner_re_val.jsonl", "ner_re_test.jsonl"]
LABEL_MAPS_PATH = f"{BASE}/label_maps.json"
CHECKPOINT_PATH = "/workspace/models/checkpoints/scibert/model.pt"
BACKBONE = "allenai/scibert_scivocab_uncased"
OUT_PATH = f"{BASE}/model_predicted_associations.jsonl"
CONFIDENCE_THRESHOLD = 0.60
MAX_LENGTH = 256


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def get_entity_subword_span(word_ids, ent):
    ws, we = ent["token_start"], ent["token_end"]
    positions = [i for i, w_id in enumerate(word_ids) if w_id is not None and ws <= w_id < we]
    if not positions:
        return None
    return (min(positions), max(positions) + 1)


def main():
    with open(LABEL_MAPS_PATH) as f:
        label_maps = json.load(f)
    id2re = {int(k): v for k, v in label_maps["re_id2label"].items()}
    num_ner_labels = len(label_maps["ner_label2id"])
    num_re_labels = len(label_maps["re_label2id"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(BACKBONE)
    model = JointNERREModel(BACKBONE, num_ner_labels, num_re_labels).to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()
    print(f"Loaded checkpoint: {CHECKPOINT_PATH}")

    results = []
    total_records = 0

    with torch.no_grad():
        for split_file in SPLITS:
            path = os.path.join(BASE, split_file)
            records = load_jsonl(path)
            for record in records:
                total_records += 1
                tokens = record["tokens"]
                chem_entities = [e for e in record["entities"] if e["type"] == "Chemical"]
                disease_entities = [e for e in record["entities"] if e["type"] == "Disease"]
                if not chem_entities or not disease_entities:
                    continue

                encoding = tokenizer(tokens, is_split_into_words=True, truncation=True,
                                     max_length=MAX_LENGTH, padding="max_length", return_tensors="pt")
                input_ids = encoding["input_ids"].to(device)
                attention_mask = encoding["attention_mask"].to(device)
                word_ids = encoding.word_ids(batch_index=0)

                sequence_output = model.encode(input_ids, attention_mask)

                pairs, pair_meta = [], []
                for chem in chem_entities:
                    chem_span = get_entity_subword_span(word_ids, chem)
                    if not chem_span:
                        continue
                    for dis in disease_entities:
                        dis_span = get_entity_subword_span(word_ids, dis)
                        if not dis_span:
                            continue
                        pairs.append((chem_span[0], chem_span[1], dis_span[0], dis_span[1]))
                        pair_meta.append((chem["text"].strip().lower(), dis["text"].strip().lower()))

                if not pairs:
                    continue

                logits = model.re_forward(sequence_output, [pairs], re_labels=None)
                probs = torch.softmax(logits, dim=-1)
                confidences, pred_ids = torch.max(probs, dim=-1)

                for (drug_name, disease_name), pred_id, conf in zip(pair_meta, pred_ids.tolist(), confidences.tolist()):
                    relation = id2re[pred_id]
                    if relation == "NONE" or conf < CONFIDENCE_THRESHOLD:
                        continue
                    results.append({
                        "drug_name": drug_name,
                        "disease_name": disease_name,
                        "relation": relation,
                        "confidence": round(conf, 4),
                        "model": "SciBERT"
                    })

    seen = {}
    for r in results:
        key = (r["drug_name"], r["disease_name"], r["relation"])
        if key not in seen or r["confidence"] > seen[key]["confidence"]:
            seen[key] = r
    unique_results = list(seen.values())

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for r in unique_results:
            f.write(json.dumps(r) + "\n")

    print(f"Scanned {total_records} records, {len(unique_results)} unique confident predictions")
    print(f"Saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
