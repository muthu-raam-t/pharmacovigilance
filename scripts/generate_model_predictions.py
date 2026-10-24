"""Runs the trained joint model as a real extraction pipeline over the corpora:
the model's OWN NER output finds the drugs and diseases, then its RE head classifies
every drug-disease pair. Confident TREATS / CAUSES / ASSOCIATED predictions are
aggregated per (drug, disease, relation) and saved for the knowledge graph.

By default the backbone with the best end-to-end test F1 (results/model_comparison.csv)
is used. Each prediction keeps the split it came from, so predictions made on test
documents (never seen in training) can be told apart.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "models"))
import pv_config as cfg  # noqa: E402
from dataset import JointNERREDataset, make_collate_fn  # noqa: E402
from evaluation import bio_to_spans  # noqa: E402
from joint_model import JointNERREModel  # noqa: E402
from text_utils import normalize_name, token_char_spans, evidence_snippet  # noqa: E402

SPLITS = ["train", "val", "test"]
OUT_PATH = cfg.PROCESSED_DIR / "model_predicted_associations.jsonl"
MAX_EVIDENCE = 3


def pick_model(model_key):
    csv_path = cfg.RESULTS_DIR / "model_comparison.csv"
    if not csv_path.exists():
        sys.exit(f"{csv_path} not found - run evaluate_model.py (or run_all_experiments.sh) first.")
    rows = list(csv.DictReader(open(csv_path)))
    if model_key == "auto":
        row = max(rows, key=lambda r: float(r["e2e_f1"]))
    else:
        matches = [r for r in rows if r["model_key"] == model_key]
        if not matches:
            sys.exit(f"Model key '{model_key}' not in {csv_path}. Available: {[r['model_key'] for r in rows]}")
        row = matches[0]
    return row["model_key"], row["model_name"], row["backbone"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_key", default="auto", help="'auto' = best end-to-end F1, or e.g. scibert")
    parser.add_argument("--threshold", type=float, default=0.70)
    parser.add_argument("--entity_source", choices=["predicted", "gold"], default="predicted")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=256)
    args = parser.parse_args()

    key, name, backbone = pick_model(args.model_key)
    ckpt = cfg.CHECKPOINT_DIR / key / "model.pt"
    print(f"Using {name} ({backbone}) from {ckpt} | entities: {args.entity_source} | threshold {args.threshold}")

    label_maps = json.load(open(cfg.PROCESSED_DIR / "label_maps.json"))
    id2ner = {int(k): v for k, v in label_maps["ner_id2label"].items()}
    id2re = {int(k): v for k, v in label_maps["re_id2label"].items()}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        tokenizer = AutoTokenizer.from_pretrained(backbone, add_prefix_space=True)
    except TypeError:
        tokenizer = AutoTokenizer.from_pretrained(backbone)
    collate = make_collate_fn(tokenizer.pad_token_id or 0)
    model = JointNERREModel(backbone, len(id2ner), len(id2re)).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    agg, n_records, n_pairs = {}, 0, 0
    with torch.no_grad():
        for split in SPLITS:
            ds = JointNERREDataset(cfg.PROCESSED_DIR / f"ner_re_{split}.jsonl", tokenizer, label_maps,
                                   args.max_length, train=False)
            for start in range(0, len(ds), args.batch_size):
                idxs = list(range(start, min(start + args.batch_size, len(ds))))
                items = [ds[i] for i in idxs]
                batch = collate(items)
                seq = model.encode(batch["input_ids"].to(device), batch["attention_mask"].to(device))
                tags_batch, _ = model.ner_forward(seq, batch["word_first_subword_idx"].to(device),
                                                  batch["word_mask"].to(device))

                all_pairs, all_meta = [], []
                for bi, ridx in enumerate(idxs):
                    record = ds.records[ridx]
                    n_words = len(items[bi]["word_spans"])
                    if args.entity_source == "predicted":
                        spans = bio_to_spans([id2ner[t] for t in tags_batch[bi]][:n_words])
                    else:
                        spans = [(e["token_start"], e["token_end"], e["type"]) for e in record["entities"]
                                 if e["token_end"] <= n_words]
                    chems = [s for s in spans if s[2] == "Chemical"]
                    diss = [s for s in spans if s[2] == "Disease"]
                    ws = items[bi]["word_spans"]
                    pairs, meta = [], []
                    for c in chems:
                        for d in diss:
                            pairs.append((ws[c[0]][0], ws[c[1] - 1][1], ws[d[0]][0], ws[d[1] - 1][1]))
                            meta.append((ridx, c, d))
                    all_pairs.append(pairs)
                    all_meta.append(meta)
                    n_records += 1

                if not any(all_pairs):
                    continue
                probs = torch.softmax(model.re_forward(seq, all_pairs).float(), dim=-1)
                conf, pred = probs.max(dim=-1)
                flat_meta = [m for meta in all_meta for m in meta]
                n_pairs += len(flat_meta)

                for (ridx, c, d), p, cf in zip(flat_meta, pred.tolist(), conf.tolist()):
                    relation = id2re[p]
                    if relation == "NONE" or cf < args.threshold:
                        continue
                    record = ds.records[ridx]
                    text = record["text"]
                    cs = token_char_spans(text)
                    if c[1] - 1 >= len(cs) or d[1] - 1 >= len(cs):
                        continue
                    drug = text[cs[c[0]][0]:cs[c[1] - 1][1]]
                    disease = text[cs[d[0]][0]:cs[d[1] - 1][1]]
                    k = (normalize_name(drug), normalize_name(disease), relation)
                    if k[0] == k[1]:
                        continue
                    entry = agg.setdefault(k, {"confidences": [], "docs": set(), "evidence": []})
                    entry["confidences"].append(cf)
                    entry["docs"].add(record["doc_id"])
                    if len(entry["evidence"]) < MAX_EVIDENCE and record["doc_id"] not in {e["doc_id"] for e in entry["evidence"]}:
                        entry["evidence"].append({
                            "doc_id": record["doc_id"], "split": split, "source_dataset": record["source_dataset"],
                            "snippet": evidence_snippet(text, cs[c[0]][0], cs[d[0]][0]),
                            "drug_mention": drug, "disease_mention": disease, "confidence": round(cf, 4),
                        })
            print(f"  {split}: done", flush=True)

    counts = {}
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for (drug, disease, relation), v in agg.items():
            counts[relation] = counts.get(relation, 0) + 1
            f.write(json.dumps({
                "drug_name": drug, "disease_name": disease, "relation": relation,
                "confidence": round(max(v["confidences"]), 4),
                "mean_confidence": round(sum(v["confidences"]) / len(v["confidences"]), 4),
                "support": len(v["docs"]), "model": name, "model_key": key,
                "evidence": sorted(v["evidence"], key=lambda e: -e["confidence"]),
            }) + "\n")

    summary = {"model": name, "model_key": key, "entity_source": args.entity_source,
               "threshold": args.threshold, "records_scanned": n_records, "pairs_classified": n_pairs,
               "unique_predictions": len(agg), "by_relation": counts}
    cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(cfg.RESULTS_DIR / "model_predictions_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
