"""Evaluates a trained checkpoint on the test split (NER, RE on gold pairs, end-to-end RE).

Writes results/metrics_<key>.json and updates the row for this model in
results/model_comparison.csv (re-running replaces the row instead of duplicating it).
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "models"))
from joint_model import JointNERREModel  # noqa: E402
from dataset import JointNERREDataset, make_collate_fn  # noqa: E402
from evaluation import run_evaluation  # noqa: E402
import pv_config as cfg  # noqa: E402

CSV_COLUMNS = ["model_name", "model_key", "backbone",
               "ner_accuracy", "ner_precision", "ner_recall", "ner_f1",
               "re_micro_precision", "re_micro_recall", "re_micro_f1", "re_macro_f1",
               "re_accuracy", "re_weighted_f1",
               "e2e_precision", "e2e_recall", "e2e_f1"]


def upsert_csv(path, row):
    rows = []
    if path.exists():
        with open(path, newline="") as f:
            rows = [r for r in csv.DictReader(f) if r.get("model_name") != row["model_name"]]
    rows.append(row)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model_name", required=True, help="Display name, e.g. SciBERT")
    parser.add_argument("--model_key", default=None, help="Short key, e.g. scibert (defaults to checkpoint folder)")
    parser.add_argument("--test_path", default=str(cfg.PROCESSED_DIR / "ner_re_test.jsonl"))
    parser.add_argument("--label_maps", default=str(cfg.PROCESSED_DIR / "label_maps.json"))
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--output_csv", default=str(cfg.RESULTS_DIR / "model_comparison.csv"))
    parser.add_argument("--no_amp", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    key = args.model_key or Path(args.checkpoint).parent.name

    with open(args.label_maps) as f:
        label_maps = json.load(f)
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.backbone, add_prefix_space=True)
    except TypeError:
        tokenizer = AutoTokenizer.from_pretrained(args.backbone)

    test_ds = JointNERREDataset(args.test_path, tokenizer, label_maps, args.max_length, train=False)
    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                        collate_fn=make_collate_fn(tokenizer.pad_token_id or 0))

    model = JointNERREModel(args.backbone, len(label_maps["ner_label2id"]), len(label_maps["re_label2id"])).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    print(f"Loaded checkpoint: {args.checkpoint}")

    m = run_evaluation(model, loader, device, label_maps, use_amp=use_amp, end_to_end=True)

    print(f"\n=== {args.model_name} - test set ({len(test_ds)} records) ===")
    print(f"NER  Acc {m['ner_accuracy']:.4f}  P {m['ner_precision']:.4f}  R {m['ner_recall']:.4f}  F1 {m['ner_f1']:.4f}")
    print(f"RE   (gold entities, {m['re_num_pairs']} pairs)  micro-P {m['re_micro_precision']:.4f}  "
          f"micro-R {m['re_micro_recall']:.4f}  micro-F1 {m['re_micro_f1']:.4f}  macro-F1 {m['re_macro_f1']:.4f}  "
          f"Acc {m['re_accuracy']:.4f}")
    print(f"E2E  (predicted entities, {m['e2e_gold_relations']} gold relations)  P {m['e2e_precision']:.4f}  "
          f"R {m['e2e_recall']:.4f}  F1 {m['e2e_f1']:.4f}")
    print("RE per class:", {k: round(v["f1-score"], 4) for k, v in m["re_per_class"].items()})

    cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = {"model_name": args.model_name, "model_key": key, "backbone": args.backbone,
           "test_records": len(test_ds), **m}
    with open(cfg.RESULTS_DIR / f"metrics_{key}.json", "w") as f:
        json.dump(out, f, indent=2)

    row = {"model_name": args.model_name, "model_key": key, "backbone": args.backbone}
    row.update({c: f"{m[c]:.4f}" for c in CSV_COLUMNS[3:]})
    upsert_csv(Path(args.output_csv), row)
    print(f"\nSaved results/metrics_{key}.json and updated {args.output_csv}")


if __name__ == "__main__":
    main()
