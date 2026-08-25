import argparse
import json
import os
import sys
import csv
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from seqeval.metrics import precision_score, recall_score, f1_score as seq_f1_score
from sklearn.metrics import precision_recall_fscore_support

sys.path.append("/workspace")
from models.joint_model import JointNERREModel
from models.dataset import JointNERREDataset, collate_fn


def evaluate(model, dataloader, device, id2ner, id2re):
    model.eval()
    true_ner_seqs, pred_ner_seqs = [], []
    true_re_labels, pred_re_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            word_first_subword_idx = batch["word_first_subword_idx"].to(device)
            word_mask = batch["word_mask"].to(device)
            word_labels = batch["word_labels"]

            sequence_output = model.encode(input_ids, attention_mask)

            pred_tags_batch, _ = model.ner_forward(sequence_output, word_first_subword_idx, word_mask, word_labels=None)

            for i, pred_seq in enumerate(pred_tags_batch):
                length = len(pred_seq)
                true_seq = word_labels[i, :length].tolist()
                true_ner_seqs.append([id2ner[str(t)] for t in true_seq])
                pred_ner_seqs.append([id2ner[str(p)] for p in pred_seq])

            pairs = batch["pairs"]
            pair_labels = batch["pair_labels"]
            has_pairs = any(len(p) > 0 for p in pairs)
            if has_pairs:
                logits = model.re_forward(sequence_output, pairs, re_labels=None)
                preds = torch.argmax(logits, dim=-1).cpu().tolist()

                flat_true = []
                for labels in pair_labels:
                    flat_true.extend(labels)

                pred_re_labels.extend(preds)
                true_re_labels.extend(flat_true)

    ner_precision = precision_score(true_ner_seqs, pred_ner_seqs)
    ner_recall = recall_score(true_ner_seqs, pred_ner_seqs)
    ner_f1 = seq_f1_score(true_ner_seqs, pred_ner_seqs)

    if true_re_labels:
        re_precision, re_recall, re_f1, _ = precision_recall_fscore_support(
            true_re_labels, pred_re_labels, average="weighted", zero_division=0
        )
    else:
        re_precision = re_recall = re_f1 = 0.0

    return {
        "ner_precision": ner_precision, "ner_recall": ner_recall, "ner_f1": ner_f1,
        "re_precision": re_precision, "re_recall": re_recall, "re_f1": re_f1
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model_name", required=True, help="Label used in the comparison CSV, e.g. BERT-base")
    parser.add_argument("--test_path", default="/workspace/data/processed/ner_re_test.jsonl")
    parser.add_argument("--label_maps", default="/workspace/data/processed/label_maps.json")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--max_pairs", type=int, default=15)
    parser.add_argument("--output_csv", default="/workspace/data/processed/model_comparison.csv")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    with open(args.label_maps) as f:
        label_maps = json.load(f)
    id2ner = label_maps["ner_id2label"]
    id2re = label_maps["re_id2label"]
    num_ner_labels = len(label_maps["ner_label2id"])
    num_re_labels = len(label_maps["re_label2id"])

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.backbone, add_prefix_space=True)
    except TypeError:
        tokenizer = AutoTokenizer.from_pretrained(args.backbone)

    test_dataset = JointNERREDataset(args.test_path, tokenizer, label_maps, args.max_length, args.max_pairs)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    model = JointNERREModel(args.backbone, num_ner_labels, num_re_labels).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    print(f"Loaded checkpoint: {args.checkpoint}")

    results = evaluate(model, test_loader, device, id2ner, id2re)

    print(f"\n=== {args.model_name} — Test Set Results ===")
    print(f"NER  — Precision: {results['ner_precision']:.3f}  Recall: {results['ner_recall']:.3f}  F1: {results['ner_f1']:.3f}")
    print(f"RE   — Precision: {results['re_precision']:.3f}  Recall: {results['re_recall']:.3f}  F1: {results['re_f1']:.3f}")

    file_exists = os.path.isfile(args.output_csv)
    with open(args.output_csv, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["model_name", "ner_precision", "ner_recall", "ner_f1", "re_precision", "re_recall", "re_f1"])
        writer.writerow([
            args.model_name,
            f"{results['ner_precision']:.4f}", f"{results['ner_recall']:.4f}", f"{results['ner_f1']:.4f}",
            f"{results['re_precision']:.4f}", f"{results['re_recall']:.4f}", f"{results['re_f1']:.4f}"
        ])
    print(f"\nAppended to {args.output_csv}")


if __name__ == "__main__":
    main()

