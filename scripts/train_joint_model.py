import argparse
import json
import os
import sys
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

sys.path.append("/workspace")
from models.joint_model import JointNERREModel
from models.dataset import JointNERREDataset, collate_fn


def evaluate(model, dataloader, device):
    model.eval()
    total_ner_loss, total_re_loss, n_batches = 0.0, 0.0, 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            word_first_subword_idx = batch["word_first_subword_idx"].to(device)
            word_mask = batch["word_mask"].to(device)
            word_labels = batch["word_labels"].to(device)

            sequence_output = model.encode(input_ids, attention_mask)
            ner_loss, _ = model.ner_forward(sequence_output, word_first_subword_idx, word_mask, word_labels)

            pairs = batch["pairs"]
            pair_labels = batch["pair_labels"]
            has_pairs = any(len(p) > 0 for p in pairs)
            if has_pairs:
                re_loss, _ = model.re_forward(sequence_output, pairs, pair_labels)
            else:
                re_loss = torch.tensor(0.0, device=device)

            total_ner_loss += ner_loss.item()
            total_re_loss += re_loss.item() if hasattr(re_loss, "item") else 0.0
            n_batches += 1

    model.train()
    return total_ner_loss / max(n_batches, 1), total_re_loss / max(n_batches, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--train_path", default="/workspace/data/processed/ner_re_train.jsonl")
    parser.add_argument("--val_path", default="/workspace/data/processed/ner_re_val.jsonl")
    parser.add_argument("--label_maps", default="/workspace/data/processed/label_maps.json")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--lambda_re", type=float, default=1.0)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--max_pairs", type=int, default=15)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    with open(args.label_maps) as f:
        label_maps = json.load(f)
    num_ner_labels = len(label_maps["ner_label2id"])
    num_re_labels = len(label_maps["re_label2id"])
    print(f"NER labels: {num_ner_labels}, RE labels: {num_re_labels}")

    tokenizer = AutoTokenizer.from_pretrained(args.backbone)

    train_dataset = JointNERREDataset(args.train_path, tokenizer, label_maps, args.max_length, args.max_pairs)
    val_dataset = JointNERREDataset(args.val_path, tokenizer, label_maps, args.max_length, args.max_pairs)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    model = JointNERREModel(args.backbone, num_ner_labels, num_re_labels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            word_first_subword_idx = batch["word_first_subword_idx"].to(device)
            word_mask = batch["word_mask"].to(device)
            word_labels = batch["word_labels"].to(device)

            sequence_output = model.encode(input_ids, attention_mask)
            ner_loss, _ = model.ner_forward(sequence_output, word_first_subword_idx, word_mask, word_labels)

            pairs = batch["pairs"]
            pair_labels = batch["pair_labels"]
            has_pairs = any(len(p) > 0 for p in pairs)
            if has_pairs:
                re_loss, _ = model.re_forward(sequence_output, pairs, pair_labels)
            else:
                re_loss = torch.tensor(0.0, device=device, requires_grad=True)

            loss = ner_loss + args.lambda_re * re_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if step % 20 == 0:
                print(f"Epoch {epoch+1} step {step}/{len(train_loader)} - loss: {loss.item():.4f} (NER: {ner_loss.item():.4f}, RE: {re_loss.item() if hasattr(re_loss,'item') else 0:.4f})")

        avg_train_loss = total_loss / len(train_loader)
        val_ner_loss, val_re_loss = evaluate(model, val_loader, device)
        print(f"\n=== Epoch {epoch+1} summary ===")
        print(f"Train loss: {avg_train_loss:.4f}")
        print(f"Val NER loss: {val_ner_loss:.4f}, Val RE loss: {val_re_loss:.4f}\n")

    checkpoint_path = os.path.join(args.output_dir, "model.pt")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()
