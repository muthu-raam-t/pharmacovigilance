"""Trains the joint NER + RE model for one backbone.

Improvements over the first version: fixed random seed, separate learning rates for
encoder and heads, weight decay, linear warm-up / decay schedule, gradient clipping,
mixed precision on GPU, dynamic padding, validation metrics after every epoch,
best-epoch checkpointing and early stopping.
"""
import argparse
import json
import os
import random
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "models"))
from joint_model import JointNERREModel  # noqa: E402
from dataset import JointNERREDataset, make_collate_fn  # noqa: E402
from evaluation import run_evaluation  # noqa: E402
import pv_config as cfg  # noqa: E402


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_tokenizer(backbone):
    try:
        return AutoTokenizer.from_pretrained(backbone, add_prefix_space=True)
    except TypeError:
        return AutoTokenizer.from_pretrained(backbone)


def build_optimizer(model, lr, head_lr, weight_decay):
    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
    head_ids = {id(p) for p in model.head_parameters()}
    groups = {"enc_decay": [], "enc_nodecay": [], "head_decay": [], "head_nodecay": []}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        part = "head" if id(p) in head_ids else "enc"
        decay = "nodecay" if any(nd in name for nd in no_decay) else "decay"
        groups[f"{part}_{decay}"].append(p)
    return torch.optim.AdamW([
        {"params": groups["enc_decay"], "lr": lr, "weight_decay": weight_decay},
        {"params": groups["enc_nodecay"], "lr": lr, "weight_decay": 0.0},
        {"params": groups["head_decay"], "lr": head_lr, "weight_decay": weight_decay},
        {"params": groups["head_nodecay"], "lr": head_lr, "weight_decay": 0.0},
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", required=True, help="Hugging Face model id or local path")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name", default=None, help="Display name, e.g. SciBERT")
    parser.add_argument("--train_path", default=str(cfg.PROCESSED_DIR / "ner_re_train.jsonl"))
    parser.add_argument("--val_path", default=str(cfg.PROCESSED_DIR / "ner_re_val.jsonl"))
    parser.add_argument("--label_maps", default=str(cfg.PROCESSED_DIR / "label_maps.json"))
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--head_lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--lambda_re", type=float, default=1.0)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--neg_ratio", type=float, default=2.0)
    parser.add_argument("--max_train_pairs", type=int, default=64)
    parser.add_argument("--patience", type=int, default=2, help="Early-stopping patience in epochs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--no_amp", action="store_true", help="Disable mixed precision")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    print(f"Using device: {device} | mixed precision: {use_amp}")

    with open(args.label_maps) as f:
        label_maps = json.load(f)
    num_ner, num_re = len(label_maps["ner_label2id"]), len(label_maps["re_label2id"])
    print(f"NER labels: {num_ner}, RE labels: {num_re} {list(label_maps['re_label2id'])}")

    tokenizer = load_tokenizer(args.backbone)
    collate = make_collate_fn(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0)

    train_ds = JointNERREDataset(args.train_path, tokenizer, label_maps, args.max_length, train=True,
                                 neg_ratio=args.neg_ratio, max_train_pairs=args.max_train_pairs, seed=args.seed)
    val_ds = JointNERREDataset(args.val_path, tokenizer, label_maps, args.max_length, train=False)
    g = torch.Generator()
    g.manual_seed(args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate,
                              num_workers=args.num_workers, generator=g,
                              worker_init_fn=partial(lambda wid, s: random.seed(s + wid), s=args.seed))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate,
                            num_workers=args.num_workers)

    model = JointNERREModel(args.backbone, num_ner, num_re).to(device)
    optimizer = build_optimizer(model, args.lr, args.head_lr, args.weight_decay)
    total_steps = max(1, len(train_loader) * args.epochs)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(args.warmup_ratio * total_steps), total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_path = os.path.join(args.output_dir, "model.pt")
    best_score, best_epoch, bad_epochs, history = -1.0, 0, 0, []

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, total_loss, n_steps = time.time(), 0.0, 0
        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            wfs = batch["word_first_subword_idx"].to(device)
            word_mask = batch["word_mask"].to(device)
            word_labels = batch["word_labels"].to(device)

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                seq_out = model.encode(input_ids, attention_mask)
                ner_loss, _ = model.ner_forward(seq_out, wfs, word_mask, word_labels)
                re_loss, _ = model.re_forward(seq_out, batch["pairs"], batch["pair_labels"])
            loss = ner_loss + (args.lambda_re * re_loss if re_loss is not None else 0.0)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item()
            n_steps += 1
            if step % 50 == 0:
                re_val = re_loss.item() if re_loss is not None else 0.0
                print(f"Epoch {epoch} step {step}/{len(train_loader)} - loss {loss.item():.4f} "
                      f"(NER {ner_loss.item():.4f}, RE {re_val:.4f})", flush=True)

        val = run_evaluation(model, val_loader, device, label_maps, use_amp=use_amp, end_to_end=False)
        score = (val["ner_f1"] + val["re_micro_f1"]) / 2
        entry = {"epoch": epoch, "train_loss": total_loss / max(n_steps, 1),
                 "val_ner_f1": val["ner_f1"], "val_re_micro_f1": val["re_micro_f1"],
                 "val_re_macro_f1": val["re_macro_f1"], "val_score": score,
                 "epoch_minutes": (time.time() - t0) / 60}
        history.append(entry)
        print(f"\n=== Epoch {epoch} === train loss {entry['train_loss']:.4f} | val NER F1 {val['ner_f1']:.4f} | "
              f"val RE micro-F1 {val['re_micro_f1']:.4f} | val RE macro-F1 {val['re_macro_f1']:.4f} | "
              f"{entry['epoch_minutes']:.1f} min\n", flush=True)

        if score > best_score:
            best_score, best_epoch, bad_epochs = score, epoch, 0
            torch.save(model.state_dict(), ckpt_path)
            print(f"New best model (epoch {epoch}) saved to {ckpt_path}", flush=True)
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stopping: no improvement for {args.patience} epochs")
                break

    summary = {"model_name": args.model_name or args.backbone, "backbone": args.backbone,
               "best_epoch": best_epoch, "best_val_score": best_score, "history": history,
               "hyperparameters": vars(args)}
    with open(os.path.join(args.output_dir, "training_history.json"), "w") as f:
        json.dump(summary, f, indent=2)
    cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    key = Path(args.output_dir).name
    with open(cfg.RESULTS_DIR / f"training_history_{key}.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Best epoch: {best_epoch} (val score {best_score:.4f}). History saved.")


if __name__ == "__main__":
    main()
