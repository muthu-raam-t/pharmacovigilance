"""Evaluation of the joint NER + RE model.

Reported metrics
  NER  : entity-level precision / recall / F1 (seqeval, micro over types), token accuracy, per-type report
  RE   : relation classification on gold entity pairs
         - micro P / R / F1 over the positive classes (NONE excluded)  <- headline RE metric
         - macro F1 over the positive classes, accuracy, support-weighted F1, per-class report
  E2E  : end-to-end relation extraction. Entities come from the model's own NER output;
         a relation counts as correct only if both entity spans, both types and the
         relation label all match the gold annotation. Gold relations whose entities were
         cut off by truncation are counted as misses.
"""
from collections import Counter

import torch
from seqeval.metrics import (precision_score, recall_score, f1_score as seq_f1_score,
                             classification_report as seq_report)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

from pair_utils import type_pair_key


def bio_to_spans(tags):
    """['B-X','I-X','O',...] -> list of (start, end, type)."""
    spans, start, cur = [], None, None
    for i, tag in enumerate(tags + ["O"]):
        if tag.startswith("B-") or tag == "O" or (tag.startswith("I-") and tag[2:] != cur):
            if cur is not None:
                spans.append((start, i, cur))
            start, cur = (i, tag[2:]) if tag != "O" else (None, None)
        # I- continuing the current entity: nothing to do
    return spans


def _prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def run_evaluation(model, dataloader, device, label_maps, use_amp=False, end_to_end=True):
    model.eval()
    id2ner = {int(k): v for k, v in label_maps["ner_id2label"].items()}
    id2re = {int(k): v for k, v in label_maps["re_id2label"].items()}
    none_id = label_maps["re_label2id"]["NONE"]
    positive_ids = sorted(i for i in id2re if i != none_id)
    allowed = {tuple(p) for p in label_maps.get("re_type_pairs", [])}

    true_ner, pred_ner = [], []
    true_re, pred_re = [], []
    e2e_tp, e2e_fp, e2e_fn = Counter(), Counter(), Counter()

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            wfs = batch["word_first_subword_idx"].to(device)
            word_mask = batch["word_mask"].to(device)
            word_labels = batch["word_labels"]

            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=use_amp and device.type == "cuda"):
                seq_out = model.encode(input_ids, attention_mask)
                pred_tags_batch, _ = model.ner_forward(seq_out, wfs, word_mask, word_labels=None)

                # --- NER ---
                pred_tag_strs = []
                for i, pred_seq in enumerate(pred_tags_batch):
                    n = len(pred_seq)
                    true_ner.append([id2ner[t] for t in word_labels[i, :n].tolist()])
                    tags = [id2ner[p] for p in pred_seq]
                    pred_ner.append(tags)
                    pred_tag_strs.append(tags)

                # --- RE on gold entity pairs ---
                if any(len(p) for p in batch["pairs"]):
                    logits = model.re_forward(seq_out, batch["pairs"], re_labels=None)
                    pred_re.extend(torch.argmax(logits, dim=-1).cpu().tolist())
                    for labels in batch["pair_labels"]:
                        true_re.extend(labels)

                # --- End-to-end RE on predicted entities ---
                if end_to_end:
                    e2e_pairs, e2e_meta = [], []
                    for i, tags in enumerate(pred_tag_strs):
                        n_words = len(batch["word_spans"][i])
                        spans = [s for s in bio_to_spans(tags[:n_words])]
                        toks = batch["tokens"][i]
                        pairs_i, meta_i = [], []
                        for a in range(len(spans)):
                            for b in range(a + 1, len(spans)):
                                sa, sb = spans[a], spans[b]
                                if allowed and type_pair_key(sa[2], sb[2]) not in allowed:
                                    continue
                                if [t.lower() for t in toks[sa[0]:sa[1]]] == [t.lower() for t in toks[sb[0]:sb[1]]]:
                                    continue
                                ws = batch["word_spans"][i]
                                pairs_i.append((ws[sa[0]][0], ws[sa[1] - 1][1], ws[sb[0]][0], ws[sb[1] - 1][1]))
                                meta_i.append((sa, sb))
                        e2e_pairs.append(pairs_i)
                        e2e_meta.append(meta_i)

                    preds_flat = []
                    if any(len(p) for p in e2e_pairs):
                        logits = model.re_forward(seq_out, e2e_pairs, re_labels=None)
                        preds_flat = torch.argmax(logits, dim=-1).cpu().tolist()

                    k = 0
                    for i, meta_i in enumerate(e2e_meta):
                        predicted = set()
                        for sa, sb in meta_i:
                            label = id2re[preds_flat[k]]
                            k += 1
                            if label != "NONE":
                                first, second = sorted((sa, sb))
                                predicted.add((first, second, label))
                        gold = batch["gold_relations"][i]
                        for rel in predicted:
                            (e2e_tp if rel in gold else e2e_fp)[rel[2]] += 1
                        for rel in gold - predicted:
                            e2e_fn[rel[2]] += 1

    model.train()

    metrics = {
        "ner_accuracy": accuracy_score([t for s in true_ner for t in s], [t for s in pred_ner for t in s]),
        "ner_precision": precision_score(true_ner, pred_ner, zero_division=0),
        "ner_recall": recall_score(true_ner, pred_ner, zero_division=0),
        "ner_f1": seq_f1_score(true_ner, pred_ner, zero_division=0),
        "ner_per_type": seq_report(true_ner, pred_ner, output_dict=True, zero_division=0),
    }
    metrics["ner_per_type"] = {k: {m: float(v) for m, v in d.items()} for k, d in metrics["ner_per_type"].items()}

    if true_re:
        p, r, f, _ = precision_recall_fscore_support(true_re, pred_re, labels=positive_ids,
                                                     average="micro", zero_division=0)
        _, _, macro_f, _ = precision_recall_fscore_support(true_re, pred_re, labels=positive_ids,
                                                           average="macro", zero_division=0)
        _, _, weighted_f, _ = precision_recall_fscore_support(true_re, pred_re, average="weighted",
                                                              zero_division=0)
        names = [id2re[i] for i in sorted(id2re)]
        report = classification_report(true_re, pred_re, labels=sorted(id2re), target_names=names,
                                       output_dict=True, zero_division=0)
        metrics.update({
            "re_micro_precision": p, "re_micro_recall": r, "re_micro_f1": f,
            "re_macro_f1": macro_f, "re_weighted_f1": weighted_f,
            "re_accuracy": accuracy_score(true_re, pred_re),
            "re_num_pairs": len(true_re),
            "re_per_class": {k: {m: float(v) for m, v in d.items()} for k, d in report.items()
                             if isinstance(d, dict)},
        })
    else:
        metrics.update({"re_micro_precision": 0.0, "re_micro_recall": 0.0, "re_micro_f1": 0.0,
                        "re_macro_f1": 0.0, "re_weighted_f1": 0.0, "re_accuracy": 0.0,
                        "re_num_pairs": 0, "re_per_class": {}})

    if end_to_end:
        tp, fp, fn = sum(e2e_tp.values()), sum(e2e_fp.values()), sum(e2e_fn.values())
        p, r, f = _prf(tp, fp, fn)
        per_class = {}
        for label in sorted(set(e2e_tp) | set(e2e_fp) | set(e2e_fn)):
            cp, cr, cf = _prf(e2e_tp[label], e2e_fp[label], e2e_fn[label])
            per_class[label] = {"precision": cp, "recall": cr, "f1-score": cf,
                                "support": e2e_tp[label] + e2e_fn[label]}
        metrics.update({"e2e_precision": p, "e2e_recall": r, "e2e_f1": f,
                        "e2e_gold_relations": tp + fn, "e2e_per_class": per_class})

    for k, v in list(metrics.items()):
        if hasattr(v, "item"):
            metrics[k] = float(v)
    return metrics
