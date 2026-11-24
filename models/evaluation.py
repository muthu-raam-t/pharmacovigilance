"""Evaluation of the joint NER + RE model.

Outputs the exact 4 metrics shown in the presentation slides:
  * NER: Accuracy, Precision, Recall, F1 (seqeval entity-level)
  * RE : Accuracy, Precision, Recall, F1 (support-weighted over candidate entity pairs)
  * E2E: Precision, Recall, F1 (with optional validation-tuned confidence threshold)
"""
from collections import Counter

import torch
from seqeval.metrics import (precision_score, recall_score, f1_score as seq_f1_score,
                             classification_report as seq_report)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

from dataset import build_entity_type2id
from pair_utils import type_pair_key, distance_bucket


def bio_to_spans(tags):
    """['B-X','I-X','O',...] -> list of (start, end, type)."""
    spans, start, cur = [], None, None
    for i, tag in enumerate(tags + ["O"]):
        if tag.startswith("B-") or tag == "O" or (tag.startswith("I-") and tag[2:] != cur):
            if cur is not None:
                spans.append((start, i, cur))
            start, cur = (i, tag[2:]) if tag != "O" else (None, None)
    return spans


def _prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def _score_e2e(e2e_records, id2re, threshold=0.0):
    tp, fp, fn = Counter(), Counter(), Counter()
    for meta_i, probs_i, gold in e2e_records:
        predicted = set()
        for (sa, sb), (pred_id, conf) in zip(meta_i, probs_i):
            label = id2re[pred_id]
            if label != "NONE" and conf >= threshold:
                first, second = sorted((sa, sb))
                predicted.add((first, second, label))
        for rel in predicted:
            (tp if rel in gold else fp)[rel[2]] += 1
        for rel in gold - predicted:
            fn[rel[2]] += 1
    p, r, f = _prf(sum(tp.values()), sum(fp.values()), sum(fn.values()))
    return p, r, f, sum(tp.values()) + sum(fn.values())


def run_evaluation(model, dataloader, device, label_maps, use_amp=False, end_to_end=True,
                   e2e_threshold=0.0, tune_e2e_threshold=False):
    model.eval()
    id2ner = {int(k): v for k, v in label_maps["ner_id2label"].items()}
    id2re = {int(k): v for k, v in label_maps["re_id2label"].items()}
    type2id = build_entity_type2id(label_maps["ner_label2id"])
    allowed = {tuple(p) for p in label_maps.get("re_type_pairs", [])}

    true_ner, pred_ner = [], []
    true_re, pred_re = [], []
    e2e_records = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            wfs = batch["word_first_subword_idx"].to(device)
            word_spans_sub = batch.get("word_spans_sub")
            if word_spans_sub is not None:
                word_spans_sub = word_spans_sub.to(device)
            word_mask = batch["word_mask"].to(device)
            word_labels = batch["word_labels"]

            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=use_amp and device.type == "cuda"):
                seq_out = model.encode(input_ids, attention_mask)
                pred_tags_batch, _ = model.ner_forward(seq_out, wfs, word_mask, word_labels=None,
                                                       word_spans_sub=word_spans_sub)

                pred_tag_strs = []
                for i, pred_seq in enumerate(pred_tags_batch):
                    n = len(pred_seq)
                    true_ner.append([id2ner[t] for t in word_labels[i, :n].tolist()])
                    tags = [id2ner[p] for p in pred_seq]
                    pred_ner.append(tags)
                    pred_tag_strs.append(tags)

                if any(len(p) for p in batch["pairs"]):
                    logits = model.re_forward(
                        seq_out, batch["pairs"], re_labels=None,
                        pair_types=batch.get("pair_types"), pair_dists=batch.get("pair_dists"),
                    )
                    pred_re.extend(torch.argmax(logits, dim=-1).cpu().tolist())
                    for labels in batch["pair_labels"]:
                        true_re.extend(labels)

                if end_to_end:
                    e2e_pairs, e2e_types, e2e_dists, e2e_meta = [], [], [], []
                    for i, tags in enumerate(pred_tag_strs):
                        n_words = len(batch["word_spans"][i])
                        spans = [s for s in bio_to_spans(tags[:n_words])]
                        toks = batch["tokens"][i]
                        pairs_i, types_i, dists_i, meta_i = [], [], [], []
                        for a in range(len(spans)):
                            for b in range(a + 1, len(spans)):
                                sa, sb = spans[a], spans[b]
                                if allowed and type_pair_key(sa[2], sb[2]) not in allowed:
                                    continue
                                if [t.lower() for t in toks[sa[0]:sa[1]]] == [t.lower() for t in toks[sb[0]:sb[1]]]:
                                    continue
                                dist = max(0, sb[0] - sa[1])
                                if dist > 60:
                                    continue
                                ws = batch["word_spans"][i]
                                pairs_i.append((ws[sa[0]][0], ws[sa[1] - 1][1], ws[sb[0]][0], ws[sb[1] - 1][1]))
                                types_i.append((type2id.get(sa[2], 0), type2id.get(sb[2], 0)))
                                dists_i.append(distance_bucket(dist))
                                meta_i.append((sa, sb))
                        e2e_pairs.append(pairs_i)
                        e2e_types.append(types_i)
                        e2e_dists.append(dists_i)
                        e2e_meta.append(meta_i)

                    preds_flat, confs_flat = [], []
                    if any(len(p) for p in e2e_pairs):
                        logits = model.re_forward(
                            seq_out, e2e_pairs, re_labels=None,
                            pair_types=e2e_types, pair_dists=e2e_dists,
                        )
                        probs = torch.softmax(logits.float(), dim=-1)
                        confs, preds = probs.max(dim=-1)
                        preds_flat = preds.cpu().tolist()
                        confs_flat = confs.cpu().tolist()

                    k = 0
                    for i, meta_i in enumerate(e2e_meta):
                        probs_i = []
                        for _ in meta_i:
                            probs_i.append((preds_flat[k], confs_flat[k]))
                            k += 1
                        e2e_records.append((meta_i, probs_i, batch["gold_relations"][i]))

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
        w_p, w_r, w_f, _ = precision_recall_fscore_support(true_re, pred_re, average="weighted", zero_division=0)
        re_acc = accuracy_score(true_re, pred_re)
        names = [id2re[i] for i in sorted(id2re)]
        report = classification_report(true_re, pred_re, labels=sorted(id2re), target_names=names,
                                       output_dict=True, zero_division=0)
        metrics.update({
            "re_accuracy": re_acc,
            "re_precision": w_p,
            "re_recall": w_r,
            "re_f1": w_f,
            "re_num_pairs": len(true_re),
            "re_per_class": {k: {m: float(v) for m, v in d.items()} for k, d in report.items()
                             if isinstance(d, dict)},
        })
    else:
        metrics.update({"re_accuracy": 0.0, "re_precision": 0.0, "re_recall": 0.0, "re_f1": 0.0,
                        "re_num_pairs": 0, "re_per_class": {}})

    if end_to_end:
        best_th = e2e_threshold
        if tune_e2e_threshold:
            best_f = -1.0
            for th in (0.0, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80):
                _, _, f_th, _ = _score_e2e(e2e_records, id2re, threshold=th)
                if f_th > best_f:
                    best_f, best_th = f_th, th
        p, r, f, total_gold = _score_e2e(e2e_records, id2re, threshold=best_th)
        metrics.update({
            "e2e_precision": p, "e2e_recall": r, "e2e_f1": f,
            "e2e_gold_relations": total_gold, "e2e_threshold": best_th,
        })

    for k, v in list(metrics.items()):
        if hasattr(v, "item"):
            metrics[k] = float(v)
    return metrics