import json
import random

import torch
from torch.utils.data import Dataset

from pair_utils import candidate_pairs, gold_relation_set, NONE_LABEL


class JointNERREDataset(Dataset):
    """
    Training mode : all positive pairs + randomly re-sampled negative pairs (neg_ratio x positives)
    Evaluation mode: every candidate pair of visible entities (deterministic)
    Sequences are NOT padded here; collate_fn pads each batch to its own longest sequence.
    """

    def __init__(self, jsonl_path, tokenizer, label_maps, max_length=256, train=True,
                 neg_ratio=2.0, max_train_pairs=64, max_eval_pairs=400, seed=42):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.train = train
        self.neg_ratio = neg_ratio
        self.max_train_pairs = max_train_pairs
        self.max_eval_pairs = max_eval_pairs
        self.ner_label2id = label_maps["ner_label2id"]
        self.re_label2id = label_maps["re_label2id"]
        self.type_pairs = label_maps.get("re_type_pairs")
        self.rng = random.Random(seed)

        with open(jsonl_path, encoding="utf-8") as f:
            self.records = [json.loads(line) for line in f if line.strip()]

        if self.type_pairs is None:
            types = set()
            for r in self.records:
                t = {e["id"]: e["type"] for e in r["entities"]}
                for rel in r["relations"]:
                    if rel["arg1"] in t and rel["arg2"] in t:
                        types.add(tuple(sorted((t[rel["arg1"]], t[rel["arg2"]]))))
            self.type_pairs = [list(p) for p in types]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        tokens, ner_tags = r["tokens"], r["ner_tags"]

        encoding = self.tokenizer(tokens, is_split_into_words=True, truncation=True,
                                  max_length=self.max_length)
        word_ids = encoding.word_ids()

        first_sub, last_sub = {}, {}
        for sub_idx, w_id in enumerate(word_ids):
            if w_id is None:
                continue
            first_sub.setdefault(w_id, sub_idx)
            last_sub[w_id] = sub_idx + 1

        # Truncation only removes words from the end, so visible words are 0..n_visible-1
        n_visible = 0
        while n_visible in first_sub:
            n_visible += 1

        word_first = [first_sub[w] for w in range(n_visible)]
        word_spans = [(first_sub[w], last_sub[w]) for w in range(n_visible)]
        o_id = self.ner_label2id["O"]
        word_labels = [self.ner_label2id.get(ner_tags[w], o_id) if w < len(ner_tags) else o_id
                       for w in range(n_visible)]

        cands = candidate_pairs(r, self.type_pairs, max_word=n_visible)
        if self.train:
            pos = [c for c in cands if c[2] != NONE_LABEL]
            neg = [c for c in cands if c[2] == NONE_LABEL]
            n_neg = min(len(neg), max(1, int(round(self.neg_ratio * max(len(pos), 1)))))
            chosen = pos[:self.max_train_pairs] + self.rng.sample(neg, n_neg)
            chosen = chosen[:self.max_train_pairs]
        else:
            chosen = cands[:self.max_eval_pairs]

        pairs, pair_labels = [], []
        for a, b, label in chosen:
            pairs.append((word_spans[a["token_start"]][0], word_spans[a["token_end"] - 1][1],
                          word_spans[b["token_start"]][0], word_spans[b["token_end"] - 1][1]))
            pair_labels.append(self.re_label2id.get(label, self.re_label2id[NONE_LABEL]))

        return {
            "input_ids": torch.tensor(encoding["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(encoding["attention_mask"], dtype=torch.long),
            "word_first_subword_idx": torch.tensor(word_first, dtype=torch.long),
            "word_labels": torch.tensor(word_labels, dtype=torch.long),
            "pairs": pairs,
            "pair_labels": pair_labels,
            # extra information used for end-to-end evaluation
            "tokens": tokens[:n_visible],
            "word_spans": word_spans,
            "gold_relations": gold_relation_set(r),
        }


def make_collate_fn(pad_token_id):
    def collate_fn(batch):
        bsz = len(batch)
        max_len = max(b["input_ids"].shape[0] for b in batch)
        max_words = max(max(b["word_first_subword_idx"].shape[0] for b in batch), 1)

        input_ids = torch.full((bsz, max_len), pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros(bsz, max_len, dtype=torch.long)
        word_first_subword_idx = torch.zeros(bsz, max_words, dtype=torch.long)
        word_mask = torch.zeros(bsz, max_words, dtype=torch.bool)
        word_labels = torch.zeros(bsz, max_words, dtype=torch.long)

        for i, b in enumerate(batch):
            n = b["input_ids"].shape[0]
            input_ids[i, :n] = b["input_ids"]
            attention_mask[i, :n] = b["attention_mask"]
            w = b["word_first_subword_idx"].shape[0]
            word_first_subword_idx[i, :w] = b["word_first_subword_idx"]
            word_mask[i, :w] = True
            word_labels[i, :w] = b["word_labels"]
            if w == 0:  # CRF needs the first time step switched on
                word_mask[i, 0] = True

        out = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "word_first_subword_idx": word_first_subword_idx,
            "word_mask": word_mask,
            "word_labels": word_labels,
        }
        for key in ("pairs", "pair_labels", "tokens", "word_spans", "gold_relations"):
            out[key] = [b[key] for b in batch]
        return out

    return collate_fn
