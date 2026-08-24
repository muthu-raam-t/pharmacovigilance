import json
import torch
from torch.utils.data import Dataset


class JointNERREDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, label_maps, max_length=256, max_pairs_per_example=15):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_pairs = max_pairs_per_example
        self.ner_label2id = label_maps["ner_label2id"]
        self.re_label2id = label_maps["re_label2id"]

        with open(jsonl_path, encoding="utf-8") as f:
            self.records = [json.loads(line) for line in f]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        tokens = r["tokens"]
        ner_tags = r["ner_tags"]

        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        word_ids = encoding.word_ids(batch_index=0)

        # Find the first subword index for each original word
        word_first_subword = {}
        for subword_idx, w_id in enumerate(word_ids):
            if w_id is not None and w_id not in word_first_subword:
                word_first_subword[w_id] = subword_idx

        num_words = min(len(tokens), self.max_length)
        first_subword_idx = []
        word_label_ids = []
        for w in range(num_words):
            if w in word_first_subword:
                first_subword_idx.append(word_first_subword[w])
                tag = ner_tags[w] if w < len(ner_tags) else "O"
                word_label_ids.append(self.ner_label2id.get(tag, self.ner_label2id["O"]))

        actual_words = len(first_subword_idx)
        pad_len = num_words - actual_words
        first_subword_idx += [0] * pad_len
        word_label_ids += [0] * pad_len
        word_mask = [True] * actual_words + [False] * pad_len

        # Map each entity's word-level span to a subword-level span
        entity_subword_spans = {}
        for ent in r["entities"]:
            ws, we = ent["token_start"], ent["token_end"]
            subword_positions = [
                i for i, w_id in enumerate(word_ids)
                if w_id is not None and ws <= w_id < we
            ]
            if subword_positions:
                entity_subword_spans[ent["id"]] = (min(subword_positions), max(subword_positions) + 1)

        # Build candidate pairs: all annotated relations, plus a sample of negatives
        pairs = []
        pair_labels = []
        labeled_pairs = set()

        for rel in r["relations"]:
            a1, a2 = rel["arg1"], rel["arg2"]
            if a1 in entity_subword_spans and a2 in entity_subword_spans:
                s1, e1 = entity_subword_spans[a1]
                s2, e2 = entity_subword_spans[a2]
                pairs.append((s1, e1, s2, e2))
                pair_labels.append(self.re_label2id.get(rel["type"], self.re_label2id["NONE"]))
                labeled_pairs.add((a1, a2))

        entity_ids = list(entity_subword_spans.keys())
        for i in range(len(entity_ids)):
            for j in range(len(entity_ids)):
                if i == j:
                    continue
                a1, a2 = entity_ids[i], entity_ids[j]
                if (a1, a2) in labeled_pairs:
                    continue
                if len(pairs) >= self.max_pairs:
                    break
                s1, e1 = entity_subword_spans[a1]
                s2, e2 = entity_subword_spans[a2]
                pairs.append((s1, e1, s2, e2))
                pair_labels.append(self.re_label2id["NONE"])

        pairs = pairs[:self.max_pairs]
        pair_labels = pair_labels[:self.max_pairs]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "word_first_subword_idx": torch.tensor(first_subword_idx, dtype=torch.long),
            "word_mask": torch.tensor(word_mask, dtype=torch.bool),
            "word_labels": torch.tensor(word_label_ids, dtype=torch.long),
            "pairs": pairs,
            "pair_labels": pair_labels,
        }


def collate_fn(batch):
    input_ids = torch.stack([b["input_ids"] for b in batch])
    attention_mask = torch.stack([b["attention_mask"] for b in batch])

    max_words = max(b["word_mask"].shape[0] for b in batch)
    word_first_subword_idx = torch.zeros(len(batch), max_words, dtype=torch.long)
    word_mask = torch.zeros(len(batch), max_words, dtype=torch.bool)
    word_labels = torch.zeros(len(batch), max_words, dtype=torch.long)

    for i, b in enumerate(batch):
        n = b["word_mask"].shape[0]
        word_first_subword_idx[i, :n] = b["word_first_subword_idx"]
        word_mask[i, :n] = b["word_mask"]
        word_labels[i, :n] = b["word_labels"]

    pairs = [b["pairs"] for b in batch]
    pair_labels = [b["pair_labels"] for b in batch]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "word_first_subword_idx": word_first_subword_idx,
        "word_mask": word_mask,
        "word_labels": word_labels,
        "pairs": pairs,
        "pair_labels": pair_labels,
    }

