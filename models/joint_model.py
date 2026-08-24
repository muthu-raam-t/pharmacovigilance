import torch
import torch.nn as nn
from transformers import AutoModel
from torchcrf import CRF


class JointNERREModel(nn.Module):
    def __init__(self, backbone_name, num_ner_labels, num_re_labels, hidden_dropout=0.1):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone_name)
        hidden_size = self.backbone.config.hidden_size

        self.dropout = nn.Dropout(hidden_dropout)

        # NER head: linear projection to tag scores, decoded via CRF
        self.ner_classifier = nn.Linear(hidden_size, num_ner_labels)
        self.crf = CRF(num_ner_labels, batch_first=True)

        # RE head: takes [entity1_avg ; entity2_avg ; CLS] -> relation class
        self.re_classifier = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.ReLU(),
            nn.Dropout(hidden_dropout),
            nn.Linear(hidden_size, num_re_labels)
        )

    def encode(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        return self.dropout(outputs.last_hidden_state)  # (batch, seq_len, hidden)

    def ner_forward(self, sequence_output, word_first_subword_idx, word_mask, word_labels=None):
        """
        word_first_subword_idx: (batch, max_words) subword index of each word's first token
        word_mask: (batch, max_words) bool, True where a real word exists
        word_labels: (batch, max_words) label ids, only used during training
        """
        batch_size, max_words = word_first_subword_idx.shape
        hidden_size = sequence_output.shape[-1]

        gathered = torch.gather(
            sequence_output, 1,
            word_first_subword_idx.unsqueeze(-1).expand(-1, -1, hidden_size)
        )  # (batch, max_words, hidden)

        emissions = self.ner_classifier(gathered)  # (batch, max_words, num_ner_labels)

        if word_labels is not None:
            safe_labels = word_labels.clone()
            safe_labels[~word_mask] = 0
            loss = -self.crf(emissions, safe_labels, mask=word_mask, reduction="mean")
            return loss, emissions
        else:
            predicted_tags = self.crf.decode(emissions, mask=word_mask)
            return predicted_tags, emissions

    def re_forward(self, sequence_output, pair_spans, re_labels=None):
        """
        pair_spans: list (len=batch) of lists of (e1_start, e1_end, e2_start, e2_end) subword-index tuples
        re_labels: list (len=batch) of lists of label ids, same shape as pair_spans, only for training
        Returns: loss (if training) and list of predicted logits per example
        """
        all_logits = []
        all_targets = []

        for b_idx, pairs in enumerate(pair_spans):
            cls_vec = sequence_output[b_idx, 0, :]  # (hidden,)
            for p_idx, (e1s, e1e, e2s, e2e) in enumerate(pairs):
                e1_vec = sequence_output[b_idx, e1s:e1e, :].mean(dim=0)
                e2_vec = sequence_output[b_idx, e2s:e2e, :].mean(dim=0)
                combined = torch.cat([e1_vec, e2_vec, cls_vec], dim=-1)
                logits = self.re_classifier(combined)
                all_logits.append(logits)
                if re_labels is not None:
                    all_targets.append(re_labels[b_idx][p_idx])

        if not all_logits:
            return (None, []) if re_labels is not None else []

        stacked_logits = torch.stack(all_logits)

        if re_labels is not None:
            target_tensor = torch.tensor(all_targets, device=stacked_logits.device, dtype=torch.long)
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(stacked_logits, target_tensor)
            return loss, stacked_logits
        else:
            return stacked_logits
