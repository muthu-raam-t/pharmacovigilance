import torch
import torch.nn as nn
from transformers import AutoModel
from torchcrf import CRF


class JointNERREModel(nn.Module):
    """
    Shared transformer encoder with two heads:
      * NER: linear emissions over the first sub-word of every word, decoded with a CRF
      * RE : [mean(entity_1) ; mean(entity_2) ; CLS] -> 2-layer MLP -> relation class
    Parameter names are unchanged from the first version, so older checkpoints still load.
    """

    def __init__(self, backbone_name, num_ner_labels, num_re_labels, hidden_dropout=0.1):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone_name)
        hidden_size = self.backbone.config.hidden_size
        self.num_re_labels = num_re_labels

        self.dropout = nn.Dropout(hidden_dropout)
        self.ner_classifier = nn.Linear(hidden_size, num_ner_labels)
        self.crf = CRF(num_ner_labels, batch_first=True)
        self.re_classifier = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.ReLU(),
            nn.Dropout(hidden_dropout),
            nn.Linear(hidden_size, num_re_labels),
        )

    def head_parameters(self):
        return list(self.ner_classifier.parameters()) + list(self.crf.parameters()) + \
            list(self.re_classifier.parameters())

    def encode(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        return self.dropout(outputs.last_hidden_state)  # (batch, seq_len, hidden)

    def ner_forward(self, sequence_output, word_first_subword_idx, word_mask, word_labels=None):
        """
        word_first_subword_idx: (batch, max_words) sub-word index of each word's first token
        word_mask:              (batch, max_words) bool, True where a real word exists
        word_labels:            (batch, max_words) label ids (training only)
        """
        hidden_size = sequence_output.shape[-1]
        gathered = torch.gather(
            sequence_output, 1,
            word_first_subword_idx.unsqueeze(-1).expand(-1, -1, hidden_size)
        )
        emissions = self.ner_classifier(gathered)

        # The CRF always runs in float32, also under mixed precision
        with torch.autocast(device_type=emissions.device.type, enabled=False):
            emissions = emissions.float()
            if word_labels is not None:
                safe_labels = word_labels.clone()
                safe_labels[~word_mask] = 0
                loss = -self.crf(emissions, safe_labels, mask=word_mask, reduction="mean")
                return loss, emissions
            return self.crf.decode(emissions, mask=word_mask), emissions

    def re_forward(self, sequence_output, pair_spans, re_labels=None):
        """
        pair_spans: list (len = batch) of lists of (e1_start, e1_end, e2_start, e2_end) sub-word spans
        re_labels:  list (len = batch) of lists of label ids (training only)
        Returns (loss, logits) when re_labels is given, otherwise logits of shape (n_pairs, n_labels).
        """
        b_idx, s1, e1, s2, e2, targets = [], [], [], [], [], []
        for b, pairs in enumerate(pair_spans):
            for p, (a_s, a_e, c_s, c_e) in enumerate(pairs):
                b_idx.append(b)
                s1.append(a_s); e1.append(a_e); s2.append(c_s); e2.append(c_e)
                if re_labels is not None:
                    targets.append(re_labels[b][p])

        device = sequence_output.device
        if not b_idx:
            empty = sequence_output.new_zeros((0, self.num_re_labels))
            return (None, empty) if re_labels is not None else empty

        # Span means via prefix sums (float32 to avoid fp16 cancellation)
        seq = sequence_output.float()
        prefix = torch.cat([torch.zeros_like(seq[:, :1]), torch.cumsum(seq, dim=1)], dim=1)
        b_t = torch.tensor(b_idx, device=device)
        s1_t, e1_t = torch.tensor(s1, device=device), torch.tensor(e1, device=device)
        s2_t, e2_t = torch.tensor(s2, device=device), torch.tensor(e2, device=device)

        ent1 = (prefix[b_t, e1_t] - prefix[b_t, s1_t]) / (e1_t - s1_t).clamp(min=1).unsqueeze(-1)
        ent2 = (prefix[b_t, e2_t] - prefix[b_t, s2_t]) / (e2_t - s2_t).clamp(min=1).unsqueeze(-1)
        cls = seq[b_t, 0]

        combined = torch.cat([ent1, ent2, cls], dim=-1).to(sequence_output.dtype)
        logits = self.re_classifier(combined)

        if re_labels is not None:
            target_tensor = torch.tensor(targets, device=device, dtype=torch.long)
            loss = nn.functional.cross_entropy(logits.float(), target_tensor)
            return loss, logits
        return logits
