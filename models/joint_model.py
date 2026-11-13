import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from torchcrf import CRF


class JointNERREModel(nn.Module):
    """
    Shared transformer encoder with two enhanced heads:
      * NER: Mean Subword Pooling + residual LayerNorm-GELU projection + CRF
      * RE : [ent1 ; ent2 ; ent1 * ent2 ; |ent1 - ent2| ; between_ctx ; CLS] -> LayerNorm + 2-layer GELU MLP
             trained with label-smoothed cross-entropy for calibrated high-precision classification.
    """

    def __init__(self, backbone_name, num_ner_labels, num_re_labels, hidden_dropout=0.1):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone_name)
        hidden_size = self.backbone.config.hidden_size
        self.num_re_labels = num_re_labels

        self.dropout = nn.Dropout(hidden_dropout)
        self.ner_proj = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(hidden_dropout),
        )
        self.ner_classifier = nn.Linear(hidden_size, num_ner_labels)
        self.crf = CRF(num_ner_labels, batch_first=True)

        # 6 interaction features: [ent1, ent2, ent1 * ent2, |ent1 - ent2|, between_ctx, cls]
        self.re_classifier = nn.Sequential(
            nn.LayerNorm(hidden_size * 6),
            nn.Linear(hidden_size * 6, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(hidden_dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(hidden_dropout),
            nn.Linear(hidden_size, num_re_labels),
        )

    def head_parameters(self):
        return (list(self.ner_proj.parameters())
                + list(self.ner_classifier.parameters())
                + list(self.crf.parameters())
                + list(self.re_classifier.parameters()))

    def encode(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        return self.dropout(outputs.last_hidden_state)

    def ner_forward(self, sequence_output, word_first_subword_idx, word_mask, word_labels=None, word_spans_sub=None):
        hidden_size = sequence_output.shape[-1]
        bsz, max_words = word_first_subword_idx.shape

        first_tok = torch.gather(
            sequence_output, 1,
            word_first_subword_idx.unsqueeze(-1).expand(-1, -1, hidden_size)
        )
        if word_spans_sub is not None:
            seq_f = sequence_output.float()
            seq_len = seq_f.shape[1]
            prefix = torch.cat([torch.zeros_like(seq_f[:, :1]), torch.cumsum(seq_f, dim=1)], dim=1)
            b_idx = torch.arange(bsz, device=sequence_output.device).unsqueeze(1).expand(-1, max_words)
            s = word_spans_sub[:, :, 0].clamp(0, seq_len - 1)
            e = word_spans_sub[:, :, 1].clamp(1, seq_len)
            diff = (e - s).clamp(min=1).unsqueeze(-1)
            mean_tok = ((prefix[b_idx, e] - prefix[b_idx, s]) / diff).to(sequence_output.dtype)
            gathered = 0.5 * (first_tok + mean_tok)
        else:
            gathered = first_tok

        # Residual contextual projection before CRF emissions
        projected = gathered + self.ner_proj(gathered)
        emissions = self.ner_classifier(projected)

        with torch.autocast(device_type=emissions.device.type, enabled=False):
            emissions = emissions.float()
            if word_labels is not None:
                safe_labels = word_labels.clone()
                safe_labels[~word_mask] = 0
                loss = -self.crf(emissions, safe_labels, mask=word_mask, reduction="mean")
                return loss, emissions
            return self.crf.decode(emissions, mask=word_mask), emissions

    def re_forward(self, sequence_output, pair_spans, re_labels=None):
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

        seq = sequence_output.float()
        seq_len = seq.shape[1]
        prefix = torch.cat([torch.zeros_like(seq[:, :1]), torch.cumsum(seq, dim=1)], dim=1)
        b_t = torch.tensor(b_idx, device=device)
        s1_t = torch.tensor(s1, device=device).clamp(0, seq_len - 1)
        e1_t = torch.tensor(e1, device=device).clamp(1, seq_len)
        s2_t = torch.tensor(s2, device=device).clamp(0, seq_len - 1)
        e2_t = torch.tensor(e2, device=device).clamp(1, seq_len)

        ent1 = (prefix[b_t, e1_t] - prefix[b_t, s1_t]) / (e1_t - s1_t).clamp(min=1).unsqueeze(-1)
        ent2 = (prefix[b_t, e2_t] - prefix[b_t, s2_t]) / (e2_t - s2_t).clamp(min=1).unsqueeze(-1)

        mid_start = torch.minimum(e1_t, e2_t).clamp(0, seq_len)
        mid_end = torch.maximum(s1_t, s2_t).clamp(0, seq_len)
        has_mid = (mid_end > mid_start).unsqueeze(-1)
        mid_span = (prefix[b_t, torch.maximum(mid_start, mid_end)] - prefix[b_t, mid_start]) / (
            (mid_end - mid_start).clamp(min=1).unsqueeze(-1)
        )
        cls = seq[b_t, 0]
        between_ctx = torch.where(has_mid, mid_span, cls)

        prod = ent1 * ent2
        diff = torch.abs(ent1 - ent2)

        combined = torch.cat([ent1, ent2, prod, diff, between_ctx, cls], dim=-1).to(sequence_output.dtype)
        logits = self.re_classifier(combined)

        if re_labels is not None:
            target_tensor = torch.tensor(targets, device=device, dtype=torch.long)
            loss = F.cross_entropy(logits.float(), target_tensor, label_smoothing=0.03)
            return loss, logits
        return logits