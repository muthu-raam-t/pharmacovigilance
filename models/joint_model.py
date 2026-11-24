import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from torchcrf import CRF


class ScalarMix(nn.Module):
    """Learns a weighted combination of the last N transformer hidden layers."""

    def __init__(self, num_layers=4):
        super().__init__()
        self.weights = nn.Parameter(torch.zeros(num_layers))
        self.gamma = nn.Parameter(torch.ones(1))

    def forward(self, hidden_states):
        norm_weights = F.softmax(self.weights, dim=0)
        stacked = torch.stack(hidden_states, dim=0)  # (N, B, L, H)
        mixed = torch.sum(norm_weights.view(-1, 1, 1, 1) * stacked, dim=0)
        return self.gamma * mixed


class BiaffineScorer(nn.Module):
    """Low-rank biaffine interaction scorer between entity1 and entity2 representations."""

    def __init__(self, in_dim, proj_dim, num_classes, dropout=0.2):
        super().__init__()
        self.proj1 = nn.Sequential(nn.Linear(in_dim, proj_dim), nn.GELU(), nn.Dropout(dropout))
        self.proj2 = nn.Sequential(nn.Linear(in_dim, proj_dim), nn.GELU(), nn.Dropout(dropout))
        self.U = nn.Parameter(torch.empty(num_classes, proj_dim, proj_dim))
        nn.init.xavier_uniform_(self.U)

    def forward(self, e1, e2):
        h1 = self.proj1(e1)  # (N, P)
        h2 = self.proj2(e2)  # (N, P)
        return torch.einsum("ni,cij,nj->nc", h1, self.U, h2)


class JointNERREModel(nn.Module):
    """
    Enhanced Task-Partitioned Joint NER + RE Architecture:
      * Encoder  : Last-4-layer ScalarMix (separate mixes for NER lexical/boundary cues vs. RE semantics)
                   with gradient checkpointing for 512-token batches on 6GB GPUs.
      * Gating   : FiTaCNN-inspired task-specific feature filter gates (ner_gate, re_gate).
      * NER Head : First+Mean Subword Pooling -> Residual BiLSTM -> Multi-Sample Dropout -> token_mean CRF.
      * RE Head  : [ent1; ent2; ent1*ent2; |ent1-ent2|; between_mean; between_max; CLS; type1; type2; dist]
                   -> Multi-Sample Dropout MLP + Biaffine Scorer, trained with label-smoothed cross-entropy.
    """

    def __init__(self, backbone_name, num_ner_labels, num_re_labels, hidden_dropout=0.15,
                 num_entity_types=16, type_emb_dim=32, dist_emb_dim=32, grad_checkpointing=False):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone_name, output_hidden_states=True)
        if grad_checkpointing and hasattr(self.backbone, "gradient_checkpointing_enable"):
            try:
                self.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            except TypeError:
                self.backbone.gradient_checkpointing_enable()

        hidden_size = self.backbone.config.hidden_size
        self.num_re_labels = num_re_labels
        self.dropout = nn.Dropout(hidden_dropout)

        # Task-specific layer mixes (last 4 transformer layers)
        self.ner_layer_mix = ScalarMix(num_layers=4)
        self.re_layer_mix = ScalarMix(num_layers=4)

        # Task-specific feature filter gates
        self.ner_gate = nn.Linear(hidden_size, hidden_size)
        self.re_gate = nn.Linear(hidden_size, hidden_size)

        # NER Head: Residual BiLSTM + Multi-Sample Dropout + CRF
        self.ner_norm = nn.LayerNorm(hidden_size)
        self.ner_lstm = nn.LSTM(
            hidden_size, hidden_size // 2, num_layers=1,
            batch_first=True, bidirectional=True
        )
        self.ner_ms_dropouts = nn.ModuleList([nn.Dropout(p) for p in (0.1, 0.2, 0.3, 0.4, 0.5)])
        self.ner_classifier = nn.Linear(hidden_size, num_ner_labels)
        self.crf = CRF(num_ner_labels, batch_first=True)

        # RE Head: Entity-type & distance embeddings + Max/Mean context + Biaffine + Multi-Sample Dropout
        self.type_emb = nn.Embedding(num_entity_types, type_emb_dim)
        self.dist_emb = nn.Embedding(8, dist_emb_dim)

        # 7 hidden vectors: [ent1, ent2, ent1*ent2, |ent1-ent2|, between_mean, between_max, cls]
        re_in_dim = hidden_size * 7 + type_emb_dim * 2 + dist_emb_dim
        self.re_proj = nn.Sequential(
            nn.LayerNorm(re_in_dim),
            nn.Linear(re_in_dim, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(hidden_dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
        )
        self.re_ms_dropouts = nn.ModuleList([nn.Dropout(p) for p in (0.1, 0.2, 0.3, 0.4, 0.5)])
        self.re_classifier = nn.Linear(hidden_size, num_re_labels)
        self.biaffine = BiaffineScorer(hidden_size, proj_dim=256, num_classes=num_re_labels, dropout=hidden_dropout)

    def head_parameters(self):
        return (list(self.ner_layer_mix.parameters())
                + list(self.re_layer_mix.parameters())
                + list(self.ner_gate.parameters())
                + list(self.re_gate.parameters())
                + list(self.ner_norm.parameters())
                + list(self.ner_lstm.parameters())
                + list(self.ner_classifier.parameters())
                + list(self.crf.parameters())
                + list(self.type_emb.parameters())
                + list(self.dist_emb.parameters())
                + list(self.re_proj.parameters())
                + list(self.re_classifier.parameters())
                + list(self.biaffine.parameters()))

    def encode(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        last4 = outputs.hidden_states[-4:]
        ner_seq = self.dropout(self.ner_layer_mix(last4))
        re_seq = self.dropout(self.re_layer_mix(last4))
        return (ner_seq, re_seq)

    def _unpack_seq(self, sequence_output):
        if isinstance(sequence_output, tuple):
            return sequence_output
        return sequence_output, sequence_output

    def ner_forward(self, sequence_output, word_first_subword_idx, word_mask, word_labels=None, word_spans_sub=None):
        ner_base, _ = self._unpack_seq(sequence_output)
        ner_seq = ner_base * torch.sigmoid(self.ner_gate(ner_base))
        hidden_size = ner_seq.shape[-1]
        bsz, max_words = word_first_subword_idx.shape

        first_tok = torch.gather(
            ner_seq, 1,
            word_first_subword_idx.unsqueeze(-1).expand(-1, -1, hidden_size)
        )
        if word_spans_sub is not None:
            seq_f = ner_seq.float()
            seq_len = seq_f.shape[1]
            prefix = torch.cat([torch.zeros_like(seq_f[:, :1]), torch.cumsum(seq_f, dim=1)], dim=1)
            b_idx = torch.arange(bsz, device=ner_seq.device).unsqueeze(1).expand(-1, max_words)
            s = word_spans_sub[:, :, 0].clamp(0, seq_len - 1)
            e = word_spans_sub[:, :, 1].clamp(1, seq_len)
            diff = (e - s).clamp(min=1).unsqueeze(-1)
            mean_tok = ((prefix[b_idx, e] - prefix[b_idx, s]) / diff).to(ner_seq.dtype)
            gathered = 0.5 * (first_tok + mean_tok)
        else:
            gathered = first_tok

        normed = self.ner_norm(gathered)
        lstm_out, _ = self.ner_lstm(normed.float())
        projected = normed + lstm_out.to(normed.dtype)

        # Multi-sample dropout averaged emissions
        emissions = sum(self.ner_classifier(dp(projected)) for dp in self.ner_ms_dropouts) / len(self.ner_ms_dropouts)

        with torch.autocast(device_type=emissions.device.type, enabled=False):
            emissions = emissions.float()
            if word_labels is not None:
                safe_labels = word_labels.clone()
                safe_labels[~word_mask] = 0
                loss = -self.crf(emissions, safe_labels, mask=word_mask, reduction="token_mean")
                return loss, emissions
            return self.crf.decode(emissions, mask=word_mask), emissions

    def re_forward(self, sequence_output, pair_spans, re_labels=None, pair_types=None, pair_dists=None):
        _, re_base = self._unpack_seq(sequence_output)
        b_idx, s1, e1, s2, e2, targets, t1_ids, t2_ids, dist_ids = [], [], [], [], [], [], [], [], []
        for b, pairs in enumerate(pair_spans):
            for p, (a_s, a_e, c_s, c_e) in enumerate(pairs):
                b_idx.append(b)
                s1.append(a_s); e1.append(a_e); s2.append(c_s); e2.append(c_e)
                if re_labels is not None:
                    targets.append(re_labels[b][p])
                if pair_types is not None:
                    t1_ids.append(pair_types[b][p][0])
                    t2_ids.append(pair_types[b][p][1])
                else:
                    t1_ids.append(0); t2_ids.append(0)
                if pair_dists is not None:
                    dist_ids.append(pair_dists[b][p])
                else:
                    dist_ids.append(0)

        device = re_base.device
        if not b_idx:
            empty = re_base.new_zeros((0, self.num_re_labels))
            return (None, empty) if re_labels is not None else empty

        re_seq = re_base * torch.sigmoid(self.re_gate(re_base))
        seq = re_seq.float()
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
        mid_mean = (prefix[b_t, torch.maximum(mid_start, mid_end)] - prefix[b_t, mid_start]) / (
            (mid_end - mid_start).clamp(min=1).unsqueeze(-1)
        )
        cls = seq[b_t, 0]
        between_mean = torch.where(has_mid, mid_mean, cls)

        # Max-pooled intervening context captures sharp relation trigger tokens ("induced", "treated", etc.)
        between_max_list = []
        for idx_p in range(len(b_idx)):
            ms, me = int(mid_start[idx_p].item()), int(mid_end[idx_p].item())
            if me > ms:
                between_max_list.append(seq[b_idx[idx_p], ms:me].max(dim=0).values)
            else:
                between_max_list.append(cls[idx_p])
        between_max = torch.stack(between_max_list, dim=0)

        prod = ent1 * ent2
        diff = torch.abs(ent1 - ent2)

        t1_emb = self.type_emb(torch.tensor(t1_ids, device=device, dtype=torch.long)).float()
        t2_emb = self.type_emb(torch.tensor(t2_ids, device=device, dtype=torch.long)).float()
        d_emb = self.dist_emb(torch.tensor(dist_ids, device=device, dtype=torch.long)).float()

        combined = torch.cat(
            [ent1, ent2, prod, diff, between_mean, between_max, cls, t1_emb, t2_emb, d_emb], dim=-1
        ).to(re_base.dtype)

        proj = self.re_proj(combined)
        mlp_logits = sum(self.re_classifier(dp(proj)) for dp in self.re_ms_dropouts) / len(self.re_ms_dropouts)
        biaffine_logits = self.biaffine(ent1.to(re_base.dtype), ent2.to(re_base.dtype))
        logits = mlp_logits + biaffine_logits

        if re_labels is not None:
            target_tensor = torch.tensor(targets, device=device, dtype=torch.long)
            loss = F.cross_entropy(logits.float(), target_tensor, label_smoothing=0.02)
            return loss, logits
        return logits