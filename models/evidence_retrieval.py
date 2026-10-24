"""Literature evidence for a drug-disease pair.

Gold evidence comes from the annotated corpora; model evidence comes from the
relation-extraction predictions file. Both are indexed ONCE per process (not on
every request) and each hit is reduced to the sentence(s) that mention the two
entities instead of the whole abstract.
"""
import json
import os
from functools import lru_cache

import pv_config as cfg
from text_utils import normalize_name, token_char_spans, evidence_snippet, entity_char_span

SPLITS = ["ner_re_train.jsonl", "ner_re_val.jsonl", "ner_re_test.jsonl"]
PREDICTIONS_FILE = "model_predicted_associations.jsonl"
MAX_PER_PAIR = 5


def _iter_jsonl(path):
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


@lru_cache(maxsize=1)
def _gold_index():
    index = {}
    for split_file in SPLITS:
        for record in _iter_jsonl(cfg.PROCESSED_DIR / split_file):
            ents = {e["id"]: e for e in record["entities"]}
            spans = token_char_spans(record["text"])
            for rel in record["relations"]:
                a, b = ents.get(rel["arg1"]), ents.get(rel["arg2"])
                if not a or not b or {a["type"], b["type"]} != {"Chemical", "Disease"}:
                    continue
                chem, dis = (a, b) if a["type"] == "Chemical" else (b, a)
                key = (normalize_name(chem["text"]), normalize_name(dis["text"]))
                hits = index.setdefault(key, [])
                if len(hits) >= MAX_PER_PAIR:
                    continue
                ca = entity_char_span(record["text"], chem["token_start"], chem["token_end"], spans)
                cb = entity_char_span(record["text"], dis["token_start"], dis["token_end"], spans)
                snippet = evidence_snippet(record["text"], ca[0], cb[0]) if ca and cb else record["text"]
                if any(h["sentence"] == snippet for h in hits):
                    continue
                hits.append({"sentence": snippet, "relation_type": rel["type"], "origin": "gold",
                             "source_dataset": record["source_dataset"], "doc_id": record.get("doc_id"),
                             "drug_mention": chem["text"], "disease_mention": dis["text"], "confidence": None})
    return index


@lru_cache(maxsize=1)
def _model_index():
    index = {}
    for row in _iter_jsonl(cfg.PROCESSED_DIR / PREDICTIONS_FILE):
        key = (row["drug_name"], row["disease_name"])
        for ev in row.get("evidence", [])[:MAX_PER_PAIR]:
            index.setdefault(key, []).append({
                "sentence": ev["snippet"], "relation_type": row["relation"], "origin": "model",
                "source_dataset": ev.get("source_dataset"), "doc_id": ev.get("doc_id"),
                "drug_mention": ev.get("drug_mention", row["drug_name"]),
                "disease_mention": ev.get("disease_mention", row["disease_name"]),
                "confidence": ev.get("confidence"), "model": row.get("model"),
            })
    return index


def find_evidence_sentences(drug_name, disease_name, max_results=3, include_model=True):
    key = (normalize_name(drug_name), normalize_name(disease_name))
    gold = list(_gold_index().get(key, []))
    model = list(_model_index().get(key, [])) if include_model else []
    seen, out = set(), []
    for ev in gold + model:
        if ev["sentence"] in seen:
            continue
        seen.add(ev["sentence"])
        out.append(ev)
        if len(out) >= max_results:
            break
    return out


if __name__ == "__main__":
    for r in find_evidence_sentences("alpha-methyldopa", "hypotensive"):
        print(f"[{r['origin']}/{r['source_dataset']}] {r['relation_type']}: \"{r['sentence']}\"")
