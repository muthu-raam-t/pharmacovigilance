"""ADE Corpus (DRUG-AE.rel) -> unified JSONL.

Fixes in this version:
  * The offsets in DRUG-AE.rel are relative to the whole abstract, not to the
    sentence, so they are NOT used. Each drug / effect is located inside its
    sentence by case-insensitive string matching instead.
  * DRUG-AE.rel has one line per drug-effect pair, so the same sentence appears
    several times. All pairs of a sentence are merged into ONE record that
    carries every drug and effect mention and every relation. This removes
    duplicate sentences (and therefore train/test leakage) and stops other
    entities in the sentence from being labelled "O".
"""
import os
from collections import OrderedDict
from preprocessing_utils import (tokenize_with_spans, assign_bio_tags, find_all_spans, write_jsonl,
                                 RAW_DIR as RAW_ROOT, PROCESSED_DIR)

RAW_PATH = str(RAW_ROOT / "ade_corpus" / "DRUG-AE.rel")
OUT_DIR = str(PROCESSED_DIR)


def parse_ade_file(filepath):
    """Groups DRUG-AE.rel lines by (PMID, sentence)."""
    groups = OrderedDict()
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            parts = [p.strip() for p in line.rstrip("\n").split("|")]
            if len(parts) < 4:
                continue
            pmid, sentence = parts[0], parts[1]
            if len(parts) >= 8:
                effect, drug = parts[2], parts[5]
            else:
                effect, drug = parts[2], parts[3]
            if not sentence or not effect or not drug:
                continue
            g = groups.setdefault((pmid, sentence), {"pmid": pmid, "sentence": sentence, "pairs": []})
            if (drug, effect) not in g["pairs"]:
                g["pairs"].append((drug, effect))
    return list(groups.values())


def _place(sentence, phrase, taken):
    """First occurrence of phrase that does not overlap an already placed mention."""
    for start, end in find_all_spans(sentence, phrase):
        if all(end <= s or start >= e for s, e in taken):
            return start, end
    return None


def process_groups(groups):
    records, unresolved_pairs = [], 0
    for k, g in enumerate(groups):
        sentence = g["sentence"]
        tokens_with_spans = tokenize_with_spans(sentence)
        tokens = [t[0] for t in tokens_with_spans]

        # Place longer phrases first so that nested names resolve sensibly
        phrases = {}
        for drug, effect in g["pairs"]:
            phrases.setdefault(("Chemical", drug.lower()), drug)
            phrases.setdefault(("Disease", effect.lower()), effect)

        taken, raw_entities, key_to_id = [], [], {}
        for (etype, key), surface in sorted(phrases.items(), key=lambda kv: -len(kv[1])):
            span = _place(sentence, surface, taken)
            if span is None:
                continue
            taken.append(span)
            ent_id = f"T{len(raw_entities)}"
            key_to_id[(etype, key)] = ent_id
            raw_entities.append({
                "id": ent_id, "text": sentence[span[0]:span[1]], "type": etype,
                "char_start": span[0], "char_end": span[1], "concept_ids": [],
            })

        ner_tags, resolved = assign_bio_tags(tokens_with_spans, raw_entities)
        resolved_ids = {e["id"] for e in resolved}

        relations = []
        for drug, effect in g["pairs"]:
            a1 = key_to_id.get(("Chemical", drug.lower()))
            a2 = key_to_id.get(("Disease", effect.lower()))
            if a1 in resolved_ids and a2 in resolved_ids:
                relations.append({"type": "ADVERSE", "arg1": a1, "arg2": a2})
            else:
                unresolved_pairs += 1

        if not relations:
            continue
        records.append({
            "doc_id": f"ADE_{g['pmid']}_{k}",
            "group_id": g["pmid"],
            "text": sentence,
            "tokens": tokens,
            "ner_tags": ner_tags,
            "entities": resolved,
            "relations": relations,
            "source_dataset": "ADE",
        })
    return records, unresolved_pairs


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    groups = parse_ade_file(RAW_PATH)
    n_pairs = sum(len(g["pairs"]) for g in groups)
    records, unresolved = process_groups(groups)
    n_rel = sum(len(r["relations"]) for r in records)
    print(f"Drug-effect pairs read: {n_pairs} in {len(groups)} unique sentences")
    print(f"Pairs resolved to entity spans: {n_rel} ({n_rel / max(n_pairs, 1):.1%}); unresolved: {unresolved}")
    write_jsonl(records, os.path.join(OUT_DIR, "ade_processed.jsonl"))
