"""Shared helpers for converting the raw corpora into the unified JSONL schema.

Every processed record has the form:
    {
      "doc_id":   unique record id,
      "group_id": PubMed id, used for leakage-free splitting,
      "text":     raw text,
      "tokens":   word tokens,
      "ner_tags": BIO tags (one per token),
      "entities": [{"id", "text", "type", "token_start", "token_end", "concept_ids"}],
      "relations":[{"type", "arg1", "arg2"}],      # mention-level
      "source_dataset": "BC5CDR" | "BioRED" | "ADE"
    }
"""
import json
import os
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE = Path(os.environ.get("WORKSPACE_DIR", REPO_ROOT))
RAW_DIR = Path(os.environ.get("RAW_DIR", WORKSPACE / "data" / "raw"))
PROCESSED_DIR = Path(os.environ.get("PROCESSED_DIR", WORKSPACE / "data" / "processed"))
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", WORKSPACE / "results"))

TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]")


def tokenize_with_spans(text):
    """Simple regex tokenizer returning (token, char_start, char_end)."""
    return [(m.group(), m.start(), m.end()) for m in TOKEN_PATTERN.finditer(text)]


def assign_bio_tags(tokens_with_spans, entities):
    """
    entities: list of dicts with id, text, type, char_start, char_end (and optional concept_ids).
    Returns BIO tags and the resolved entities (with token_start / token_end).

    Overlapping or nested mentions are resolved by keeping the longest mention;
    any mention that overlaps an already-tagged token is dropped, so the BIO
    sequence is always valid.
    """
    tags = ["O"] * len(tokens_with_spans)
    resolved = []

    ordered = sorted(entities, key=lambda e: (-(e["char_end"] - e["char_start"]), e["char_start"]))
    for ent in ordered:
        e_start, e_end = ent["char_start"], ent["char_end"]
        idxs = [i for i, (_, t_start, t_end) in enumerate(tokens_with_spans)
                if t_start < e_end and t_end > e_start]
        if not idxs or any(tags[i] != "O" for i in idxs):
            continue
        for j, idx in enumerate(idxs):
            tags[idx] = ("B-" if j == 0 else "I-") + ent["type"]
        resolved.append({
            "id": ent["id"],
            "text": ent["text"],
            "type": ent["type"],
            "token_start": idxs[0],
            "token_end": idxs[-1] + 1,
            "concept_ids": ent.get("concept_ids", []),
        })

    resolved.sort(key=lambda e: e["token_start"])
    return tags, resolved


def split_concept_ids(raw):
    """PubTator concept fields may hold several ids separated by '|', ',' or ';'."""
    if raw is None:
        return []
    ids = [c.strip() for c in re.split(r"[|,;]", str(raw))]
    return [c for c in ids if c and c not in ("-1", "-")]


def expand_concept_relations(resolved_entities, concept_relations):
    """
    Converts concept-level relations (concept_a, concept_b, type) into
    mention-level relations between every pair of mentions of the two concepts.
    """
    mentions_by_concept = {}
    for ent in resolved_entities:
        for cid in ent["concept_ids"]:
            mentions_by_concept.setdefault(cid, []).append(ent["id"])

    relations, seen = [], set()
    for c1, c2, rel_type in concept_relations:
        for m1 in mentions_by_concept.get(c1, []):
            for m2 in mentions_by_concept.get(c2, []):
                if m1 == m2:
                    continue
                key = (min(m1, m2), max(m1, m2))
                if key in seen:
                    continue
                seen.add(key)
                relations.append({"type": rel_type, "arg1": m1, "arg2": m2})
    return relations


def find_all_spans(text, phrase):
    """All case-insensitive occurrences of phrase in text, preferring word-boundary matches."""
    if not phrase:
        return []
    strict = re.compile(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", re.IGNORECASE)
    spans = [(m.start(), m.end()) for m in strict.finditer(text)]
    if not spans:
        loose = re.compile(re.escape(phrase), re.IGNORECASE)
        spans = [(m.start(), m.end()) for m in loose.finditer(text)]
    return spans


def write_jsonl(records, out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {len(records)} records to {out_path}")


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
