"""Candidate entity-pair construction shared by the transformer dataset, the
evaluation code and the TF-IDF baseline, so every model is scored on exactly
the same pairs.

Pairs are UNORDERED (relations in these corpora are not direction-sensitive at
the mention level), restricted to entity-type combinations that actually carry
relations in the training data, and pairs of two mentions with identical text
are skipped (they are the same concept).
"""

NONE_LABEL = "NONE"


def type_pair_key(t1, t2):
    return tuple(sorted((t1, t2)))


def relation_lookup(record):
    lookup = {}
    for rel in record["relations"]:
        key = tuple(sorted((rel["arg1"], rel["arg2"])))
        lookup.setdefault(key, rel["type"])
    return lookup


def entity_text(record, ent):
    return " ".join(record["tokens"][ent["token_start"]:ent["token_end"]]).lower()


def candidate_pairs(record, allowed_type_pairs, max_word=None):
    """
    Returns a list of (entity_a, entity_b, label) with entity_a before entity_b in the text.
    max_word: if given, only entities that end at or before this word index are used
              (i.e. entities that survive transformer truncation).
    """
    allowed = {tuple(p) for p in allowed_type_pairs}
    ents = sorted(record["entities"], key=lambda e: (e["token_start"], e["token_end"]))
    if max_word is not None:
        ents = [e for e in ents if e["token_end"] <= max_word]
    lookup = relation_lookup(record)
    texts = {e["id"]: entity_text(record, e) for e in ents}

    pairs = []
    for i in range(len(ents)):
        for j in range(i + 1, len(ents)):
            a, b = ents[i], ents[j]
            if type_pair_key(a["type"], b["type"]) not in allowed:
                continue
            if texts[a["id"]] == texts[b["id"]]:
                continue
            label = lookup.get(tuple(sorted((a["id"], b["id"]))), NONE_LABEL)
            pairs.append((a, b, label))
    return pairs


def gold_relation_set(record):
    """Canonical set of gold relations: ((start, end, type), (start, end, type), label)."""
    by_id = {e["id"]: e for e in record["entities"]}
    out = set()
    for rel in record["relations"]:
        a, b = by_id.get(rel["arg1"]), by_id.get(rel["arg2"])
        if a is None or b is None:
            continue
        sa = (a["token_start"], a["token_end"], a["type"])
        sb = (b["token_start"], b["token_end"], b["type"])
        first, second = sorted((sa, sb))
        out.add((first, second, rel["type"]))
    return out
