import re
import json

def tokenize_with_spans(text):
    """Simple regex tokenizer returning (token, char_start, char_end)."""
    tokens = []
    for match in re.finditer(r"\w+|[^\w\s]", text):
        tokens.append((match.group(), match.start(), match.end()))
    return tokens

def assign_bio_tags(tokens_with_spans, entities):
    """
    entities: list of dicts with char_start, char_end, type
    Returns: ner_tags list, and entities updated with token_start/token_end
    """
    tags = ["O"] * len(tokens_with_spans)
    resolved_entities = []

    for ent in entities:
        e_start, e_end, e_type = ent["char_start"], ent["char_end"], ent["type"]
        matched_token_idxs = [
            i for i, (_, t_start, t_end) in enumerate(tokens_with_spans)
            if t_start < e_end and t_end > e_start
        ]
        if not matched_token_idxs:
            continue
        for j, idx in enumerate(matched_token_idxs):
            tags[idx] = f"B-{e_type}" if j == 0 else f"I-{e_type}"

        resolved_entities.append({
            "id": ent.get("id"),
            "text": ent.get("text"),
            "type": e_type,
            "token_start": matched_token_idxs[0],
            "token_end": matched_token_idxs[-1] + 1
        })

    return tags, resolved_entities

def write_jsonl(records, out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {len(records)} records to {out_path}")
