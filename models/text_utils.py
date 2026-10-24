"""Text helpers shared by evidence retrieval and model-prediction generation."""
import re

TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]")          # identical to scripts/preprocessing_utils.py
SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])")


def normalize_name(name):
    return " ".join(str(name).strip().lower().split())


def token_char_spans(text):
    return [(m.start(), m.end()) for m in TOKEN_PATTERN.finditer(text)]


def sentence_spans(text):
    spans, start = [], 0
    for m in SENTENCE_BOUNDARY.finditer(text):
        spans.append((start, m.start()))
        start = m.end()
    spans.append((start, len(text)))
    return spans


def evidence_snippet(text, char_a, char_b):
    """Sentence containing both character positions, or the two sentences joined with an ellipsis."""
    sents = sentence_spans(text)

    def which(pos):
        for i, (s, e) in enumerate(sents):
            if s <= pos < e or (pos == e and i == len(sents) - 1):
                return i
        return len(sents) - 1

    i, j = sorted((which(char_a), which(char_b)))
    if i == j:
        s, e = sents[i]
        return text[s:e].strip()
    return text[sents[i][0]:sents[i][1]].strip() + " … " + text[sents[j][0]:sents[j][1]].strip()


def entity_char_span(text, token_start, token_end, spans=None):
    spans = spans if spans is not None else token_char_spans(text)
    if token_end - 1 >= len(spans):
        return None
    return spans[token_start][0], spans[token_end - 1][1]
