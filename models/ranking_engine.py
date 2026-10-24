"""Fuses SIDER and OnSIDES side-effect evidence into one ranked side-effect profile.

rank score = frequency weight (common 3, occasional 2, rare 1, unknown 0)
           + 3 if the effect is in a boxed warning on the drug label (OnSIDES)
           + 1 if both independent sources report it
           + fraction of the drug's labels that list it (0-1, OnSIDES)
"""
from functools import lru_cache

from severity_scoring import get_severity_scores_for_drug, compute_severity_score
from text_utils import normalize_name

SEVERITY_WEIGHTS = {"common": 3, "occasional": 2, "rare": 1, "unknown": 0}
BOXED_WARNING_BONUS = 3
BOTH_SOURCES_BONUS = 1


@lru_cache(maxsize=2048)
def _ranked(drug_name):
    ranked = []
    for entry in get_severity_scores_for_drug(drug_name):
        bucket, freq = compute_severity_score(entry["frequency"])
        both = len(entry["sources"]) > 1
        score = (SEVERITY_WEIGHTS[bucket]
                 + (BOXED_WARNING_BONUS if entry["boxed_warning"] else 0)
                 + (BOTH_SOURCES_BONUS if both else 0)
                 + (entry["label_fraction"] or 0.0))
        ranked.append({
            "side_effect": entry["side_effect"],
            "frequency": freq,
            "severity_bucket": bucket,
            "severity_weight": SEVERITY_WEIGHTS[bucket],
            "boxed_warning": entry["boxed_warning"],
            "label_fraction": entry["label_fraction"],
            "sources": entry["sources"],
            "confirmed_by_both_sources": both,
            "rank_score": round(score, 4),
        })
    ranked.sort(key=lambda x: (x["rank_score"], x["frequency"] or 0.0), reverse=True)
    return tuple(ranked)


def get_ranked_side_effects(drug_name):
    """Ranked list of side effects (most severe / frequent first). Results are cached per drug."""
    return [dict(e) for e in _ranked(normalize_name(drug_name))]


def side_effect_burden(ranked):
    """Single number summarising how heavy a side-effect profile is (lower is better)."""
    unknown_weight = 0.25
    score = 0.0
    for e in ranked:
        score += SEVERITY_WEIGHTS[e["severity_bucket"]] or unknown_weight
        if e["boxed_warning"]:
            score += BOXED_WARNING_BONUS
    return round(score, 2)


if __name__ == "__main__":
    results = get_ranked_side_effects("aspirin")
    print(f"Ranked {len(results)} side effects for aspirin\n")
    for r in results[:15]:
        f = f"{r['frequency']:.3f}" if r["frequency"] is not None else "N/A"
        print(f"  {r['side_effect']:<30} {r['severity_bucket']:<10} freq={f:<6} sources={'+'.join(r['sources'])}"
              f"{'  BOXED' if r['boxed_warning'] else ''}")
