"""Decomposes a verdict into the individual factors that produced it."""

EVIDENCE_WEIGHTS = {"curated": 1.0, "gold": 0.9}


def build_score_breakdown(suitability, side_effects, evidence_sentences):
    factors = []
    verdict, etype = suitability["verdict"], suitability.get("evidence_type", "none")

    # 1. Primary evidence
    if etype == "curated":
        factors.append({"factor": "Established clinical indication", "impact": "positive", "weight": 1.0,
                        "detail": "Listed as a standard treatment in the curated clinical reference."})
    elif etype == "gold":
        impact = {"LIKELY_SUITABLE": "positive", "NOT_RECOMMENDED": "negative"}.get(verdict, "neutral")
        label = {"LIKELY_SUITABLE": "Human-annotated treatment evidence",
                 "NOT_RECOMMENDED": "Human-annotated adverse-effect evidence",
                 "MIXED_EVIDENCE": "Conflicting human-annotated evidence",
                 "ASSOCIATION_ONLY": "Human-annotated association (direction unknown)"}.get(verdict, "Literature evidence")
        factors.append({"factor": label, "impact": impact, "weight": EVIDENCE_WEIGHTS["gold"],
                        "detail": "Derived from expert annotations in BC5CDR, BioRED or the ADE corpus."})
    elif etype == "model":
        conf = suitability.get("model_confidence") or 0.0
        impact = {"LIKELY_SUITABLE": "positive", "NOT_RECOMMENDED": "negative"}.get(verdict, "neutral")
        factors.append({"factor": f"{suitability.get('model_name') or 'Model'} prediction ({conf * 100:.0f}% confidence)",
                        "impact": impact, "weight": round(conf, 2),
                        "detail": "Automatically extracted from literature by the relation-extraction model; "
                                  "weight equals the model's confidence."})
    elif verdict not in ("UNKNOWN_DRUG", "UNKNOWN_DISEASE"):
        factors.append({"factor": "No direct evidence", "impact": "neutral", "weight": 0.0,
                        "detail": "No annotated or predicted relation connects this drug and disease."})

    # 2. Side-effect burden
    total = suitability.get("side_effect_count") or 0
    if total:
        common = suitability.get("common_side_effect_count") or 0
        boxed = suitability.get("boxed_warning_count") or 0
        ratio = common / total
        if boxed:
            impact, weight = "negative", 0.6
        elif ratio > 0.3:
            impact, weight = "negative", 0.5
        elif total > 100:
            impact, weight = "negative", 0.3
        else:
            impact, weight = "neutral", 0.1
        detail = f"{common} of {total} known side effects are common or occasional (>= 1%)."
        if boxed:
            detail += f" {boxed} {'appears' if boxed == 1 else 'appear'} in an FDA boxed warning."
        factors.append({"factor": f"Side-effect burden ({total} known effects)", "impact": impact,
                        "weight": weight, "detail": detail})
    elif verdict not in ("UNKNOWN_DRUG",):
        factors.append({"factor": "No side-effect data", "impact": "neutral", "weight": 0.0,
                        "detail": "The drug was not found in SIDER or OnSIDES."})

    # 3. Data coverage
    srcs = suitability.get("side_effect_sources") or []
    if srcs:
        factors.append({"factor": f"Side-effect sources: {' + '.join(srcs)}", "impact": "informational",
                        "weight": 0.1 * len(srcs),
                        "detail": "Effects reported by both sources are ranked higher."})

    # 4. Supporting literature
    if evidence_sentences:
        n_gold = sum(e.get("origin") == "gold" for e in evidence_sentences)
        factors.append({"factor": f"Supporting literature ({len(evidence_sentences)} sentence(s))",
                        "impact": "informational", "weight": 0.2,
                        "detail": f"{n_gold} human-annotated and {len(evidence_sentences) - n_gold} model-extracted "
                                  f"source sentence(s) are shown below."})
    return factors


if __name__ == "__main__":
    demo = {"verdict": "NO_DIRECT_EVIDENCE", "evidence_type": "none", "side_effect_count": 86,
            "common_side_effect_count": 10, "boxed_warning_count": 0, "side_effect_sources": ["SIDER"]}
    for f in build_score_breakdown(demo, [], []):
        print(f"[{f['impact']}] {f['factor']} (weight={f['weight']}) - {f['detail']}")
