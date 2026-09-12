def build_score_breakdown(suitability_result, side_effects, evidence_sentences):
    """
    Decomposes the suitability verdict into individual weighted factors,
    so the user can see exactly why the system reached its conclusion.
    """
    factors = []

    if suitability_result["verdict"] == "NOT_RECOMMENDED":
        factors.append({
            "factor": "Direct adverse-event evidence found",
            "impact": "negative",
            "weight": 1.0,
            "detail": "The drug appears in literature as a cause of this exact condition."
        })
    elif suitability_result["verdict"] == "LIKELY_SUITABLE":
        factors.append({
            "factor": "Positive clinical association found",
            "impact": "positive",
            "weight": 0.7,
            "detail": "The drug is mentioned in literature in connection with this condition, "
                      "without adverse-event evidence for this specific pairing."
        })
    else:
        factors.append({
            "factor": "No direct evidence found",
            "impact": "neutral",
            "weight": 0.0,
            "detail": "No annotated sentence connects this drug and disease in the available corpora."
        })

    common_count = suitability_result.get("common_side_effect_count", 0) or 0
    total_count = suitability_result.get("side_effect_count", 0) or 0

    if total_count > 0:
        burden_ratio = common_count / total_count if total_count else 0
        if burden_ratio > 0.3:
            impact = "negative"
            weight = 0.5
        elif total_count > 50:
            impact = "negative"
            weight = 0.3
        else:
            impact = "neutral"
            weight = 0.1

        factors.append({
            "factor": f"Side-effect burden ({total_count} known effects)",
            "impact": impact,
            "weight": weight,
            "detail": f"{common_count} of {total_count} known side effects are classified as common/occasional."
        })

    if evidence_sentences:
        factors.append({
            "factor": f"Supporting literature ({len(evidence_sentences)} sentence(s))",
            "impact": "informational",
            "weight": 0.2,
            "detail": "See evidence sentences for the exact source text."
        })

    return factors


if __name__ == "__main__":
    fake_suitability = {"verdict": "NO_DIRECT_EVIDENCE", "side_effect_count": 86, "common_side_effect_count": 0}
    breakdown = build_score_breakdown(fake_suitability, [], [])
    for f in breakdown:
        print(f"[{f['impact']}] {f['factor']} (weight={f['weight']})")
        print(f"    {f['detail']}")
