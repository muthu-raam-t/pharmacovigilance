"""End-to-end analysis of one drug-disease pair (used by the API)."""
from alternative_drugs import suggest_alternatives_for_disease
from evidence_retrieval import find_evidence_sentences
from explanation_generator import generate_explanation
from ranking_engine import get_ranked_side_effects
from score_breakdown import build_score_breakdown
from suitability import check_suitability

ALTERNATIVE_VERDICTS = ("NOT_RECOMMENDED", "NO_DIRECT_EVIDENCE", "MIXED_EVIDENCE", "ASSOCIATION_ONLY")


def analyze(drug_name, disease_name, max_side_effects=15):
    side_effects = get_ranked_side_effects(drug_name)
    suitability = check_suitability(drug_name, disease_name, side_effects=side_effects)
    evidence = find_evidence_sentences(drug_name, disease_name)
    breakdown = build_score_breakdown(suitability, side_effects, evidence)
    explanation = generate_explanation(drug_name, disease_name, suitability, breakdown, evidence, side_effects)

    alternatives = None
    if suitability["verdict"] in ALTERNATIVE_VERDICTS:
        alternatives = suggest_alternatives_for_disease(disease_name, exclude_drug=drug_name, top_n=5)

    return {
        "drug": drug_name.strip(),
        "disease": disease_name.strip(),
        "suitability": suitability,
        "score_breakdown": breakdown,
        "evidence_sentences": evidence,
        "explanation": explanation,
        "side_effects": side_effects[:max_side_effects],
        "total_side_effects": len(side_effects),
        "alternatives": alternatives,
    }


if __name__ == "__main__":
    result = analyze("aspirin", "headache")
    print(result["explanation"])
    if result["alternatives"] and "alternatives" in result["alternatives"]:
        print("\nAlternatives:")
        for alt in result["alternatives"]["alternatives"]:
            print(f"  - {alt['drug_name']} ({alt['source']}, burden {alt['burden_score']})")
