import sys
sys.path.append("/workspace/models")

from suitability import check_suitability
from alternative_drugs import suggest_alternatives_for_disease
from ranking_engine import get_ranked_side_effects
from evidence_retrieval import find_evidence_sentences
from score_breakdown import build_score_breakdown
from explanation_generator import generate_explanation


def analyze(drug_name, disease_name):
    suitability = check_suitability(drug_name, disease_name)
    side_effects = get_ranked_side_effects(drug_name)
    evidence = find_evidence_sentences(drug_name, disease_name)
    breakdown = build_score_breakdown(suitability, side_effects, evidence)
    explanation = generate_explanation(drug_name, disease_name, suitability, breakdown, evidence)

    alternatives = None
    if suitability["verdict"] in ("NOT_RECOMMENDED", "NO_DIRECT_EVIDENCE"):
        alternatives = suggest_alternatives_for_disease(disease_name, exclude_drug=drug_name, top_n=5)

    return {
        "drug": drug_name,
        "disease": disease_name,
        "suitability": suitability,
        "score_breakdown": breakdown,
        "evidence_sentences": evidence,
        "explanation": explanation,
        "side_effects": side_effects[:15],
        "total_side_effects": len(side_effects),
        "alternatives": alternatives,
    }


if __name__ == "__main__":
    result = analyze("aspirin", "headache")
    print(result["explanation"])
    print("\n" + "=" * 50)
    print(f"\nTotal known side effects: {result['total_side_effects']}")
    if result["alternatives"] and "alternatives" in result["alternatives"]:
        print("\nAlternatives suggested:")
        for alt in result["alternatives"]["alternatives"]:
            print(f"  - {alt['drug_name']} (side_effects={alt['side_effect_count']})")
