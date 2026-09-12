import sys
sys.path.append("/workspace/models")

from suitability import check_suitability
from alternative_drugs import suggest_alternatives_for_disease
from ranking_engine import get_ranked_side_effects


def analyze(drug_name, disease_name):
    """
    Single entry point: given a drug and a disease, returns:
    1. Suitability verdict
    2. Alternatives (only if not clearly suitable, or always for comparison)
    3. Side effects of the original drug
    """
    suitability = check_suitability(drug_name, disease_name)
    side_effects = get_ranked_side_effects(drug_name)

    alternatives = None
    if suitability["verdict"] in ("NOT_RECOMMENDED", "NO_DIRECT_EVIDENCE"):
        alt_result = suggest_alternatives_for_disease(disease_name, exclude_drug=drug_name, top_n=5)
        alternatives = alt_result

    return {
        "drug": drug_name,
        "disease": disease_name,
        "suitability": suitability,
        "side_effects": side_effects[:15],
        "total_side_effects": len(side_effects),
        "alternatives": alternatives,
    }


if __name__ == "__main__":
    result = analyze("aspirin", "headache")

    print(f"=== Analysis: {result['drug']} for {result['disease']} ===\n")
    print(f"Verdict: {result['suitability']['verdict']}")
    print(f"Reason: {result['suitability']['reason']}\n")

    print(f"Total known side effects: {result['total_side_effects']}")
    for e in result["side_effects"][:5]:
        print(f"  - {e['side_effect']} ({e['severity_bucket']})")

    if result["alternatives"]:
        print("\nAlternatives suggested:")
        if "error" in result["alternatives"]:
            print(f"  {result['alternatives']['error']}")
        else:
            for alt in result["alternatives"]["alternatives"]:
                print(f"  - {alt['drug_name']} (side_effects={alt['side_effect_count']})")
