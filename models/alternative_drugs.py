import psycopg2
import sys
sys.path.append("/workspace/models")

from ranking_engine import get_ranked_side_effects

DB_CONFIG = dict(host="postgres", dbname="drug_safety_db", user="drug_user", password="drug_password")
ATC_PREFIX_LENGTH = 4


def get_atc_codes(drug_name):
    drug_name = drug_name.strip().lower()
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute("""
        SELECT a.atc_code
        FROM sider_drug_atc a
        JOIN sider_drug_names n ON a.cid = n.cid
        WHERE LOWER(TRIM(n.drug_name)) = %s
    """, (drug_name,))
    codes = [row[0] for row in cur.fetchall()]
    cur.close()
    conn.close()
    return codes


def get_drugs_in_same_class(atc_code, exclude_drug=None):
    prefix = atc_code[:ATC_PREFIX_LENGTH]
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute("""
        SELECT DISTINCT n.drug_name
        FROM sider_drug_atc a
        JOIN sider_drug_names n ON a.cid = n.cid
        WHERE a.atc_code LIKE %s
    """, (prefix + '%',))
    drugs = [row[0] for row in cur.fetchall()]
    cur.close()
    conn.close()

    if exclude_drug:
        drugs = [d for d in drugs if d.strip().lower() != exclude_drug.strip().lower()]
    return drugs


def suggest_alternatives(drug_name, top_n=5):
    atc_codes = get_atc_codes(drug_name)
    if not atc_codes:
        return {"error": f"No ATC classification found for '{drug_name}', cannot suggest alternatives."}

    candidates = set()
    for code in atc_codes:
        candidates.update(get_drugs_in_same_class(code, exclude_drug=drug_name))

    if not candidates:
        return {"error": f"No other drugs found in the same ATC class as '{drug_name}'."}

    original_effects = get_ranked_side_effects(drug_name)
    original_count = len(original_effects)

    scored_candidates = []
    for candidate in candidates:
        candidate_effects = get_ranked_side_effects(candidate)
        scored_candidates.append({
            "drug_name": candidate,
            "side_effect_count": len(candidate_effects),
            "safer_than_original": len(candidate_effects) < original_count
        })

    scored_candidates.sort(key=lambda x: x["side_effect_count"])

    return {
        "original_drug": drug_name,
        "original_side_effect_count": original_count,
        "shared_atc_codes": atc_codes,
        "candidates_checked": len(candidates),
        "alternatives": scored_candidates[:top_n]
    }


if __name__ == "__main__":
    test_drug = "aspirin"
    result = suggest_alternatives(test_drug, top_n=5)

    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        print(f"Alternatives for {test_drug}:")
        print(f"Original side effect count: {result['original_side_effect_count']}")
        print(f"Shared ATC codes: {result['shared_atc_codes']}")
        print(f"Candidates checked: {result['candidates_checked']}\n")
        for alt in result["alternatives"]:
            print(f"  {alt['drug_name']:<30} side_effects={alt['side_effect_count']:<5} safer={alt['safer_than_original']}")
