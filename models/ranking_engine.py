import sys
sys.path.append("/workspace/models")

from kg_query import KnowledgeGraphClient
from severity_scoring import get_severity_scores_for_drug, compute_severity_score


SEVERITY_WEIGHTS = {"common": 3, "occasional": 2, "rare": 1, "unknown": 0}


def get_ranked_side_effects(drug_name):
    """
    Combines graph-confirmed side effects with SIDER frequency data.
    Returns a ranked list: most severe/common side effects first,
    unknown-frequency effects last (still included, just deprioritized).
    """
    kg = KnowledgeGraphClient()
    graph_confirmed = set(kg.get_side_effects_for_drug(drug_name))
    kg.close()

    scored = get_severity_scores_for_drug(drug_name)

    ranked = []
    for entry in scored:
        bucket, freq = compute_severity_score(entry["frequency"])
        in_graph = entry["side_effect"] in graph_confirmed
        ranked.append({
            "side_effect": entry["side_effect"],
            "frequency": freq,
            "severity_bucket": bucket,
            "severity_weight": SEVERITY_WEIGHTS[bucket],
            "confirmed_in_graph": in_graph,
            "source": entry["source"]
        })

    ranked.sort(key=lambda x: (x["severity_weight"], x["frequency"] or 0), reverse=True)
    return ranked


if __name__ == "__main__":
    test_drug = "aspirin"
    results = get_ranked_side_effects(test_drug)
    print(f"Ranked {len(results)} side effects for {test_drug}\n")
    for r in results[:15]:
        freq_display = f"{r['frequency']:.3f}" if r["frequency"] is not None else "N/A"
        print(f"  {r['side_effect']:<30} severity={r['severity_bucket']:<10} freq={freq_display:<6} graph_confirmed={r['confirmed_in_graph']}")
