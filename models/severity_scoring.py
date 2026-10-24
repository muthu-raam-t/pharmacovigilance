"""Side-effect evidence from two independent structured sources.

SIDER   : side effects + reported frequency (placebo arms excluded, MedDRA preferred terms only)
OnSIDES : adverse effects extracted from FDA drug labels (single-ingredient products only),
          including whether the effect appears in a boxed warning
"""
from db import pg_query, table_exists
from text_utils import normalize_name

SIDER_FREQ_SQL = """
SELECT LOWER(TRIM(f.side_effect_name)) AS side_effect,
       MAX(CAST(NULLIF(TRIM(f.freq_upper), '') AS FLOAT)) AS max_freq
FROM sider_side_effect_freq f
JOIN sider_drug_names n ON f.cid_flat = n.cid
WHERE LOWER(TRIM(n.drug_name)) = %s
  AND f.meddra_type = 'PT'
  AND COALESCE(TRIM(f.placebo_flag), '') <> 'placebo'
GROUP BY LOWER(TRIM(f.side_effect_name))
"""

SIDER_ALL_SQL = "SELECT side_effect_name FROM drug_side_effects_clean WHERE drug_name = %s"

ONSIDES_SQL = """
SELECT e.side_effect_name, e.n_labels, e.boxed_warning, c.n_labels AS total_labels
FROM onsides_drug_effects e
JOIN onsides_drug_label_counts c ON c.drug_name = e.drug_name
WHERE e.drug_name = %s
"""


def get_sider_effects(drug_name):
    drug = normalize_name(drug_name)
    freq = {se: f for se, f in pg_query(SIDER_FREQ_SQL, (drug,))}
    names = {row[0] for row in pg_query(SIDER_ALL_SQL, (drug,))} if table_exists("drug_side_effects_clean") else set()
    names |= set(freq)
    return {se: freq.get(se) for se in names}


def get_onsides_effects(drug_name):
    if not (table_exists("onsides_drug_effects") and table_exists("onsides_drug_label_counts")):
        return {}
    out = {}
    for se, n_labels, boxed, total in pg_query(ONSIDES_SQL, (normalize_name(drug_name),)):
        out[se] = {"label_fraction": (n_labels / total) if total else None, "boxed_warning": bool(boxed),
                   "n_labels": n_labels}
    return out


def get_severity_scores_for_drug(drug_name):
    """
    Returns one entry per side effect:
      side_effect, frequency (0-1 or None), sources (list), boxed_warning (bool), label_fraction (0-1 or None)
    """
    sider = get_sider_effects(drug_name)
    onsides = get_onsides_effects(drug_name)
    results = []
    for se in sorted(set(sider) | set(onsides)):
        sources = []
        if se in sider:
            sources.append("SIDER")
        if se in onsides:
            sources.append("OnSIDES")
        o = onsides.get(se, {})
        results.append({
            "side_effect": se,
            "frequency": sider.get(se),
            "sources": sources,
            "boxed_warning": o.get("boxed_warning", False),
            "label_fraction": o.get("label_fraction"),
        })
    return results


def compute_severity_score(frequency):
    """Frequency bucket (MedDRA / CIOMS convention). Returns (bucket, frequency)."""
    if frequency is None or frequency <= 0.0:
        return "unknown", frequency
    if frequency >= 0.10:
        return "common", frequency
    if frequency >= 0.01:
        return "occasional", frequency
    return "rare", frequency


if __name__ == "__main__":
    scores = get_severity_scores_for_drug("aspirin")
    print(f"Found {len(scores)} side effects for aspirin")
    for s in scores[:10]:
        bucket, freq = compute_severity_score(s["frequency"])
        print(f"  {s['side_effect']}: {freq if freq is not None else 'N/A'} ({bucket}) {s['sources']}"
              f"{' BOXED WARNING' if s['boxed_warning'] else ''}")
