import psycopg2

DB_CONFIG = dict(host="postgres", dbname="drug_safety_db", user="drug_user", password="drug_password")


def get_severity_scores_for_drug(drug_name):
    """
    Returns a list of dicts: side_effect, frequency (0-1 float or None), source
    Uses SIDER frequency data when available; falls back to unranked
    side-effect list (frequency=None) when no frequency data exists for
    this drug, since SIDER's frequency table covers only a subset of drugs.
    """
    drug_name = drug_name.strip().lower()
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    freq_query = """
    SELECT
        LOWER(TRIM(f.side_effect_name)) AS side_effect,
        MAX(CAST(NULLIF(f.freq_upper, '') AS FLOAT)) AS max_freq
    FROM sider_side_effect_freq f
    JOIN sider_drug_names n ON f.cid_flat = n.cid
    WHERE LOWER(TRIM(n.drug_name)) = %s
    GROUP BY LOWER(TRIM(f.side_effect_name))
    ORDER BY max_freq DESC NULLS LAST
    """
    cur.execute(freq_query, (drug_name,))
    freq_rows = cur.fetchall()

    results = []
    if freq_rows:
        for side_effect, freq in freq_rows:
            results.append({
                "side_effect": side_effect,
                "frequency": freq if freq is not None else None,
                "source": "SIDER (frequency data)"
            })
    else:
        fallback_query = """
        SELECT DISTINCT LOWER(TRIM(se.side_effect_name)) AS side_effect
        FROM sider_side_effects se
        JOIN sider_drug_names n ON se.cid_flat = n.cid
        WHERE LOWER(TRIM(n.drug_name)) = %s
        """
        cur.execute(fallback_query, (drug_name,))
        for (side_effect,) in cur.fetchall():
            results.append({
                "side_effect": side_effect,
                "frequency": None,
                "source": "SIDER (no frequency data available)"
            })

    cur.close()
    conn.close()
    return results


def compute_severity_score(frequency):
    """
    Simple severity bucket based on reported frequency.
    Returns 'unknown' when no frequency data exists for that side effect.
    """
    if frequency is None:
        return "unknown", None
    elif frequency >= 0.10:
        return "common", frequency
    elif frequency >= 0.01:
        return "occasional", frequency
    elif frequency > 0.0:
        return "rare", frequency
    else:
        return "unknown", 0.0


if __name__ == "__main__":
    test_drug = "aspirin"
    scores = get_severity_scores_for_drug(test_drug)
    print(f"Found {len(scores)} scored side effects for {test_drug}")
    for s in scores[:10]:
        bucket, freq = compute_severity_score(s["frequency"])
        freq_display = f"{freq:.3f}" if freq is not None else "N/A"
        print(f"  {s['side_effect']}: {freq_display} ({bucket}) [{s['source']}]")
