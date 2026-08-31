import psycopg2

DB_CONFIG = dict(host="postgres", dbname="drug_safety_db", user="drug_user", password="drug_password")


def get_severity_scores_for_drug(drug_name):
    """
    Returns a list of dicts: side_effect, frequency (0-1 float), source
    Ranked by frequency descending (most common side effects first).
    """
    drug_name = drug_name.strip().lower()
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    query = """
    SELECT
        LOWER(TRIM(f.side_effect_name)) AS side_effect,
        MAX(CAST(NULLIF(f.freq_upper, '') AS FLOAT)) AS max_freq
    FROM sider_side_effect_freq f
    JOIN sider_drug_names n ON f.cid_flat = n.cid
    WHERE LOWER(TRIM(n.drug_name)) = %s
    GROUP BY LOWER(TRIM(f.side_effect_name))
    ORDER BY max_freq DESC NULLS LAST
    """
    cur.execute(query, (drug_name,))
    rows = cur.fetchall()

    results = []
    for side_effect, freq in rows:
        results.append({
            "side_effect": side_effect,
            "frequency": freq if freq is not None else 0.0,
            "source": "SIDER"
        })

    cur.close()
    conn.close()
    return results


def compute_severity_score(frequency):
    """
    Simple severity bucket based on reported frequency.
    High frequency = more common = weighted higher for visibility.
    """
    if frequency >= 0.10:
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
        print(f"  {s['side_effect']}: {freq:.3f} ({bucket})")
