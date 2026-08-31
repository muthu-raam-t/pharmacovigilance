import psycopg2

DB_CONFIG = dict(host="postgres", dbname="drug_safety_db", user="drug_user", password="drug_password")

CREATE_DRUG_SIDE_EFFECTS = """
DROP TABLE IF EXISTS drug_side_effects_clean;
CREATE TABLE drug_side_effects_clean AS
SELECT DISTINCT
    LOWER(TRIM(n.drug_name)) AS drug_name,
    LOWER(TRIM(se.side_effect_name)) AS side_effect_name,
    'SIDER' AS source
FROM sider_side_effects se
JOIN sider_drug_names n ON se.cid_flat = n.cid
WHERE se.side_effect_name IS NOT NULL AND n.drug_name IS NOT NULL;
"""

CREATE_INDEX = """
CREATE INDEX IF NOT EXISTS idx_dsec_drug ON drug_side_effects_clean(drug_name);
CREATE INDEX IF NOT EXISTS idx_dsec_effect ON drug_side_effects_clean(side_effect_name);
"""

if __name__ == "__main__":
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    print("Building drug_side_effects_clean from SIDER...")
    cur.execute(CREATE_DRUG_SIDE_EFFECTS)
    conn.commit()

    cur.execute(CREATE_INDEX)
    conn.commit()

    cur.execute("SELECT COUNT(*) FROM drug_side_effects_clean;")
    count = cur.fetchone()[0]
    print(f"drug_side_effects_clean: {count} rows")

    cur.close()
    conn.close()
