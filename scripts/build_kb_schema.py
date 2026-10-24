"""Builds the derived knowledge-base tables used by the reasoning engine.

  drug_side_effects_clean     SIDER drug -> side effect, MedDRA preferred terms only (no LLT duplicates)
  onsides_drug_effects        OnSIDES drug -> adverse effect, aggregated over single-ingredient product labels,
                              with the number of labels and whether it appears in a boxed warning
  onsides_drug_label_counts   number of single-ingredient labels per drug (denominator for label fractions)
"""
import sys
import time
from pathlib import Path

import psycopg2

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "models"))
import pv_config as cfg  # noqa: E402

STEPS = [
    ("Indexes for OnSIDES joins", """
        CREATE INDEX IF NOT EXISTS idx_onsides_p2r_product ON onsides_product_to_rxnorm(rxnorm_product_id);
        CREATE INDEX IF NOT EXISTS idx_onsides_i2p_product ON onsides_vocab_rxnorm_ingredient_to_product(product_id);
        CREATE INDEX IF NOT EXISTS idx_onsides_i2p_ingredient ON onsides_vocab_rxnorm_ingredient_to_product(ingredient_id);
        CREATE INDEX IF NOT EXISTS idx_onsides_ingredient_id ON onsides_vocab_rxnorm_ingredient(rxnorm_id);
        CREATE INDEX IF NOT EXISTS idx_onsides_meddra_id ON onsides_vocab_meddra(meddra_id);
        CREATE INDEX IF NOT EXISTS idx_sider_freq_type ON sider_side_effect_freq(cid_flat, meddra_type);
    """),
    ("drug_side_effects_clean (SIDER, preferred terms)", """
        DROP TABLE IF EXISTS drug_side_effects_clean;
        CREATE TABLE drug_side_effects_clean AS
        SELECT DISTINCT
            LOWER(TRIM(n.drug_name))         AS drug_name,
            LOWER(TRIM(se.side_effect_name)) AS side_effect_name,
            'SIDER'                          AS source
        FROM sider_side_effects se
        JOIN sider_drug_names n ON se.cid_flat = n.cid
        WHERE se.side_effect_name IS NOT NULL AND n.drug_name IS NOT NULL
          AND se.meddra_type = 'PT';
        CREATE INDEX idx_dsec_drug ON drug_side_effects_clean(drug_name);
        CREATE INDEX idx_dsec_effect ON drug_side_effects_clean(side_effect_name);
    """),
    ("onsides_single_ingredient_labels", """
        DROP TABLE IF EXISTS onsides_single_ingredient_labels;
        CREATE TABLE onsides_single_ingredient_labels AS
        WITH single_products AS (
            SELECT product_id, MIN(ingredient_id) AS ingredient_id
            FROM onsides_vocab_rxnorm_ingredient_to_product
            GROUP BY product_id
            HAVING COUNT(DISTINCT ingredient_id) = 1
        )
        SELECT DISTINCT pr.label_id, LOWER(TRIM(i.rxnorm_name)) AS drug_name
        FROM onsides_product_to_rxnorm pr
        JOIN single_products sp ON sp.product_id = pr.rxnorm_product_id
        JOIN onsides_vocab_rxnorm_ingredient i ON i.rxnorm_id = sp.ingredient_id;
        CREATE INDEX idx_osil_label ON onsides_single_ingredient_labels(label_id);
    """),
    ("onsides_drug_label_counts", """
        DROP TABLE IF EXISTS onsides_drug_label_counts;
        CREATE TABLE onsides_drug_label_counts AS
        SELECT drug_name, COUNT(DISTINCT label_id) AS n_labels
        FROM onsides_single_ingredient_labels
        GROUP BY drug_name;
        CREATE INDEX idx_odlc_drug ON onsides_drug_label_counts(drug_name);
    """),
    ("onsides_drug_effects (aggregating ~28M label rows, takes a few minutes)", """
        DROP TABLE IF EXISTS onsides_drug_effects;
        CREATE TABLE onsides_drug_effects AS
        SELECT l.drug_name,
               LOWER(TRIM(m.meddra_name))             AS side_effect_name,
               COUNT(DISTINCT pae.product_label_id)   AS n_labels,
               BOOL_OR(UPPER(pae.label_section) IN ('BW', 'BOXED WARNING', 'BOXED_WARNING')) AS boxed_warning
        FROM onsides_product_adverse_effect pae
        JOIN onsides_single_ingredient_labels l ON l.label_id = pae.product_label_id
        JOIN onsides_vocab_meddra m ON m.meddra_id = pae.effect_meddra_id
        GROUP BY l.drug_name, LOWER(TRIM(m.meddra_name));
        CREATE INDEX idx_ode_drug ON onsides_drug_effects(drug_name);
    """),
]

CHECKS = [
    ("drug_side_effects_clean rows", "SELECT COUNT(*) FROM drug_side_effects_clean"),
    ("SIDER drugs with side effects", "SELECT COUNT(DISTINCT drug_name) FROM drug_side_effects_clean"),
    ("OnSIDES single-ingredient labels", "SELECT COUNT(*) FROM onsides_single_ingredient_labels"),
    ("OnSIDES drugs", "SELECT COUNT(*) FROM onsides_drug_label_counts"),
    ("onsides_drug_effects rows", "SELECT COUNT(*) FROM onsides_drug_effects"),
    ("OnSIDES boxed-warning drug-effect pairs", "SELECT COUNT(*) FROM onsides_drug_effects WHERE boxed_warning"),
    ("Drugs in both SIDER and OnSIDES", """SELECT COUNT(*) FROM (SELECT DISTINCT drug_name FROM drug_side_effects_clean) s
                                           JOIN onsides_drug_label_counts o USING (drug_name)"""),
]

if __name__ == "__main__":
    conn = psycopg2.connect(**cfg.POSTGRES)
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute("SET work_mem = '256MB'")

    cur.execute("SELECT label_section, COUNT(*) FROM onsides_product_adverse_effect GROUP BY 1 ORDER BY 2 DESC LIMIT 10")
    print("OnSIDES label sections found:", cur.fetchall())

    for title, sql in STEPS:
        t0 = time.time()
        print(f"Building {title} ...", flush=True)
        cur.execute(sql)
        print(f"   done in {time.time() - t0:.1f}s", flush=True)

    print("\n=== Knowledge-base summary ===")
    for label, sql in CHECKS:
        cur.execute(sql)
        print(f"{label:<42} {cur.fetchone()[0]:>12,}")

    cur.close()
    conn.close()
