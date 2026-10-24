"""Bulk-loads the raw SIDER and OnSIDES files into PostgreSQL (12 tables).

WARNING: this DROPS and reloads the tables. It only needs to be run once on a fresh database.
"""
import sys
import time
from pathlib import Path

import psycopg2

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "models"))
import pv_config as cfg  # noqa: E402

SIDER = cfg.RAW_DIR / "sider"
ONSIDES = cfg.RAW_DIR / "onsides"

TABLES = [
    # (table_name, filepath, columns, format, has_header)
    ("sider_drug_names", SIDER / "drug_names.tsv", ["cid", "drug_name"], "text", False),
    ("sider_drug_atc", SIDER / "drug_atc.tsv", ["cid", "atc_code"], "text", False),
    ("sider_side_effects", SIDER / "meddra_all_se.tsv",
     ["cid_flat", "cid_stereo", "umls_concept_label", "meddra_type", "umls_concept_meddra", "side_effect_name"],
     "text", False),
    ("sider_side_effect_freq", SIDER / "meddra_freq.tsv",
     ["cid_flat", "cid_stereo", "umls_concept", "placebo_flag", "freq_description",
      "freq_lower", "freq_upper", "meddra_type", "umls_concept_meddra", "side_effect_name"], "text", False),
    ("onsides_product_label", ONSIDES / "product_label.csv",
     ["label_id", "source", "source_product_name", "source_product_id", "source_label_url"], "csv", True),
    ("onsides_product_adverse_effect", ONSIDES / "product_adverse_effect.csv",
     ["product_label_id", "effect_id", "label_section", "effect_meddra_id", "match_method", "pred0", "pred1"],
     "csv", True),
    ("onsides_product_to_rxnorm", ONSIDES / "product_to_rxnorm.csv", ["label_id", "rxnorm_product_id"], "csv", True),
    ("onsides_vocab_meddra", ONSIDES / "vocab_meddra_adverse_effect.csv",
     ["meddra_id", "meddra_name", "meddra_term_type"], "csv", True),
    ("onsides_vocab_rxnorm_ingredient", ONSIDES / "vocab_rxnorm_ingredient.csv",
     ["rxnorm_id", "rxnorm_name", "rxnorm_term_type"], "csv", True),
    ("onsides_vocab_rxnorm_ingredient_to_product", ONSIDES / "vocab_rxnorm_ingredient_to_product.csv",
     ["product_id", "ingredient_id"], "csv", True),
    ("onsides_vocab_rxnorm_product", ONSIDES / "vocab_rxnorm_product.csv",
     ["rxnorm_id", "rxnorm_name", "rxnorm_term_type"], "csv", True),
    ("onsides_high_confidence", ONSIDES / "high_confidence.csv", ["ingredient_id", "effect_meddra_id"], "csv", True),
]

INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_sider_names_cid ON sider_drug_names(cid);",
    "CREATE INDEX IF NOT EXISTS idx_sider_se_cid ON sider_side_effects(cid_flat);",
    "CREATE INDEX IF NOT EXISTS idx_sider_freq_cid ON sider_side_effect_freq(cid_flat);",
    "CREATE INDEX IF NOT EXISTS idx_onsides_pae_label ON onsides_product_adverse_effect(product_label_id);",
    "CREATE INDEX IF NOT EXISTS idx_onsides_pae_meddra ON onsides_product_adverse_effect(effect_meddra_id);",
    "CREATE INDEX IF NOT EXISTS idx_onsides_p2r_label ON onsides_product_to_rxnorm(label_id);",
]


def load_all():
    conn = psycopg2.connect(**cfg.POSTGRES)
    cur = conn.cursor()
    for table, path, columns, fmt, header in TABLES:
        start = time.time()
        print(f"Loading {table} from {path} ...")
        try:
            cur.execute(f"DROP TABLE IF EXISTS {table};")
            cur.execute(f"CREATE TABLE {table} ({', '.join(f'{chr(34)}{c}{chr(34)} TEXT' for c in columns)});")
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                if fmt == "csv":
                    sql = f"COPY {table} FROM STDIN WITH (FORMAT csv, HEADER {'true' if header else 'false'})"
                else:
                    sql = f"COPY {table} FROM STDIN WITH (FORMAT text)"
                cur.copy_expert(sql, f)
            cur.execute(f"SELECT COUNT(*) FROM {table};")
            count = cur.fetchone()[0]
            conn.commit()
            print(f"  -> {count:,} rows ({time.time() - start:.1f}s)")
        except Exception as e:  # noqa: BLE001
            conn.rollback()
            print(f"  !! FAILED loading {table}: {e}")
    for stmt in INDEXES:
        cur.execute(stmt)
    conn.commit()
    cur.close()
    conn.close()
    print("All done.")


if __name__ == "__main__":
    load_all()
