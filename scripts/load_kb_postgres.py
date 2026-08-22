import psycopg2
import time

DB_CONFIG = dict(
    host="postgres",
    dbname="drug_safety_db",
    user="drug_user",
    password="drug_password"
)

TABLES = [
    # (table_name, filepath, columns, delimiter_format, has_header)
    (
        "sider_drug_names",
        "/workspace/data/raw/sider/drug_names.tsv",
        ["cid", "drug_name"],
        "text", False
    ),
    (
        "sider_drug_atc",
        "/workspace/data/raw/sider/drug_atc.tsv",
        ["cid", "atc_code"],
        "text", False
    ),
    (
        "sider_side_effects",
        "/workspace/data/raw/sider/meddra_all_se.tsv",
        ["cid_flat", "cid_stereo", "umls_concept_label", "meddra_type", "umls_concept_meddra", "side_effect_name"],
        "text", False
    ),
    (
        "sider_side_effect_freq",
        "/workspace/data/raw/sider/meddra_freq.tsv",
        ["cid_flat", "cid_stereo", "umls_concept", "placebo_flag", "freq_description",
         "freq_lower", "freq_upper", "meddra_type", "umls_concept_meddra", "side_effect_name"],
        "text", False
    ),
    (
        "onsides_product_label",
        "/workspace/data/raw/onsides/product_label.csv",
        ["label_id", "source", "source_product_name", "source_product_id", "source_label_url"],
        "csv", True
    ),
    (
        "onsides_product_adverse_effect",
        "/workspace/data/raw/onsides/product_adverse_effect.csv",
        ["product_label_id", "effect_id", "label_section", "effect_meddra_id", "match_method", "pred0", "pred1"],
        "csv", True
    ),
    (
        "onsides_product_to_rxnorm",
        "/workspace/data/raw/onsides/product_to_rxnorm.csv",
        ["label_id", "rxnorm_product_id"],
        "csv", True
    ),
    (
        "onsides_vocab_meddra",
        "/workspace/data/raw/onsides/vocab_meddra_adverse_effect.csv",
        ["meddra_id", "meddra_name", "meddra_term_type"],
        "csv", True
    ),
    (
        "onsides_vocab_rxnorm_ingredient",
        "/workspace/data/raw/onsides/vocab_rxnorm_ingredient.csv",
        ["rxnorm_id", "rxnorm_name", "rxnorm_term_type"],
        "csv", True
    ),
    (
        "onsides_vocab_rxnorm_ingredient_to_product",
        "/workspace/data/raw/onsides/vocab_rxnorm_ingredient_to_product.csv",
        ["product_id", "ingredient_id"],
        "csv", True
    ),
    (
        "onsides_vocab_rxnorm_product",
        "/workspace/data/raw/onsides/vocab_rxnorm_product.csv",
        ["rxnorm_id", "rxnorm_name", "rxnorm_term_type"],
        "csv", True
    ),
    (
        "onsides_high_confidence",
        "/workspace/data/raw/onsides/high_confidence.csv",
        ["ingredient_id", "effect_meddra_id"],
        "csv", True
    ),
]


def create_table(cur, table_name, columns):
    cur.execute(f"DROP TABLE IF EXISTS {table_name};")
    col_defs = ", ".join([f'"{c}" TEXT' for c in columns])
    cur.execute(f"CREATE TABLE {table_name} ({col_defs});")


def copy_file_into_table(cur, table_name, filepath, fmt, has_header):
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        if fmt == "csv":
            header_clause = "HEADER true" if has_header else "HEADER false"
            sql = f"COPY {table_name} FROM STDIN WITH (FORMAT csv, {header_clause})"
        else:
            sql = f"COPY {table_name} FROM STDIN WITH (FORMAT text)"
        cur.copy_expert(sql, f)


def load_all():
    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = False
    cur = conn.cursor()

    for table_name, filepath, columns, fmt, has_header in TABLES:
        start = time.time()
        print(f"Loading {table_name} from {filepath} ...")
        try:
            create_table(cur, table_name, columns)
            copy_file_into_table(cur, table_name, filepath, fmt, has_header)
            cur.execute(f"SELECT COUNT(*) FROM {table_name};")
            count = cur.fetchone()[0]
            conn.commit()
            elapsed = time.time() - start
            print(f"  -> {count} rows loaded into {table_name} ({elapsed:.1f}s)")
        except Exception as e:
            conn.rollback()
            print(f"  !! FAILED loading {table_name}: {e}")

    # Helpful indexes for later evidence-fusion queries
    index_statements = [
        "CREATE INDEX IF NOT EXISTS idx_sider_names_cid ON sider_drug_names(cid);",
        "CREATE INDEX IF NOT EXISTS idx_sider_se_cid ON sider_side_effects(cid_flat);",
        "CREATE INDEX IF NOT EXISTS idx_sider_freq_cid ON sider_side_effect_freq(cid_flat);",
        "CREATE INDEX IF NOT EXISTS idx_onsides_pae_label ON onsides_product_adverse_effect(product_label_id);",
        "CREATE INDEX IF NOT EXISTS idx_onsides_pae_meddra ON onsides_product_adverse_effect(effect_meddra_id);",
        "CREATE INDEX IF NOT EXISTS idx_onsides_p2r_label ON onsides_product_to_rxnorm(label_id);",
    ]
    for stmt in index_statements:
        cur.execute(stmt)
    conn.commit()
    print("Indexes created.")

    cur.close()
    conn.close()
    print("All done.")


if __name__ == "__main__":
    load_all()
