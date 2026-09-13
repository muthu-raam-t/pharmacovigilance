import psycopg2

DB_CONFIG = dict(host="postgres", dbname="drug_safety_db", user="drug_user", password="drug_password")


def is_known_drug_in_sider(drug_name):
    drug_name = drug_name.strip().lower()
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM sider_drug_names WHERE LOWER(TRIM(drug_name)) = %s", (drug_name,))
    count = cur.fetchone()[0]
    cur.close()
    conn.close()
    return count > 0
