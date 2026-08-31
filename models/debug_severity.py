import psycopg2

DB_CONFIG = dict(host="postgres", dbname="drug_safety_db", user="drug_user", password="drug_password")

conn = psycopg2.connect(**DB_CONFIG)
cur = conn.cursor()

print("1. Does 'aspirin' exist in sider_drug_names?")
cur.execute("SELECT cid, drug_name FROM sider_drug_names WHERE LOWER(TRIM(drug_name)) = 'aspirin'")
print(cur.fetchall())

print("\n2. Any rows in sider_side_effect_freq at all?")
cur.execute("SELECT COUNT(*) FROM sider_side_effect_freq")
print(cur.fetchone())

print("\n3. Sample rows from sider_side_effect_freq:")
cur.execute("SELECT cid_flat, side_effect_name, freq_upper FROM sider_side_effect_freq LIMIT 5")
for row in cur.fetchall():
    print(row)

print("\n4. Does aspirin's CID appear in sider_side_effect_freq?")
cur.execute("""
    SELECT n.cid, n.drug_name, COUNT(f.side_effect_name)
    FROM sider_drug_names n
    LEFT JOIN sider_side_effect_freq f ON n.cid = f.cid_flat
    WHERE LOWER(TRIM(n.drug_name)) = 'aspirin'
    GROUP BY n.cid, n.drug_name
""")
print(cur.fetchall())

cur.close()
conn.close()
