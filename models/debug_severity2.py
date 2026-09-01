import psycopg2

DB_CONFIG = dict(host="postgres", dbname="drug_safety_db", user="drug_user", password="drug_password")

conn = psycopg2.connect(**DB_CONFIG)
cur = conn.cursor()

print("1. Exact search for aspirin's CID in freq table:")
cur.execute("SELECT COUNT(*) FROM sider_side_effect_freq WHERE cid_flat = 'CID100002244'")
print(cur.fetchone())

print("\n2. How many unique drugs actually have frequency data at all?")
cur.execute("SELECT COUNT(DISTINCT cid_flat) FROM sider_side_effect_freq")
print(cur.fetchone())

print("\n3. Pick a drug that DOES have frequency data, to confirm the join logic works:")
cur.execute("""
    SELECT n.drug_name, COUNT(f.side_effect_name)
    FROM sider_drug_names n
    JOIN sider_side_effect_freq f ON n.cid = f.cid_flat
    GROUP BY n.drug_name
    ORDER BY COUNT(f.side_effect_name) DESC
    LIMIT 5
""")
print(cur.fetchall())

cur.close()
conn.close()
