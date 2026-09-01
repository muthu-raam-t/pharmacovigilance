import psycopg2

DB_CONFIG = dict(host="postgres", dbname="drug_safety_db", user="drug_user", password="drug_password")
conn = psycopg2.connect(**DB_CONFIG)
cur = conn.cursor()

cur.execute("SELECT cid, drug_name FROM sider_drug_names WHERE LOWER(TRIM(drug_name)) = 'aspirin'")
print("Aspirin CID:", cur.fetchall())

cur.execute("SELECT COUNT(*) FROM sider_drug_atc")
print("Total ATC rows:", cur.fetchone())

cur.execute("SELECT * FROM sider_drug_atc WHERE cid = 'CID100002244'")
print("Aspirin's ATC rows:", cur.fetchall())

cur.close()
conn.close()
