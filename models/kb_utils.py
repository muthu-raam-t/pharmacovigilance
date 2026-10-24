"""Drug look-ups against the structured knowledge base (SIDER and OnSIDES)."""
from functools import lru_cache

from db import pg_query, table_exists
from text_utils import normalize_name


@lru_cache(maxsize=4096)
def is_known_drug_in_sider(drug_name):
    rows = pg_query("SELECT 1 FROM sider_drug_names WHERE LOWER(TRIM(drug_name)) = %s LIMIT 1",
                    (normalize_name(drug_name),))
    return bool(rows)


@lru_cache(maxsize=4096)
def is_known_drug_in_onsides(drug_name):
    if not table_exists("onsides_drug_label_counts"):
        return False
    rows = pg_query("SELECT 1 FROM onsides_drug_label_counts WHERE drug_name = %s LIMIT 1",
                    (normalize_name(drug_name),))
    return bool(rows)


def is_known_drug(drug_name):
    """True when the drug exists in SIDER or in the OnSIDES label data."""
    return is_known_drug_in_sider(drug_name) or is_known_drug_in_onsides(drug_name)


def side_effect_sources_for(drug_name):
    sources = []
    if is_known_drug_in_sider(drug_name):
        sources.append("SIDER")
    if is_known_drug_in_onsides(drug_name):
        sources.append("OnSIDES")
    return sources
