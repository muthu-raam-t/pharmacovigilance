"""Shared, pooled database connections (one PostgreSQL pool and one Neo4j driver per process)."""
import threading
from contextlib import contextmanager
from functools import lru_cache

import psycopg2
from psycopg2 import pool
from neo4j import GraphDatabase

import pv_config as cfg

_lock = threading.Lock()
_pg_pool = None
_neo4j_driver = None


def get_pg_pool():
    global _pg_pool
    if _pg_pool is None:
        with _lock:
            if _pg_pool is None:
                _pg_pool = pool.ThreadedConnectionPool(1, 10, **cfg.POSTGRES)
    return _pg_pool


@contextmanager
def pg_cursor():
    p = get_pg_pool()
    conn = p.getconn()
    broken = False
    try:
        with conn.cursor() as cur:
            yield cur
        conn.commit()
    except (psycopg2.OperationalError, psycopg2.InterfaceError):
        broken = True
        raise
    except Exception:
        conn.rollback()
        raise
    finally:
        p.putconn(conn, close=broken)


def pg_query(sql, params=()):
    with pg_cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchall()


@lru_cache(maxsize=64)
def table_exists(table_name):
    rows = pg_query("SELECT to_regclass(%s) IS NOT NULL", (f"public.{table_name}",))
    return bool(rows and rows[0][0])


def get_neo4j_driver():
    global _neo4j_driver
    if _neo4j_driver is None:
        with _lock:
            if _neo4j_driver is None:
                _neo4j_driver = GraphDatabase.driver(cfg.NEO4J_URI, auth=cfg.NEO4J_AUTH)
    return _neo4j_driver


def neo4j_query(cypher, **params):
    with get_neo4j_driver().session() as session:
        return [record.data() for record in session.run(cypher, **params)]


def check_connections():
    status = {}
    try:
        pg_query("SELECT 1")
        status["postgres"] = "ok"
    except Exception as e:  # noqa: BLE001
        status["postgres"] = f"error: {type(e).__name__}"
    try:
        neo4j_query("RETURN 1 AS ok")
        status["neo4j"] = "ok"
    except Exception as e:  # noqa: BLE001
        status["neo4j"] = f"error: {type(e).__name__}"
    return status
