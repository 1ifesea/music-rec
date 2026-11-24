"""数据库访问层：基于 PyMySQL 的轻量连接池。

不引入 ORM，理由是课程项目要求"能讲清楚每条 SQL"，
参数化查询直接可见，答辩时容易解释，也避免 ORM 版本问题。
"""
import queue
import threading
from contextlib import contextmanager

import pymysql
from pymysql.cursors import DictCursor

from .config import settings

_pool: "queue.LifoQueue" = queue.LifoQueue(maxsize=8)
_local = threading.local()


def _new_conn(with_db: bool = True):
    """建立一条新连接。with_db=False 用于建库阶段。"""
    return pymysql.connect(
        host=settings.DB_HOST,
        port=settings.DB_PORT,
        user=settings.DB_USER,
        password=settings.DB_PASSWORD,
        database=settings.DB_NAME if with_db else None,
        charset="utf8mb4",
        cursorclass=DictCursor,
        autocommit=True,
        connect_timeout=5,
    )


def acquire():
    """从池里取一条连接，池空则新建。"""
    try:
        return _pool.get_nowait()
    except queue.Empty:
        return _new_conn()


def release(conn) -> None:
    """归还连接；池满则直接关闭。"""
    try:
        _pool.put_nowait(conn)
    except queue.Full:
        try:
            conn.close()
        except Exception:
            pass


@contextmanager
def cursor():
    """with 语法取游标，异常时丢弃该连接，避免污染连接池。"""
    conn = acquire()
    try:
        with conn.cursor() as cur:
            yield cur
        release(conn)
    except Exception:
        try:
            conn.close()
        except Exception:
            pass
        raise


def query_all(sql: str, params: tuple = ()):
    with cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchall()


def query_one(sql: str, params: tuple = ()):
    with cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchone()


def execute(sql: str, params: tuple = ()) -> int:
    """执行写操作，返回受影响行数。"""
    with cursor() as cur:
        cur.execute(sql, params)
        return cur.rowcount


def insert(sql: str, params: tuple = ()) -> int:
    """插入并返回自增主键。"""
    with cursor() as cur:
        cur.execute(sql, params)
        return cur.lastrowid


def ping() -> bool:
    try:
        conn = _new_conn()
        conn.close()
        return True
    except Exception:
        return False
