"""表结构定义与初始化。

字段与《软件工程项目计划书 V1.1》5.3 节的 ER 描述保持一致，
额外增加了两处工程必需的东西：
1. music 表补了 album / description / duration / play_count，用于文本模态和热度统计；
2. 新增 rec_click 表，对应需求 REQ010「推荐效果简单统计」。
"""
import pymysql

from .config import settings
from . import db

# ---------------------------------------------------------------- DDL

DDL_STATEMENTS = [
    # 用户表
    """
    CREATE TABLE IF NOT EXISTS `user` (
        `id`            INT AUTO_INCREMENT PRIMARY KEY,
        `username`      VARCHAR(50)  NOT NULL UNIQUE,
        `password_hash` VARCHAR(255) NOT NULL,
        `nickname`      VARCHAR(50)  NOT NULL DEFAULT '',
        `role`          VARCHAR(20)  NOT NULL DEFAULT 'user',   -- user / admin / sim(模拟用户)
        `create_time`   DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    # 歌曲表
    """
    CREATE TABLE IF NOT EXISTS `music` (
        `id`          INT AUTO_INCREMENT PRIMARY KEY,
        `music_name`  VARCHAR(200) NOT NULL,
        `singer`      VARCHAR(120) NOT NULL DEFAULT '',
        `album`       VARCHAR(200) NOT NULL DEFAULT '',
        `style_tag`   VARCHAR(50)  NOT NULL DEFAULT '',
        `description` TEXT,
        `lyric`       TEXT,
        `audio_path`  VARCHAR(300) NOT NULL DEFAULT '',
        `duration`    FLOAT        NOT NULL DEFAULT 0,
        `play_count`  INT          NOT NULL DEFAULT 0,
        `create_time` DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
        KEY `idx_style` (`style_tag`),
        KEY `idx_name`  (`music_name`)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    # 播放历史表
    """
    CREATE TABLE IF NOT EXISTS `user_play_history` (
        `id`        INT AUTO_INCREMENT PRIMARY KEY,
        `user_id`   INT      NOT NULL,
        `music_id`  INT      NOT NULL,
        `play_time` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        KEY `idx_user_time` (`user_id`, `play_time`),
        KEY `idx_music` (`music_id`)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    # 收藏表
    """
    CREATE TABLE IF NOT EXISTS `user_favorite` (
        `id`           INT AUTO_INCREMENT PRIMARY KEY,
        `user_id`      INT      NOT NULL,
        `music_id`     INT      NOT NULL,
        `collect_time` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        UNIQUE KEY `uk_user_music` (`user_id`, `music_id`),
        KEY `idx_music` (`music_id`)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    # 歌曲多模态特征表
    """
    CREATE TABLE IF NOT EXISTS `music_feature` (
        `id`            INT AUTO_INCREMENT PRIMARY KEY,
        `music_id`      INT      NOT NULL UNIQUE,
        `audio_feature` LONGTEXT NOT NULL,   -- JSON 数组，长度 AUDIO_DIM
        `text_feature`  LONGTEXT NOT NULL,   -- JSON 数组，长度 TEXT_DIM
        `update_time`   DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    # 推荐点击反馈表（REQ010）
    """
    CREATE TABLE IF NOT EXISTS `rec_click` (
        `id`          INT AUTO_INCREMENT PRIMARY KEY,
        `user_id`     INT         NOT NULL,
        `music_id`    INT         NOT NULL,
        `req_id`      VARCHAR(40) NOT NULL DEFAULT '',
        `position`    INT         NOT NULL DEFAULT 0,
        `clicked`     TINYINT     NOT NULL DEFAULT 0,
        `create_time` DATETIME    NOT NULL DEFAULT CURRENT_TIMESTAMP,
        KEY `idx_user` (`user_id`),
        KEY `idx_req`  (`req_id`)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
]


def create_database() -> None:
    """创建库（不删数据，可重复执行）。"""
    conn = pymysql.connect(
        host=settings.DB_HOST,
        port=settings.DB_PORT,
        user=settings.DB_USER,
        password=settings.DB_PASSWORD,
        charset="utf8mb4",
        autocommit=True,
        connect_timeout=5,
    )
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"CREATE DATABASE IF NOT EXISTS `{settings.DB_NAME}` "
                f"DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci"
            )
    finally:
        conn.close()


def create_tables() -> None:
    for ddl in DDL_STATEMENTS:
        db.execute(ddl)


def init_all() -> None:
    """建库 + 建表 + 初始化管理员账号。"""
    from .security import hash_password  # 延迟导入，避免循环依赖

    create_database()
    create_tables()

    admin = db.query_one("SELECT id FROM `user` WHERE username = %s", ("admin",))
    if not admin:
        db.insert(
            "INSERT INTO `user` (username, password_hash, nickname, role) VALUES (%s, %s, %s, %s)",
            ("admin", hash_password("123456"), "管理员", "admin"),
        )
        print("[init] 已创建管理员账号 admin / 123456")


if __name__ == "__main__":
    init_all()
    print("[init] 数据库与表结构就绪")
