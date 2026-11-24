"""生成模拟用户与交互数据。

为什么需要这一步：课程阶段没有真实用户，而门控网络必须靠「用户—歌曲」交互才能学出
各模态权重。所以按「风格偏好画像」合成一批用户行为。

生成规则（可复现，随机种子固定）：
1. 每个用户有 1 个主风格标签，60% 概率再带 1 个次风格；
2. 点歌概率 = 风格匹配权重 × 歌曲热度（Zipf）× 随机扰动；
3. 交互条数 8~120 条，另有 40 个「新用户」只有 0~2 条，用于冷启动观察；
4. 10% 的播放会转成收藏。

这批数据在报告里必须标注为模拟数据，不能表述成真实用户行为。
"""
from __future__ import annotations

import argparse
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import db, schema  # noqa: E402
from app.security import hash_password  # noqa: E402

SIM_USERS = 300
COLD_USERS = 40          # 其中 40 个是"新用户"，交互极少


def _load_songs() -> tuple[list[int], dict[int, str]]:
    rows = db.query_all("SELECT id, style_tag FROM music")
    ids = [r["id"] for r in rows]
    tags = {r["id"]: r["style_tag"] for r in rows}
    if not ids:
        raise SystemExit("music 表为空，请先运行 scripts/load_tracks.py")
    return ids, tags


def _build_weights(songs: list[int], tags: dict[int, str],
                   primary: str, secondary: str | None, rng: random.Random) -> list[float]:
    """给每首歌算一个被选中的相对概率。"""
    weights = []
    for sid in songs:
        tag = tags[sid]
        if tag == primary:
            base = 3.0
        elif secondary and tag == secondary:
            base = 1.6
        else:
            base = 0.25
        # 热度：长尾分布，模拟"热门歌被点得多"
        base *= 0.6 + 0.4 / (1 + rng.random() * 3)
        weights.append(base * rng.random())
    return weights


def seed(reset: bool = False, n_users: int = SIM_USERS) -> None:
    schema.init_all()
    rng = random.Random(2026)
    songs, tags = _load_songs()
    genres = sorted(set(tags.values()))

    if reset:
        db.execute("DELETE FROM user_play_history")
        db.execute("DELETE FROM user_favorite")
        db.execute("DELETE FROM rec_click")
        db.execute("DELETE FROM `user` WHERE role = 'sim'")

    now = datetime.now()
    sim_prefix_exist = db.query_one("SELECT COUNT(*) AS c FROM `user` WHERE role = 'sim'")["c"]
    if sim_prefix_exist >= n_users:
        print(f"[seed] 已有 {sim_prefix_exist} 个模拟用户，跳过生成（如需重建加 --reset）")
        return

    user_rows, play_rows, fav_rows = [], [], []
    base_id = db.query_one("SELECT COALESCE(MAX(id), 0) AS m FROM `user`")["m"]
    pwd = hash_password("sim123456")

    for i in range(n_users):
        uid = base_id + i + 1
        primary = rng.choice(genres)
        secondary = rng.choice(genres) if rng.random() < 0.6 else None
        if secondary == primary:
            secondary = None
        user_rows.append((uid, f"sim{uid:03d}", pwd, f"模拟用户{primary}", "sim"))

        cold = i >= n_users - COLD_USERS
        n_play = rng.randint(0, 2) if cold else rng.randint(8, 120)
        if n_play == 0:
            continue

        weights = _build_weights(songs, tags, primary, secondary, rng)
        # 按权重采样（不重复地听，重复听由 play_count 体现）
        picked = rng.choices(songs, weights=weights, k=n_play)
        picked = list(dict.fromkeys(picked))[:n_play]

        times = sorted(now - timedelta(minutes=rng.randint(0, 180 * 24 * 60)) for _ in range(len(picked)))
        for sid, ts in zip(picked, times):
            play_rows.append((uid, sid, ts.strftime("%Y-%m-%d %H:%M:%S")))
            if rng.random() < 0.1:
                fav_rows.append((uid, sid, ts.strftime("%Y-%m-%d %H:%M:%S")))

    with db.cursor() as cur:
        cur.executemany(
            "INSERT INTO `user` (id, username, password_hash, nickname, role) VALUES (%s,%s,%s,%s,%s)",
            user_rows,
        )
        cur.executemany(
            "INSERT INTO user_play_history (user_id, music_id, play_time) VALUES (%s,%s,%s)",
            play_rows,
        )
        cur.executemany(
            "INSERT IGNORE INTO user_favorite (user_id, music_id, collect_time) VALUES (%s,%s,%s)",
            fav_rows,
        )

    # 同步歌曲热度
    db.execute("UPDATE music m JOIN (SELECT music_id, COUNT(*) c FROM user_play_history GROUP BY music_id) t "
               "ON m.id = t.music_id SET m.play_count = t.c")

    print(f"[seed] 模拟用户 {len(user_rows)} 个，播放记录 {len(play_rows)} 条，收藏 {len(fav_rows)} 条")


def seed_demo_accounts() -> None:
    """建 3 个给老师现场演示的账号：两个偏好不同的老用户 + 一个新用户。"""
    demos = [
        ("demo1", "小林", "流行", 60),
        ("demo2", "阿哲", "电子", 60),
        ("demo3", "新同学", None, 0),
    ]
    songs, tags = _load_songs()
    rng = random.Random(99)
    now = datetime.now()

    for username, nickname, genre, n_play in demos:
        if db.query_one("SELECT id FROM `user` WHERE username = %s", (username,)):
            continue
        uid = db.insert(
            "INSERT INTO `user` (username, password_hash, nickname, role) VALUES (%s,%s,%s,%s)",
            (username, hash_password("123456"), nickname, "user"),
        )
        if n_play == 0 or not genre:
            continue
        pool = [s for s in songs if tags[s] == genre]
        other = [s for s in songs if tags[s] != genre]
        picked = rng.sample(pool, min(n_play, len(pool)))
        played = []
        for sid in picked:
            ts = now - timedelta(days=rng.randint(1, 90), minutes=rng.randint(0, 600))
            played.append((uid, sid, ts.strftime("%Y-%m-%d %H:%M:%S")))
            if rng.random() < 0.2:
                db.execute("INSERT IGNORE INTO user_favorite (user_id, music_id, collect_time) VALUES (%s,%s,%s)",
                           (uid, sid, ts.strftime("%Y-%m-%d %H:%M:%S")))
        # 掺一点跨风格行为，让画像更真实
        for sid in rng.sample(other, min(5, len(other))):
            played.append((uid, sid, now.strftime("%Y-%m-%d %H:%M:%S")))
        with db.cursor() as cur:
            cur.executemany("INSERT INTO user_play_history (user_id, music_id, play_time) VALUES (%s,%s,%s)",
                            played)
        print(f"[seed] 演示账号 {username} / 123456 已创建（偏好{genre}）")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--users", type=int, default=SIM_USERS)
    args = parser.parse_args()
    seed(reset=args.reset, n_users=args.users)
    seed_demo_accounts()


if __name__ == "__main__":
    main()
