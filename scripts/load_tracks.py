"""把 data/meta/tracks.csv 导入 music 表。

保持 music_id 与 CSV 一致（而不是让 MySQL 自增），这样特征文件、模型下标、数据库三者
指向同一首歌，避免训练完模型后下标错位。
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import db, schema  # noqa: E402
from app.config import settings  # noqa: E402


def load(reset: bool = False) -> int:
    csv_path = settings.META_DIR / "tracks.csv"
    if not csv_path.exists():
        raise SystemExit("找不到 data/meta/tracks.csv，请先运行 scripts/prepare_data.py")

    schema.init_all()
    if reset:
        db.execute("DELETE FROM music_feature")
        db.execute("DELETE FROM music")
        print("[load] 已清空 music / music_feature")

    rows = []
    with csv_path.open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            rows.append(r)

    existing = {r["id"] for r in db.query_all("SELECT id FROM music")}
    inserted = 0
    for r in rows:
        mid = int(r["music_id"])
        if mid in existing:
            continue
        db.insert(
            """INSERT INTO music
               (id, music_name, singer, album, style_tag, description, lyric, audio_path, duration)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)""",
            (mid, r["music_name"], r["singer"], r["album"], r["style_tag"],
             r["description"], r["lyric"], r["audio_path"], float(r["duration"] or 0)),
        )
        inserted += 1

    total = db.query_one("SELECT COUNT(*) AS c FROM music")["c"]
    print(f"[load] 新增 {inserted} 首，music 表现有 {total} 首")
    return total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="清空歌曲表后重新导入")
    args = parser.parse_args()
    load(reset=args.reset)


if __name__ == "__main__":
    main()
