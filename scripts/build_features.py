"""批量抽取歌曲多模态特征。

产出三样东西，三者一一对应：
1) data/features/features.npz —— 音频 64 维 + 文本 32 维，供训练与在线打分直接读；
2) data/models/text_encoder.joblib —— 文本编码器，后台新增歌曲时复用它算文本特征；
3) music_feature 表 —— 按计划书 5.3 要求把向量落到库里，方便文档与查库核对。

用法：
    python scripts/build_features.py            # 增量：只算还没有特征的歌
    python scripts/build_features.py --force    # 全量重算
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import db, schema  # noqa: E402
from app.config import settings  # noqa: E402
from recsys.features import (  # noqa: E402
    AUDIO_DIM,
    build_text_corpus,
    encode_text,
    extract_audio_feature,
    fit_text_encoder,
    save_features,
    to_json,
)


def _worker(args: tuple[int, str]) -> tuple[int, list[float]]:
    """子进程里算单首歌的音频特征；失败返回零向量，保证批处理不中断。"""
    music_id, path = args
    try:
        feat = extract_audio_feature(path)
    except Exception as exc:                      # pragma: no cover
        print(f"[feature] music_id={music_id} 提取失败：{exc}")
        feat = np.zeros(AUDIO_DIM, dtype=np.float32)
    return music_id, feat.tolist()


def build(force: bool = False, workers: int = 4) -> None:
    schema.init_all()
    rows = db.query_all(
        "SELECT id, music_name, singer, album, style_tag, description, lyric, audio_path FROM music ORDER BY id"
    )
    if not rows:
        raise SystemExit("music 表为空，请先运行 scripts/load_tracks.py")

    existing = set()
    if not force:
        existing = {r["music_id"] for r in db.query_all("SELECT music_id FROM music_feature")}

    # ---------- 1) 音频特征 ----------
    todo = [(r["id"], str(ROOT / r["audio_path"])) for r in rows if r["id"] not in existing]
    audio_map: dict[int, list[float]] = {}

    if todo:
        # 用线程池而不是进程池：librosa/numpy 的 FFT、BLAS 计算会释放 GIL，
        # 线程池能拿到大部分并行收益，同时避开 Windows 上进程池 spawn + numba 的兼容问题。
        print(f"[feature] 需要提取 {len(todo)} 首音频特征，使用 {workers} 个线程")
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for i, (mid, feat) in enumerate(pool.map(_worker, todo), start=1):
                audio_map[mid] = feat
                if i % 50 == 0:
                    print(f"[feature] 音频 {i}/{len(todo)}")
    else:
        print("[feature] 音频特征已全部存在，跳过")

    # 已存在的音频特征从库里读回，保证全量一致
    if existing:
        for r in db.query_all("SELECT music_id, audio_feature FROM music_feature"):
            audio_map.setdefault(r["music_id"], json.loads(r["audio_feature"]))

    # ---------- 2) 文本特征 ----------
    corpus = build_text_corpus(rows)
    vectorizer, svd = fit_text_encoder(corpus)
    text_feats = encode_text(corpus, vectorizer, svd)
    settings.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"vectorizer": vectorizer, "svd": svd}, settings.MODEL_DIR / "text_encoder.joblib")
    print(f"[feature] 文本编码器已拟合：TF-IDF {len(vectorizer.vocabulary_)} 维 → SVD {text_feats.shape[1]} 维")

    # ---------- 3) 对齐写盘 ----------
    music_ids = [r["id"] for r in rows]
    audio_matrix = np.array([audio_map[mid] for mid in music_ids], dtype=np.float32)
    assert audio_matrix.shape[1] == AUDIO_DIM, f"音频维度异常：{audio_matrix.shape}"

    settings.FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    npz_path = settings.FEATURE_DIR / "features.npz"
    save_features(audio_matrix, text_feats, music_ids, npz_path)
    print(f"[feature] 已写出 {npz_path}：{audio_matrix.shape[0]} 首 × ({audio_matrix.shape[1]} 音频 + {text_feats.shape[1]} 文本)")

    # ---------- 4) 落库 ----------
    payload = [
        (mid, to_json(audio_matrix[i]), to_json(text_feats[i]))
        for i, mid in enumerate(music_ids)
    ]
    with db.cursor() as cur:
        cur.executemany(
            """INSERT INTO music_feature (music_id, audio_feature, text_feature)
               VALUES (%s, %s, %s)
               ON DUPLICATE KEY UPDATE audio_feature = VALUES(audio_feature),
                                       text_feature = VALUES(text_feature),
                                       update_time = CURRENT_TIMESTAMP""",
            payload,
        )
    print(f"[feature] music_feature 表已同步 {len(payload)} 行")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="忽略已有特征，全量重算")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    build(force=args.force, workers=args.workers)


if __name__ == "__main__":
    main()
