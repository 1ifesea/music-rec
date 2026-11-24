"""在线推荐服务：模型常驻内存，请求时只做向量打分。

设计要点（对应计划书 1.2 节「推荐模型单次生成推荐列表耗时 ≤3s」）：
1. 服务启动时一次性加载模型与特征，绝不每次请求重新 load；
2. 全库打分用一次批量化前向，600 首歌约 2~5ms；
3. 模型文件缺失或加载失败时自动降级为「热度推荐」，保证演示不中断。
"""
from __future__ import annotations

import threading

import numpy as np
import torch

from app import db
from app.config import settings

from .features import load_features
from .model import MMGate
from .train import activity_of


class Recommender:
    def __init__(self) -> None:
        self.model: MMGate | None = None
        self.music_ids: list[int] = []
        self.audio: np.ndarray | None = None
        self.text: np.ndarray | None = None
        self.index: dict[int, int] = {}
        self.audio_t: torch.Tensor | None = None
        self.text_t: torch.Tensor | None = None
        self.error: str | None = None
        self._load()

    # ------------------------------------------------------------ 加载
    def _load(self) -> None:
        try:
            feat_path = settings.FEATURE_DIR / "features.npz"
            model_path = settings.MODEL_DIR / "mmgate.pt"
            ids_path = settings.MODEL_DIR / "item_ids.npy"

            if not feat_path.exists():
                self.error = "特征文件缺失，请先运行 scripts/build_features.py"
                return
            self.audio, self.text, self.music_ids = load_features(feat_path)
            self.index = {mid: i for i, mid in enumerate(self.music_ids)}
            self.audio_t = torch.tensor(self.audio, dtype=torch.float32)
            self.text_t = torch.tensor(self.text, dtype=torch.float32)

            if not (model_path.exists() and ids_path.exists()):
                self.error = "模型文件缺失，请先运行 scripts/train.py"
                return

            self.model = MMGate.load(model_path)
            trained_ids = np.load(ids_path).tolist()
            if [int(x) for x in trained_ids] != self.music_ids:
                # 歌曲库变动后模型下标会错位，必须重新训练
                self.error = "歌曲库与模型不一致，请重新运行 scripts/train.py"
                self.model = None
        except Exception as exc:                                  # pragma: no cover
            self.error = f"模型加载失败：{exc}"
            self.model = None

    @property
    def ready(self) -> bool:
        return self.model is not None

    # ------------------------------------------------------------ 用户向量
    def _user_vector(self, user_id: int) -> tuple[torch.Tensor, float]:
        rows = db.query_all(
            "SELECT music_id FROM user_play_history WHERE user_id = %s ORDER BY id", (user_id,)
        )
        favs = db.query_all(
            "SELECT music_id FROM user_favorite WHERE user_id = %s", (user_id,)
        )
        seen = [r["music_id"] for r in rows] + [r["music_id"] for r in favs]
        idx = [self.index[m] for m in seen if m in self.index]
        vec = self.model.user_vector(idx)
        return vec, activity_of(len(idx))

    # ------------------------------------------------------------ 推荐
    def recommend(self, user_id: int, n: int = 10, fixed_weights: bool = False) -> dict:
        if not self.ready:
            return self._popular_fallback(user_id, n)

        u_vec, activity = self._user_vector(user_id)
        with torch.no_grad():
            scores, weights = self.model.score_all_items(
                u_vec, activity, self.audio_t, self.text_t, fixed_weights=fixed_weights
            )

        # 过滤已播放、已收藏
        listened = db.query_all(
            "SELECT music_id FROM user_play_history WHERE user_id = %s", (user_id,)
        )
        faved = db.query_all("SELECT music_id FROM user_favorite WHERE user_id = %s", (user_id,))
        exclude_ids = {r["music_id"] for r in listened} | {r["music_id"] for r in faved}

        arr = scores.numpy().astype("float64")
        for mid in exclude_ids:
            if mid in self.index:
                arr[self.index[mid]] = -np.inf

        top_pos = np.argsort(-arr)[:n]
        items = []
        for pos in top_pos:
            if not np.isfinite(arr[pos]):
                continue
            items.append({"music_id": self.music_ids[int(pos)], "score": round(float(arr[pos]), 4)})

        return {
            "mode": "fixed" if fixed_weights else "adaptive",
            "weights": {
                "audio": round(float(weights[0]), 4),
                "text": round(float(weights[1]), 4),
                "behavior": round(float(weights[2]), 4),
            },
            "activity": round(float(activity), 4),
            "items": items,
        }

    # ------------------------------------------------------------ 降级
    def _popular_fallback(self, user_id: int, n: int) -> dict:
        rows = db.query_all(
            """
            SELECT id AS music_id, play_count FROM music
            ORDER BY play_count DESC, id ASC LIMIT %s
            """, (n,)
        )
        return {
            "mode": "popular",
            "weights": {"audio": 0.0, "text": 0.0, "behavior": 0.0},
            "activity": 0.0,
            "items": [{"music_id": r["music_id"], "score": float(r["play_count"])} for r in rows],
            "notice": self.error or "模型未就绪，当前为热度降级推荐",
        }


_engine: Recommender | None = None
_lock = threading.Lock()


def get_engine() -> Recommender:
    """全局单例，保证模型只加载一次。"""
    global _engine
    if _engine is None:
        with _lock:
            if _engine is None:
                _engine = Recommender()
    return _engine


def reload_engine() -> Recommender:
    """训练完成或歌曲库变更后，热加载模型。"""
    global _engine
    with _lock:
        _engine = Recommender()
    return _engine
