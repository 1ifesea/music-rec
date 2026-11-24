"""推荐模块（对应 REQ005 特征、REQ006 自适应融合推荐、REQ007 展示、REQ010 点击统计）。

接口设计上给答辩留了两个"证据位"：
  GET /api/recommend          —— 个性化列表 + 该用户当前的三模态权重
  GET /api/recommend/compare  —— 同一用户下「自适应权重」与「固定等权」的结果并排对比
"""
import json

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel

from .. import db
from ..config import settings
from ..deps import current_user
from recsys.serving import get_engine

router = APIRouter(prefix="/api/recommend", tags=["recommend"])


class ClickIn(BaseModel):
    music_id: int
    position: int = 0
    req_id: str = ""


def _enrich(items: list[dict]) -> list[dict]:
    """把 music_id 补成完整歌曲信息，一次 IN 查询搞定。"""
    if not items:
        return []
    ids = [i["music_id"] for i in items]
    rows = db.query_all(
        f"""SELECT id, music_name, singer, album, style_tag, duration, play_count
            FROM music WHERE id IN ({','.join(['%s'] * len(ids))})""", tuple(ids))
    by_id = {r["id"]: r for r in rows}
    for it in items:
        meta = by_id.get(it["music_id"], {})
        it.update({
            "music_name": meta.get("music_name", ""),
            "singer": meta.get("singer", ""),
            "album": meta.get("album", ""),
            "style_tag": meta.get("style_tag", ""),
            "duration": meta.get("duration", 0),
        })
    return items


@router.get("")
def recommend(n: int = Query(default=10, ge=1, le=50),
              mode: str = Query(default="adaptive", pattern="^(adaptive|fixed)$"),
              user=Depends(current_user)):
    engine = get_engine()
    result = engine.recommend(user["id"], n=n, fixed_weights=(mode == "fixed"))
    result["items"] = _enrich(result["items"])
    result["user"] = {"id": user["id"], "nickname": user["nickname"]}
    return result


@router.get("/weights")
def weights(user=Depends(current_user)):
    """只返回权重与活跃度，前端用小块条形图展示"每个用户权重不同"。"""
    engine = get_engine()
    result = engine.recommend(user["id"], n=3)
    return {
        "ready": engine.ready,
        "mode": result["mode"],
        "weights": result["weights"],
        "activity": result["activity"],
        "notice": result.get("notice"),
    }


@router.get("/compare")
def compare(n: int = Query(default=10, ge=1, le=20), user=Depends(current_user)):
    """自适应 vs 固定等权：同一用户、同一时刻的两份推荐结果。"""
    engine = get_engine()
    adaptive = engine.recommend(user["id"], n=n, fixed_weights=False)
    fixed = engine.recommend(user["id"], n=n, fixed_weights=True)
    return {
        "adaptive": {"weights": adaptive["weights"], "items": _enrich(adaptive["items"])},
        "fixed": {"weights": fixed["weights"], "items": _enrich(fixed["items"])},
        "overlap": len({i["music_id"] for i in adaptive["items"]} & {i["music_id"] for i in fixed["items"]}),
    }


@router.post("/click")
def record_click(payload: ClickIn, user=Depends(current_user)):
    db.insert(
        "INSERT INTO rec_click (user_id, music_id, req_id, position, clicked) VALUES (%s,%s,%s,%s,1)",
        (user["id"], payload.music_id, payload.req_id, payload.position),
    )
    return {"ok": True}


@router.get("/metrics")
def offline_metrics():
    """离线评估结果（HR@10 / NDCG@10 / 平均权重），来自 recsys/train.py 的输出。"""
    path = settings.MODEL_DIR / "metrics.json"
    if not path.exists():
        return {"ready": False, "notice": "尚未训练，请先运行 scripts/train.py"}
    return {"ready": True, **json.loads(path.read_text(encoding="utf-8"))}


@router.get("/stats")
def click_stats():
    """REQ010：推荐位点击情况统计。"""
    total = db.query_one("SELECT COUNT(*) AS c FROM rec_click")["c"]
    users = db.query_one("SELECT COUNT(DISTINCT user_id) AS c FROM rec_click")["c"]
    top = db.query_all(
        """SELECT c.music_id, m.music_name, m.singer, COUNT(*) AS clicks
           FROM rec_click c JOIN music m ON m.id = c.music_id
           GROUP BY c.music_id, m.music_name, m.singer
           ORDER BY clicks DESC LIMIT 10""")
    return {"click_total": total, "click_users": users, "top_clicked": top}
