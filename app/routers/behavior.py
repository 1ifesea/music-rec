"""行为记录模块（对应 REQ003 记录播放行为、REQ004 收藏与历史记录）。"""
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from .. import db
from ..deps import current_user

router = APIRouter(prefix="/api/behavior", tags=["behavior"])


class MusicRef(BaseModel):
    music_id: int


@router.post("/play")
def record_play(payload: MusicRef, user=Depends(current_user)):
    """播放一次就落一条历史，同时累加歌曲热度。"""
    if not db.query_one("SELECT id FROM music WHERE id = %s", (payload.music_id,)):
        raise HTTPException(status_code=404, detail="歌曲不存在")
    db.insert("INSERT INTO user_play_history (user_id, music_id) VALUES (%s, %s)",
              (user["id"], payload.music_id))
    db.execute("UPDATE music SET play_count = play_count + 1 WHERE id = %s", (payload.music_id,))
    return {"ok": True}


@router.get("/history")
def play_history(limit: int = Query(default=50, ge=1, le=200), user=Depends(current_user)):
    rows = db.query_all(
        """SELECT h.id, h.music_id, h.play_time, m.music_name, m.singer, m.style_tag
           FROM user_play_history h JOIN music m ON m.id = h.music_id
           WHERE h.user_id = %s
           ORDER BY h.id DESC LIMIT %s""", (user["id"], limit))
    return {"items": rows}


@router.get("/favorites")
def favorites(user=Depends(current_user)):
    rows = db.query_all(
        """SELECT f.music_id, f.collect_time, m.music_name, m.singer, m.style_tag, m.duration
           FROM user_favorite f JOIN music m ON m.id = f.music_id
           WHERE f.user_id = %s ORDER BY f.id DESC""", (user["id"],))
    return {"items": rows}


@router.post("/favorite")
def add_favorite(payload: MusicRef, user=Depends(current_user)):
    if not db.query_one("SELECT id FROM music WHERE id = %s", (payload.music_id,)):
        raise HTTPException(status_code=404, detail="歌曲不存在")
    db.execute("INSERT IGNORE INTO user_favorite (user_id, music_id) VALUES (%s, %s)",
               (user["id"], payload.music_id))
    return {"ok": True, "favorited": True}


@router.delete("/favorite/{music_id}")
def remove_favorite(music_id: int, user=Depends(current_user)):
    db.execute("DELETE FROM user_favorite WHERE user_id = %s AND music_id = %s",
               (user["id"], music_id))
    return {"ok": True, "favorited": False}


@router.get("/stats")
def personal_stats(user=Depends(current_user)):
    """个人听歌画像：用于前端"我的"页面展示，也方便演示行为模态的输入。"""
    plays = db.query_one("SELECT COUNT(*) AS c FROM user_play_history WHERE user_id = %s",
                         (user["id"],))["c"]
    favs = db.query_one("SELECT COUNT(*) AS c FROM user_favorite WHERE user_id = %s",
                        (user["id"],))["c"]
    top_genres = db.query_all(
        """SELECT m.style_tag AS tag, COUNT(*) AS c
           FROM user_play_history h JOIN music m ON m.id = h.music_id
           WHERE h.user_id = %s GROUP BY m.style_tag ORDER BY c DESC LIMIT 3""", (user["id"],))
    return {"play_count": plays, "favorite_count": favs, "top_genres": top_genres}
