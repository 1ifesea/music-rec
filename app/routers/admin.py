"""后台管理模块（对应 REQ009：歌曲新增、编辑、删除）。

注意：通过后台上架/修改歌曲之后，特征与模型需要重新生成，
接口会返回 notice 提示执行 scripts/build_features.py 与 scripts/train.py。
这是课程阶段刻意保留的"离线训练、在线服务"边界，答辩时可以讲成一条设计约束。
"""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from .. import db
from ..deps import current_admin
from recsys.serving import get_engine, reload_engine

router = APIRouter(prefix="/api/admin", tags=["admin"])

REBUILD_NOTICE = "歌曲库已变更，请运行 python scripts/build_features.py && python scripts/train.py 后调用 /api/admin/reload 生效"


class MusicIn(BaseModel):
    music_name: str
    singer: str = ""
    album: str = ""
    style_tag: str = ""
    description: str = ""
    lyric: str = ""
    audio_path: str = ""
    duration: float = 0.0


@router.get("/overview")
def overview(admin=Depends(current_admin)):
    def scalar(sql: str) -> int:
        return db.query_one(sql)["c"]

    engine = get_engine()
    return {
        "users": scalar("SELECT COUNT(*) AS c FROM `user` WHERE role = 'user'"),
        "sim_users": scalar("SELECT COUNT(*) AS c FROM `user` WHERE role = 'sim'"),
        "music": scalar("SELECT COUNT(*) AS c FROM music"),
        "features": scalar("SELECT COUNT(*) AS c FROM music_feature"),
        "plays": scalar("SELECT COUNT(*) AS c FROM user_play_history"),
        "favorites": scalar("SELECT COUNT(*) AS c FROM user_favorite"),
        "clicks": scalar("SELECT COUNT(*) AS c FROM rec_click"),
        "model_ready": engine.ready,
        "model_notice": engine.error,
    }


@router.get("/users")
def list_users(admin=Depends(current_admin)):
    rows = db.query_all(
        """SELECT u.id, u.username, u.nickname, u.role, u.create_time,
                  (SELECT COUNT(*) FROM user_play_history h WHERE h.user_id = u.id) AS plays
           FROM `user` u WHERE u.role <> 'sim'
           ORDER BY u.id DESC LIMIT 200""")
    return {"items": rows}


@router.get("/music")
def admin_music(admin=Depends(current_admin)):
    rows = db.query_all(
        """SELECT id, music_name, singer, album, style_tag, audio_path, duration, play_count
           FROM music ORDER BY id DESC LIMIT 300""")
    return {"items": rows}


@router.post("/music")
def create_music(payload: MusicIn, admin=Depends(current_admin)):
    if not payload.music_name.strip():
        raise HTTPException(status_code=400, detail="歌名不能为空")
    mid = db.insert(
        """INSERT INTO music (music_name, singer, album, style_tag, description, lyric, audio_path, duration)
           VALUES (%s,%s,%s,%s,%s,%s,%s,%s)""",
        (payload.music_name, payload.singer, payload.album, payload.style_tag,
         payload.description, payload.lyric, payload.audio_path, payload.duration))
    return {"ok": True, "id": mid, "notice": REBUILD_NOTICE}


@router.put("/music/{music_id}")
def update_music(music_id: int, payload: MusicIn, admin=Depends(current_admin)):
    if not db.query_one("SELECT id FROM music WHERE id = %s", (music_id,)):
        raise HTTPException(status_code=404, detail="歌曲不存在")
    db.execute(
        """UPDATE music SET music_name=%s, singer=%s, album=%s, style_tag=%s,
                  description=%s, lyric=%s, audio_path=%s, duration=%s WHERE id=%s""",
        (payload.music_name, payload.singer, payload.album, payload.style_tag,
         payload.description, payload.lyric, payload.audio_path, payload.duration, music_id))
    return {"ok": True, "notice": REBUILD_NOTICE}


@router.delete("/music/{music_id}")
def delete_music(music_id: int, admin=Depends(current_admin)):
    if not db.query_one("SELECT id FROM music WHERE id = %s", (music_id,)):
        raise HTTPException(status_code=404, detail="歌曲不存在")
    db.execute("DELETE FROM music_feature WHERE music_id = %s", (music_id,))
    db.execute("DELETE FROM music WHERE id = %s", (music_id,))
    return {"ok": True, "notice": REBUILD_NOTICE}


@router.post("/reload")
def reload_model(admin=Depends(current_admin)):
    """重新训练后热加载模型，不必重启服务。"""
    engine = reload_engine()
    return {"ok": engine.ready, "notice": engine.error}
