"""音乐资源模块（对应 REQ002 浏览、REQ008 搜索）与音频播放（REQ003 的音频供给部分）。

音频播放刻意不做流媒体解码（计划书 3.2 已明确排除），
只是把本地音频文件按 HTTP Range 分片返回，让浏览器 <audio> 能拖动进度条。
"""
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse, Response

from .. import db
from ..config import ROOT_DIR

router = APIRouter(prefix="/api", tags=["music"])

MEDIA_TYPES = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".flac": "audio/flac",
    ".ogg": "audio/ogg",
    ".m4a": "audio/mp4",
}


@router.get("/music/genres")
def list_genres():
    rows = db.query_all(
        "SELECT style_tag AS tag, COUNT(*) AS count FROM music GROUP BY style_tag ORDER BY count DESC"
    )
    return {"items": rows}


@router.get("/music")
def list_music(
    keyword: str = Query(default="", description="歌名/歌手关键词"),
    genre: str = Query(default="", description="风格标签"),
    page: int = Query(default=1, ge=1),
    size: int = Query(default=20, ge=1, le=100),
):
    where, params = [], []
    if keyword:
        where.append("(music_name LIKE %s OR singer LIKE %s)")
        params.extend([f"%{keyword}%", f"%{keyword}%"])
    if genre:
        where.append("style_tag = %s")
        params.append(genre)
    clause = f"WHERE {' AND '.join(where)}" if where else ""

    total = db.query_one(f"SELECT COUNT(*) AS c FROM music {clause}", tuple(params))["c"]
    rows = db.query_all(
        f"""SELECT id, music_name, singer, album, style_tag, duration, play_count
            FROM music {clause}
            ORDER BY play_count DESC, id ASC
            LIMIT {int(size)} OFFSET {int((page - 1) * size)}""",
        tuple(params),
    )
    return {"total": total, "page": page, "size": size, "items": rows}


@router.get("/music/{music_id}")
def music_detail(music_id: int):
    row = db.query_one(
        """SELECT id, music_name, singer, album, style_tag, description, lyric,
                  audio_path, duration, play_count
           FROM music WHERE id = %s""", (music_id,)
    )
    if not row:
        raise HTTPException(status_code=404, detail="歌曲不存在")
    return row


@router.get("/audio/{music_id}")
def stream_audio(music_id: int, request: Request):
    row = db.query_one("SELECT audio_path FROM music WHERE id = %s", (music_id,))
    if not row:
        raise HTTPException(status_code=404, detail="歌曲不存在")

    path = Path(row["audio_path"])
    if not path.is_absolute():
        path = ROOT_DIR / path
    if not path.exists():
        raise HTTPException(status_code=404, detail="音频文件缺失")

    media_type = MEDIA_TYPES.get(path.suffix.lower(), "application/octet-stream")
    size = path.stat().st_size
    range_header = request.headers.get("range")

    if not range_header:
        return FileResponse(path, media_type=media_type)

    # 解析 "bytes=start-end"，支持浏览器拖动进度条时的分段请求
    try:
        unit, _, spec = range_header.partition("=")
        start_str, _, end_str = spec.partition("-")
        start = int(start_str) if start_str else 0
        end = int(end_str) if end_str else size - 1
    except ValueError:
        raise HTTPException(status_code=416, detail="Range 头格式不正确")
    end = min(end, size - 1)
    if start >= size or start > end:
        return Response(status_code=416, headers={"Content-Range": f"bytes */{size}"})

    with path.open("rb") as f:
        f.seek(start)
        chunk = f.read(end - start + 1)
    return Response(
        content=chunk,
        status_code=206,
        media_type=media_type,
        headers={
            "Content-Range": f"bytes {start}-{end}/{size}",
            "Accept-Ranges": "bytes",
            "Content-Length": str(len(chunk)),
        },
    )
