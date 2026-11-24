"""FastAPI 应用入口。

启动顺序：建库建表 → 加载推荐模型（常驻内存）→ 挂载静态页面。
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import db, schema
from .config import settings
from .routers import admin, auth, behavior, music, recommend
from recsys.serving import get_engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动阶段：确保表结构存在，并预热推荐模型（模型只加载一次）
    try:
        schema.init_all()
    except Exception as exc:                       # pragma: no cover
        print(f"[startup] 数据库初始化失败：{exc}")

    engine = get_engine()
    if engine.ready:
        print("[startup] 推荐模型已加载")
    else:
        print(f"[startup] 推荐模型未就绪：{engine.error}")
    yield


app = FastAPI(title="多模态音乐个性化推荐系统", version="1.0", lifespan=lifespan)

app.include_router(auth.router)
app.include_router(music.router)
app.include_router(behavior.router)
app.include_router(recommend.router)
app.include_router(admin.router)

# 前端静态资源（Vue3 运行时、样式、脚本），无需 Node 构建
assets_dir = settings.WEB_DIR / "assets"
if assets_dir.exists():
    app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")


@app.get("/health")
def health():
    engine = get_engine()
    return {
        "db": db.ping(),
        "model_ready": engine.ready,
        "model_notice": engine.error,
        "music_count": db.query_one("SELECT COUNT(*) AS c FROM music")["c"] if db.ping() else 0,
    }


PAGES = {
    "/": "index.html",
    "/login": "login.html",
    "/library": "library.html",
    "/mine": "mine.html",
    "/admin": "admin.html",
    "/compare": "compare.html",
}


@app.get("/favicon.ico")
def favicon():
    icon = settings.WEB_DIR / "assets" / "favicon.svg"
    if icon.exists():
        return FileResponse(icon)
    raise HTTPException(status_code=404)


for route, filename in PAGES.items():
    def _make(name: str):
        def _page():
            path = settings.WEB_DIR / name
            if not path.exists():
                raise HTTPException(status_code=404, detail=f"{name} 不存在")
            return FileResponse(path, media_type="text/html")
        return _page

    app.get(route, include_in_schema=False)(_make(filename))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=False)
