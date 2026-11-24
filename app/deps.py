"""FastAPI 依赖：从请求头里解析当前登录用户。"""
from fastapi import Depends, Header, HTTPException

from . import db
from .security import parse_token


def _extract_token(authorization: str | None) -> str | None:
    if not authorization:
        return None
    parts = authorization.split(" ", 1)
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1].strip()
    return None


def current_user(authorization: str | None = Header(default=None)):
    """解析 Bearer 令牌并查出用户；未登录或失效抛 401。"""
    token = _extract_token(authorization)
    if not token:
        raise HTTPException(status_code=401, detail="请先登录")
    payload = parse_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="登录状态已失效，请重新登录")
    user = db.query_one(
        "SELECT id, username, nickname, role FROM `user` WHERE id = %s", (payload["uid"],)
    )
    if not user:
        raise HTTPException(status_code=401, detail="账号不存在")
    return user


def current_admin(user=Depends(current_user)):
    """管理员权限校验。"""
    if user["role"] != "admin":
        raise HTTPException(status_code=403, detail="需要管理员权限")
    return user
