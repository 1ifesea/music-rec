"""用户模块接口（对应需求 REQ001：注册、登录、修改个人基础信息）。"""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from .. import db
from ..deps import current_user
from ..security import create_token, hash_password, verify_password

router = APIRouter(prefix="/api/auth", tags=["auth"])


class RegisterIn(BaseModel):
    username: str = Field(min_length=3, max_length=50)
    password: str = Field(min_length=6, max_length=64)
    nickname: str = ""


class LoginIn(BaseModel):
    username: str
    password: str


def _public_user(row: dict) -> dict:
    return {
        "id": row["id"],
        "username": row["username"],
        "nickname": row["nickname"],
        "role": row["role"],
    }


@router.post("/register")
def register(payload: RegisterIn):
    if db.query_one("SELECT id FROM `user` WHERE username = %s", (payload.username,)):
        raise HTTPException(status_code=400, detail="用户名已被占用")

    uid = db.insert(
        "INSERT INTO `user` (username, password_hash, nickname, role) VALUES (%s,%s,%s,%s)",
        (payload.username, hash_password(payload.password),
         payload.nickname or payload.username, "user"),
    )
    user = db.query_one("SELECT id, username, nickname, role FROM `user` WHERE id = %s", (uid,))
    return {"token": create_token(user["id"], user["role"]), "user": _public_user(user)}


@router.post("/login")
def login(payload: LoginIn):
    row = db.query_one(
        "SELECT id, username, nickname, role, password_hash FROM `user` WHERE username = %s",
        (payload.username,),
    )
    if not row or not verify_password(payload.password, row["password_hash"]):
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    return {"token": create_token(row["id"], row["role"]), "user": _public_user(row)}


@router.get("/me")
def me(user=Depends(current_user)):
    return _public_user(user)


@router.put("/profile")
def update_profile(payload: dict, user=Depends(current_user)):
    nickname = str(payload.get("nickname", "")).strip()
    if not nickname:
        raise HTTPException(status_code=400, detail="昵称不能为空")
    db.execute("UPDATE `user` SET nickname = %s WHERE id = %s", (nickname, user["id"]))
    return {"ok": True, "nickname": nickname}
