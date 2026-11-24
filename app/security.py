"""口令哈希与登录令牌。

刻意不使用 passlib / python-jose：这两个包版本变动频繁，
自己写 PBKDF2 + HMAC 签名反而更容易在答辩时讲明白原理。
"""
import base64
import hashlib
import hmac
import json
import os
import time

from .config import settings

_ITERATIONS = 100_000


def hash_password(password: str) -> str:
    """PBKDF2-HMAC-SHA256，格式：pbkdf2_sha256$轮数$盐$摘要。"""
    salt = os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, _ITERATIONS)
    return f"pbkdf2_sha256${_ITERATIONS}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        algo, iterations, salt_hex, digest_hex = stored.split("$")
        if algo != "pbkdf2_sha256":
            return False
        salt = bytes.fromhex(salt_hex)
        expected = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, int(iterations))
        return hmac.compare_digest(expected.hex(), digest_hex)
    except Exception:
        return False


def _b64encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _b64decode(text: str) -> bytes:
    padding = "=" * (-len(text) % 4)
    return base64.urlsafe_b64decode(text + padding)


def create_token(user_id: int, role: str) -> str:
    """签发令牌：payload 明文 + HMAC 签名，防止前端篡改用户身份。"""
    payload = {
        "uid": int(user_id),
        "role": role,
        "exp": int(time.time()) + settings.TOKEN_EXPIRE_HOURS * 3600,
    }
    body = _b64encode(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    signature = _b64encode(
        hmac.new(settings.SECRET_KEY.encode("utf-8"), body.encode("utf-8"), hashlib.sha256).digest()
    )
    return f"{body}.{signature}"


def parse_token(token: str) -> dict | None:
    """校验签名与过期时间，失败返回 None。"""
    try:
        body, signature = token.split(".")
        expected = _b64encode(
            hmac.new(settings.SECRET_KEY.encode("utf-8"), body.encode("utf-8"), hashlib.sha256).digest()
        )
        # 用 compare_digest 防止时序侧信道
        if not hmac.compare_digest(expected, signature):
            return None
        payload = json.loads(_b64decode(body))
        if int(payload.get("exp", 0)) < int(time.time()):
            return None
        return payload
    except Exception:
        return None
