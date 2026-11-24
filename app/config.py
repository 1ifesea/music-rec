"""全局配置：优先读环境变量，其次读项目根目录的 .env 文件。

这样做的原因是课程演示时只需要改 .env 一处，不用到处改代码。
"""
import os
from pathlib import Path

# 项目根目录：app/config.py 的上两级
ROOT_DIR = Path(__file__).resolve().parent.parent


def _load_env_file(path: Path) -> None:
    """极简 .env 解析，避免引入 python-dotenv 依赖。"""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        # 已存在的环境变量优先，不覆盖
        os.environ.setdefault(key, value)


_load_env_file(ROOT_DIR / ".env")


class Settings:
    """所有可变配置集中在这里。"""

    # ---- 数据库 ----
    DB_HOST: str = os.getenv("DB_HOST", "127.0.0.1")
    DB_PORT: int = int(os.getenv("DB_PORT", "3306"))
    DB_USER: str = os.getenv("DB_USER", "root")
    DB_PASSWORD: str = os.getenv("DB_PASSWORD", "")
    DB_NAME: str = os.getenv("DB_NAME", "musicrec")

    # ---- 安全 ----
    # 生成 token 用的密钥；课程环境固定即可，生产环境要换成随机长串
    SECRET_KEY: str = os.getenv("SECRET_KEY", "musicrec-course-design-2026")
    TOKEN_EXPIRE_HOURS: int = int(os.getenv("TOKEN_EXPIRE_HOURS", "72"))

    # ---- 路径 ----
    DATA_DIR: Path = ROOT_DIR / "data"
    AUDIO_DIR: Path = DATA_DIR / "audio"
    META_DIR: Path = DATA_DIR / "meta"
    FEATURE_DIR: Path = DATA_DIR / "features"
    MODEL_DIR: Path = DATA_DIR / "models"
    WEB_DIR: Path = ROOT_DIR / "web"

    # ---- 算法 ----
    AUDIO_DIM: int = 64      # 音频特征维度，由 recsys/features.py 固定产出
    TEXT_DIM: int = 32       # 文本特征维度（TF-IDF + SVD 降维后）
    EMB_DIM: int = 32        # 用户/歌曲隐向量维度
    TOP_N_DEFAULT: int = 10  # 默认推荐数量
    CANDIDATE_LIMIT: int = 200  # 参与排序的候选上限，用于控时

    @classmethod
    def ensure_dirs(cls) -> None:
        for d in (cls.DATA_DIR, cls.AUDIO_DIR, cls.META_DIR, cls.FEATURE_DIR, cls.MODEL_DIR):
            d.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_dirs()
