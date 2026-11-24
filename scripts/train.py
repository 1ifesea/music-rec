"""训练入口（薄封装，真正的逻辑在 recsys/train.py）。

用法：
    python scripts/train.py
    python scripts/train.py --epochs 40
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from recsys.train import train  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    metrics = train(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
    import json
    from app.config import settings

    out = settings.MODEL_DIR / "metrics.json"
    out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"[train] 模型与指标已保存到 {settings.MODEL_DIR}")


if __name__ == "__main__":
    main()
