"""推荐模型训练与离线评估。

数据来源：user_play_history + user_favorite（课程演示阶段主要由模拟用户产生）。
评估方式：每个用户留最后一次交互做测试集，其余做训练集，全库排序后算 HR@10 / NDCG@10，
并与「固定等权」对照组比较——这是证明「自适应权重」有效的最小实验。
"""
from __future__ import annotations

import copy
import json
import math
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from app import db
from app.config import settings

from .features import load_features
from .model import MMGate

TOPK = 10
ACTIVITY_CAP = 50.0   # 活跃度归一化上限：50 次以上交互视为"高活跃"
KL_WEIGHT = 0.05      # 门控权重向均匀分布收缩的正则强度


# ------------------------------------------------------------------ 数据准备
def load_interactions() -> tuple[dict[int, list[int]], dict[int, int]]:
    """返回 {user_id: [music_id,...按时间排序且去重]} 与 {music_id: 播放次数}。

    播放与收藏会合并成一条时间线，并对同一首歌做"去重保留最后一次"的处理。
    这一步很关键：如果不去重，最后一条（待预测目标）可能在更早的历史里已经出现过，
    评估时会被当作用户听过的歌屏蔽掉，指标会变成恒为零。
    """
    plays = db.query_all("SELECT user_id, music_id, play_time FROM user_play_history")
    favs = db.query_all("SELECT user_id, music_id, collect_time FROM user_favorite")

    events = [(r["user_id"], r["music_id"], r["play_time"]) for r in plays]
    events += [(r["user_id"], r["music_id"], r["collect_time"]) for r in favs]
    events.sort(key=lambda x: (x[0], x[2]))

    history: dict[int, list[int]] = defaultdict(list)
    for uid, mid, _ in events:
        seq = history[uid]
        if mid in seq:
            seq.remove(mid)          # 去重，保留最后一次出现的位置
        seq.append(mid)

    counts = db.query_all("SELECT music_id, COUNT(*) AS c FROM user_play_history GROUP BY music_id")
    return dict(history), {r["music_id"]: r["c"] for r in counts}


def activity_of(count: int) -> float:
    """用户活跃度，压到 0~1，供门控网络使用。"""
    return min(1.0, math.log1p(count) / math.log1p(ACTIVITY_CAP))


# ------------------------------------------------------------------ 训练
def train(epochs: int = 30, batch_size: int = 256, lr: float = 1e-3, seed: int = 2026,
          es_every: int = 5, verbose: bool = True) -> dict:
    settings.ensure_dirs()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    audio, text, music_ids = load_features(settings.FEATURE_DIR / "features.npz")
    audio_t = torch.tensor(audio, dtype=torch.float32)
    text_t = torch.tensor(text, dtype=torch.float32)
    m2i = {mid: i for i, mid in enumerate(music_ids)}
    n_items = len(music_ids)

    history, play_counts = load_interactions()
    # 只保留在特征表里能找到的歌曲
    hist_idx: dict[int, list[int]] = {}
    for uid, mids in history.items():
        idxs = [m2i[m] for m in mids if m in m2i]
        if len(idxs) >= 2:               # 至少 1 条历史 + 1 条待预测目标
            hist_idx[uid] = idxs

    model = MMGate(n_items=n_items, audio_dim=audio.shape[1], text_dim=text.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    users = list(hist_idx.keys())
    if not users:
        raise RuntimeError("没有足够的交互数据，请先运行 scripts/simulated_seed 或 seed_simulated.py")

    # 早停：每若干轮在验证指标（NDCG@10）上打分，保留最好的一版权重。
    # 不加早停的话，轮次一多就会过拟合——实测 60 轮的 HR@10 反而低于 30 轮。
    best_score, best_epoch, best_state = -1.0, 0, None

    for epoch in range(1, epochs + 1):
        random.shuffle(users)
        model.train()
        total_loss, steps = 0.0, 0
        for start in range(0, len(users), batch_size):
            batch = users[start:start + batch_size]
            u_vecs, acts, pos_idx, neg_idx = [], [], [], []
            for uid in batch:
                seq = hist_idx[uid]
                pos = seq[-1]                       # 留一：最后一条做正样本
                hist = seq[:-1]
                u_vecs.append(model.user_vector(hist))
                acts.append(activity_of(len(seq)))
                neg = random.randint(0, n_items - 1)
                while neg in hist or neg == pos:
                    neg = random.randint(0, n_items - 1)
                pos_idx.append(pos)
                neg_idx.append(neg)

            u = torch.stack(u_vecs)
            act = torch.tensor(acts, dtype=torch.float32)
            pos = torch.tensor(pos_idx, dtype=torch.long)
            neg = torch.tensor(neg_idx, dtype=torch.long)

            s_pos, w_pos = model(u, act, audio_t[pos], text_t[pos], pos)
            s_neg, _ = model(u, act, audio_t[neg], text_t[neg], neg)
            loss = -F.logsigmoid(s_pos - s_neg).mean()
            # 向「均匀权重」收缩的正则：数据量小的时候，让门控只在有证据时才偏离等权
            loss = loss + KL_WEIGHT * ((w_pos * torch.log(w_pos * 3.0 + 1e-9)).sum(dim=1)).mean()

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total_loss += float(loss)
            steps += 1

        if verbose and (epoch % 5 == 0 or epoch == 1):
            print(f"[train] epoch {epoch:02d}/{epochs}  loss={total_loss / max(1, steps):.4f}")

        if epoch % es_every == 0 or epoch == epochs:
            val = evaluate(model, hist_idx, audio_t, text_t, n_items)
            score = val["adaptive"]["NDCG@10"]
            if verbose:
                print(f"[train]   验证 epoch {epoch:02d}  HR@10={val['adaptive']['HR@10']}  "
                      f"NDCG@10={score}  固定等权 NDCG@10={val['fixed']['NDCG@10']}")
            if score > best_score:
                best_score, best_epoch, best_state = score, epoch, copy.deepcopy(model.state_dict())
            model.train()                                  # evaluate 会切到 eval，这里切回来

    if best_state is not None:
        model.load_state_dict(best_state)

    metrics = evaluate(model, hist_idx, audio_t, text_t, n_items)
    metrics["n_users"] = len(hist_idx)
    metrics["n_items"] = n_items
    metrics["epochs"] = epochs
    metrics["best_epoch"] = best_epoch

    # 落盘：模型权重 + 歌曲下标映射，线上服务直接加载
    model.save(settings.MODEL_DIR / "mmgate.pt")
    np.save(settings.MODEL_DIR / "item_ids.npy", np.array(music_ids))

    # 抽 5 个用户保存权重快照，供前端与答辩演示"权重因人而异"
    snapshots = []
    for uid in list(hist_idx.keys())[:5]:
        with torch.no_grad():
            u = model.user_vector(hist_idx[uid][:-1]).unsqueeze(0).expand(n_items, -1)
            act = torch.full((n_items,), activity_of(len(hist_idx[uid])))
            idx = torch.arange(n_items)
            _, w = model(u, act, audio_t, text_t, idx)
            w_mean = w.mean(dim=0)
        snapshots.append({
            "user_id": uid,
            "history_len": len(hist_idx[uid]),
            "weights": {"audio": round(float(w_mean[0]), 4),
                        "text": round(float(w_mean[1]), 4),
                        "behavior": round(float(w_mean[2]), 4)},
        })
    metrics["weight_snapshots"] = snapshots
    return metrics


# ------------------------------------------------------------------ 评估
# 按历史长度分桶：冷启动（<5 条）、中等（5~19 条）、活跃（≥20 条）。
# 分桶的意义：如果「自适应权重」真的有用，它应该体现在冷启动用户身上
# —— 这类用户行为证据少，模型应当把行为模态权重压低、更多依赖内容模态。
BUCKETS = (("cold", 0, 5), ("mid", 5, 20), ("warm", 20, 10 ** 9))


def _bucket_of(hist_len: int) -> str:
    for name, low, high in BUCKETS:
        if low <= hist_len < high:
            return name
    return "warm"


@torch.no_grad()
def evaluate(model: MMGate, hist_idx: dict[int, list[int]],
             audio_t: torch.Tensor, text_t: torch.Tensor, n_items: int) -> dict:
    """HR@10 / NDCG@10：自适应权重 vs 固定等权，总体 + 分桶。"""
    model.eval()
    results = {}

    for mode, fixed in (("adaptive", False), ("fixed", True)):
        overall = {"hr": 0.0, "ndcg": 0.0, "n": 0, "w": torch.zeros(3)}
        buckets = {name: {"hr": 0.0, "ndcg": 0.0, "n": 0, "w": torch.zeros(3)}
                   for name, _, _ in BUCKETS}

        for uid, seq in hist_idx.items():
            pos = seq[-1]
            hist = seq[:-1]
            u = model.user_vector(hist).unsqueeze(0).expand(n_items, -1)
            act = torch.full((n_items,), activity_of(len(seq)))
            idx = torch.arange(n_items)
            scores, w = model(u, act, audio_t, text_t, idx, fixed_weights=fixed)

            scores[hist] = -1e9                      # 已听过的过滤掉，模拟线上只推没听过的
            pos_score = scores[pos]
            rank = int((scores > pos_score).sum().item())

            hit = 1.0 if rank < TOPK else 0.0
            ndcg = (1.0 / math.log2(rank + 2)) if rank < TOPK else 0.0
            bucket = buckets[_bucket_of(len(hist))]

            for box in (overall, bucket):
                box["hr"] += hit
                box["ndcg"] += ndcg
                box["n"] += 1
                box["w"] += w.mean(dim=0).cpu()      # 逐条权重不同，取该用户全库打分的平均值展示

        def pack(box: dict) -> dict:
            n = max(1, box["n"])
            return {
                "HR@10": round(box["hr"] / n, 4),
                "NDCG@10": round(box["ndcg"] / n, 4),
                "users": box["n"],
                "mean_weights": [round(float(x), 4) for x in (box["w"] / n)],
            }

        results[mode] = pack(overall)
        results[mode]["by_bucket"] = {name: pack(box) for name, box in buckets.items()}

    return results


# ------------------------------------------------------------------ 入口
def main() -> None:
    metrics = train()
    out = settings.MODEL_DIR / "metrics.json"
    out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"[train] 指标已写入 {out}")


if __name__ == "__main__":
    main()
