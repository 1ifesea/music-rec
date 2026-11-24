"""多模态门控融合模型 MMGate（Multi-Modal Gate）。

结构对应计划书 5.2 节的「多模态融合层：采用 MLP 自适应学习不同模态权重」，
但做了一个关键改进：把权重做成**显式**的。

    音频特征 ──► 音频编码器 ──► a ┐
    文本特征 ──► 文本编码器 ──► t ├─► 加权求和 ──► 打分 MLP ──► 分数
    用户隐向量 ⊙ 歌曲隐向量 ─► b ┘        ▲
                                       │
    用户隐向量 + 活跃度 ──► 门控网络 ──► softmax(w音频, w文本, w行为)

为什么要显式门控：如果只把三路特征拼起来丢进 MLP，权重藏在参数里，谁也拿不出来；
做成门控之后，每个用户的三个权重可以直接导出打印，答辩时能当场展示「同一个模型，
不同用户拿到的权重不同」——这才是「自适应学习权重」的证据。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MMGate(nn.Module):
    def __init__(self, n_items: int, audio_dim: int = 64, text_dim: int = 32,
                 emb_dim: int = 32, hidden: int = 64):
        super().__init__()
        self.n_items = n_items
        self.emb_dim = emb_dim

        # 歌曲隐向量：行为模态的载体（类似 GMF 的 item embedding）
        self.item_embedding = nn.Embedding(n_items, emb_dim)

        # 音频模态编码器
        self.audio_enc = nn.Sequential(
            nn.Linear(audio_dim, 32), nn.ReLU(),
            nn.Linear(32, emb_dim), nn.ReLU(),
        )
        # 文本模态编码器
        self.text_enc = nn.Sequential(
            nn.Linear(text_dim, 32), nn.ReLU(),
            nn.Linear(32, emb_dim), nn.ReLU(),
        )
        # 门控网络：输入「用户向量 + 活跃度 + 三模态向量」，逐条（用户 × 歌曲）算权重。
        # 只用用户向量驱动时信号太弱，退化成常数权重反而引入噪声；
        # 把三模态向量也喂进去，模型才能判断"这首歌上哪一模态更可信"。
        self.gate = nn.Sequential(
            nn.Linear(emb_dim * 4 + 1, 32), nn.ReLU(),
            nn.Linear(32, 3),
        )
        # 融合后的打分层
        self.fusion = nn.Sequential(
            nn.Linear(emb_dim * 3 + 3, hidden), nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, 1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.item_embedding.weight, std=0.05)
        for m in (self.audio_enc, self.text_enc, self.gate, self.fusion):
            for layer in m:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)

    # ------------------------------------------------------------ 用户向量
    def user_vector(self, history_indices: list[int] | None) -> torch.Tensor:
        """用户隐向量 = 历史歌曲隐向量的均值；无历史时为零向量（冷启动）。"""
        if not history_indices:
            return torch.zeros(self.emb_dim)
        idx = torch.as_tensor(history_indices, dtype=torch.long)
        vec = self.item_embedding(idx).mean(dim=0)
        return F.normalize(vec, dim=-1)

    # ------------------------------------------------------------ 前向
    def forward(self, user_vec: torch.Tensor, activity: torch.Tensor,
                audio: torch.Tensor, text: torch.Tensor, item_idx: torch.Tensor,
                fixed_weights: bool = False):
        """返回 (分数, 三模态权重)。

        user_vec: (B, d)   activity: (B,) 0~1
        audio:    (B, 64)  text:     (B, 32)
        item_idx: (B,)     歌曲隐向量下标
        """
        item_vec = self.item_embedding(item_idx)           # 行为模态的歌曲侧
        a_vec = self.audio_enc(audio)
        t_vec = self.text_enc(text)
        b_vec = user_vec * item_vec                        # 行为模态：用户与歌曲交互

        if fixed_weights:
            # 对照组：三模态固定等权，用于证明「自适应」确实带来增益
            weights = torch.full((user_vec.size(0), 3), 1.0 / 3.0, device=user_vec.device)
        else:
            # 门控同时看到"是谁"（用户向量、活跃度）和"这首歌是什么"（三模态向量）
            gate_in = torch.cat([user_vec, activity.unsqueeze(-1), a_vec, t_vec, b_vec], dim=-1)
            weights = torch.softmax(self.gate(gate_in), dim=-1)

        fused = (weights[:, 0:1] * a_vec
                 + weights[:, 1:2] * t_vec
                 + weights[:, 2:3] * b_vec)

        logits = self.fusion(torch.cat([fused, user_vec, item_vec, weights], dim=-1))
        return logits.squeeze(-1), weights

    # ------------------------------------------------------------ 在线打分
    @torch.no_grad()
    def score_all_items(self, user_vec: torch.Tensor, activity: float,
                        audio_all: torch.Tensor, text_all: torch.Tensor,
                        fixed_weights: bool = False):
        """对全库歌曲打分，返回 (分数向量(n,), 权重向量(3,))。"""
        n = audio_all.size(0)
        u = user_vec.unsqueeze(0).expand(n, -1)
        act = torch.full((n,), float(activity))
        idx = torch.arange(n)
        logits, weights = self.forward(u, act, audio_all, text_all, idx, fixed_weights)
        # 逐条权重不同，返回"该用户在全库上的平均权重"，用于界面展示与答辩说明
        return logits, weights.mean(dim=0)

    # ------------------------------------------------------------ 存取
    def save(self, path) -> None:
        torch.save({
            "state_dict": self.state_dict(),
            "n_items": self.n_items,
            "audio_dim": self.audio_enc[0].in_features,
            "text_dim": self.text_enc[0].in_features,
            "emb_dim": self.emb_dim,
        }, path)

    @staticmethod
    def load(path):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model = MMGate(
            n_items=ckpt["n_items"],
            audio_dim=ckpt["audio_dim"],
            text_dim=ckpt["text_dim"],
            emb_dim=ckpt["emb_dim"],
        )
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        return model
