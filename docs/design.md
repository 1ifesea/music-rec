# 系统设计说明（可作为《系统设计文档》底稿）

## 一、总体架构

三层，与计划书 5.2 一致；差别在于算法层被拆成「离线」与「在线」两段。

```
浏览器（Vue3 浏览器版 + 手写 CSS）
        │  HTTP/JSON（Bearer 令牌）
        ▼
FastAPI 业务层
   ├─ auth      注册 / 登录 / 改资料
   ├─ music     歌曲列表、搜索、详情、音频流（Range）
   ├─ behavior  播放记录、收藏、个人统计
   ├─ recommend 推荐列表、权重、对比、点击统计
   └─ admin     歌曲增删改、概览、模型热加载
        │
        ├──── SQL（参数化）────► MySQL 8.0
        │
        └──── 进程内调用 ──────► 推荐引擎（模型常驻内存）
                                  ├ 特征库 features.npz（离线算好）
                                  └ MMGate 打分 → Top-N
```

**为什么要拆离线/在线**：Librosa 提一首歌的特征约 0.2 秒，如果放在请求链路里，
一次推荐要解析几十首歌的音频，3 秒的性能目标必然破。所以音频只在离线阶段读取一次，
线上只做向量打分（600 首歌一次批量前向约 2~5 ms）。

## 二、数据库设计

| 表 | 关键字段 | 作用 |
| --- | --- | --- |
| `user` | id, username, password_hash, nickname, role | 账号；role 区分 user / admin / sim（模拟用户） |
| `music` | id, music_name, singer, album, style_tag, description, lyric, audio_path, duration, play_count | 歌曲元数据；description 进文本模态 |
| `user_play_history` | user_id, music_id, play_time | 播放行为，推荐训练的主数据源 |
| `user_favorite` | user_id, music_id, collect_time（唯一键 user+music） | 收藏，作为强正向信号 |
| `music_feature` | music_id, audio_feature(JSON), text_feature(JSON) | 按计划书要求把向量落库，便于查库核对 |
| `rec_click` | user_id, music_id, req_id, position, clicked | REQ010 推荐点击统计 |

说明：`music_feature` 里的两个 LONGTEXT 是给文档与审查用的**镜像**；
训练和在线服务实际读的是 `data/features/features.npz`（同一份数据导出，带压缩、读取更快）。

## 三、接口清单

| 方法 | 路径 | 鉴权 | 说明 |
| --- | --- | --- | --- |
| POST | /api/auth/register | — | 注册并返回令牌 |
| POST | /api/auth/login | — | 登录 |
| GET | /api/auth/me | 用户 | 当前用户 |
| PUT | /api/auth/profile | 用户 | 修改昵称 |
| GET | /api/music | — | 分页 + keyword + genre |
| GET | /api/music/genres | — | 风格列表与数量 |
| GET | /api/music/{id} | — | 歌曲详情 |
| GET | /api/audio/{id} | — | 音频（支持 Range，206 分段） |
| POST | /api/behavior/play | 用户 | 记录播放 + 热度 +1 |
| GET | /api/behavior/history | 用户 | 最近播放 |
| GET | /api/behavior/favorites | 用户 | 收藏列表 |
| POST/DELETE | /api/behavior/favorite | 用户 | 收藏 / 取消 |
| GET | /api/behavior/stats | 用户 | 个人画像 |
| GET | /api/recommend | 用户 | 推荐列表 + 三模态权重 |
| GET | /api/recommend/weights | 用户 | 只返回权重 |
| GET | /api/recommend/compare | 用户 | 自适应 vs 固定等权对照 |
| POST | /api/recommend/click | 用户 | 记录推荐点击 |
| GET | /api/recommend/metrics | — | 离线评估结果 |
| GET | /api/recommend/stats | — | 点击统计 |
| GET/POST/PUT/DELETE | /api/admin/... | 管理员 | 概览、歌曲管理、热加载 |

完整交互式文档：服务启动后访问 `/docs`。

## 四、算法设计

### 4.1 特征

**音频（64 维）**：MFCC 20 维取均值与标准差（40）＋色度 12 ＋ 频谱对比度 7 ＋
谱质心/带宽/滚降（3）＋零交叉率（1）＋节奏（1，谱通量自相关估计的 BPM 归一化）。
整段按帧池化后 L2 归一化。

> 节奏是自己算的，没有用 `librosa.beat.beat_track`。原因见第五节第 4 条。

**文本（32 维）**：`歌名 + 歌手 + 专辑 + 风格 + 描述 +（可选）歌词` 拼成一段话，
TF-IDF（char 1~2 gram，3000 维上限）后用 SVD 降到 32 维，行向量 L2 归一化。
编码器保存在 `data/models/text_encoder.joblib`，后台新增歌曲时复用，保证训练与推理同一套编码。

**行为（32 维）**：歌曲隐向量 `e_i` 与用户隐向量 `u` 的 Hadamard 积。用户隐向量本身由历史歌曲
隐向量的均值得到：

```
u = normalize( mean_{i ∈ H(u)} e_i )
```

没有历史时 `u = 0`，行为模态自然为 0。

### 4.2 门控融合

```
输入：u（用户隐向量）、a（用户活跃度）、x_audio（64）、x_text（32）、e_i（歌曲隐向量）

a_vec = AudioEnc(x_audio)        → 32 维
t_vec = TextEnc(x_text)          → 32 维
b_vec = u ⊙ e_i                  → 32 维

w = softmax( Gate([u, a]) )       → (w_audio, w_text, w_behavior)，和为 1

fused = w_audio·a_vec + w_text·t_vec + w_behavior·b_vec
score = MLP([fused, u, e_i, w])  → 标量
```

关键点：`w` 是**显式**的中间产物，可以随请求返回、可以打印、可以画成条形图。
对照组把 `w` 固定成 (1/3, 1/3, 1/3)，其余结构完全相同——这样两组的差异只来自权重来源。

冷启动的处理也在这里：新用户 `u = 0`、`a = 0`，门控输入告诉模型"这个用户没有行为证据"，
训练后模型会把 `w_behavior` 压低，推荐退化为内容驱动。这正好是计划书 1.1 里"解决冷启动"的落点。

### 4.3 训练与评估

- 损失：BPR pairwise，`-log σ(score(u, pos) - score(u, neg))`，正样本取用户最后一条交互，
  负样本随机采样并排除已交互项。
- 优化：Adam，lr=1e-3，30 轮，batch=256，梯度裁剪 5.0。
- 评估：留一法。对每个用户，屏蔽其历史后对全库打分，看目标歌曲排在第几，算 HR@10 与 NDCG@10。
- 对照组：同一模型、同一测试集，仅把 `w` 固定为等权。

### 4.4 在线服务

`recsys/serving.py` 里的 `Recommender` 是单例：进程启动时加载一次模型与特征，
之后每个请求只做：取用户历史 → 算 `u` 和活跃度 → 全库打分 → 过滤已听/已收藏 → 取 Top-N。
模型文件缺失或与歌曲库不一致时，自动降级为「热度推荐」并在响应里给 `notice`，保证演示不中断。

## 五、关键设计取舍（答辩常被问到的点）

1. **为什么不用 ORM？** 课程项目要求能讲清每条 SQL；参数化查询直接可见，也避免 ORM 版本问题。
2. **为什么前端不用 Vite / Element-Plus？** 环境里没有 Node。用 Vue3 的浏览器版 + 手写 CSS，
   页面直接由 FastAPI 静态返回，答辩机器上不需要 `npm install`，少一类翻车点。
   技术栈实质仍是 Vue3，只是构建方式不同，报告里加一句说明即可。
3. **为什么文本模态用 TF-IDF + SVD 而不是 Transformers？** 一是中文短文本上句向量收益有限，
   二是要额外下载 500 MB 以上模型，本机内存 16 GB、装完 torch 后余量紧张；
   三是离线可跑。换回 Transformers 只需改 `features.py` 里的一个函数。
4. **为什么自己算节奏？** `librosa.beat.beat_track` 在当前 numba 版本上会触发底层访问冲突（进程直接退出），
   换成谱通量自相关估计后稳定，且原理讲得清、可复现。
5. **特征为什么既存 npz 又存 MySQL？** npz 是训练/推理的实际输入（快）；`music_feature` 表是
   计划书要求的数据模型，同时方便老师查库核对向量确实存在。两者由同一个脚本写出，不会不一致。
6. **后台改了歌为什么不立刻生效？** 因为模型是在固定特征集上训练的，新增歌曲意味着特征与下标
   都要变。正确流程是重跑 `build_features` → `train` → `/api/admin/reload`。这是"离线训练 +
   在线服务"的必然边界，不是遗漏，答辩时可以讲成一条设计约束。

## 六、性能与验证

| 项目 | 目标 | 实测（本机 16G / RTX4060，但模型跑 CPU） |
| --- | --- | --- |
| 页面响应 | ≤ 2s | 列表/详情类接口 < 100 ms |
| 推荐生成 | ≤ 3s | 全库 600 首打分 + 过滤，约 5~20 ms |
| 数据库查询 | ≤ 1s | 单表查询毫秒级 |
| 并发 | ≥ 30 | uvicorn 单进程 + 连接池 8，本地压测足够 |

## 七、已知限制

- 音频不做流媒体解码，直接以 Range 方式返回本地文件（计划书 3.2 已排除）。
- 交互数据为模拟生成，指标只代表该模拟分布。
- 未做移动端 APP、社交、付费模块（计划书 3.2 已排除）。
