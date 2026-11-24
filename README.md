# 多模态音乐个性化推荐系统

软件工程课程设计项目。在线音乐平台基础业务 + 音频 / 文本 / 用户行为三模态自适应融合推荐。

**核心看点**：多模态权重不是人工设定的，而是由一个门控网络根据用户画像实时算出，
并且可以把每个用户的三个权重直接打印出来 —— 这是「模型自适应学习权重」可验证的证据。

---

## 技术栈

| 层次 | 选型 | 说明 |
| --- | --- | --- |
| 前端 | Vue 3（浏览器版，无需构建）+ 手写 CSS | 不依赖 Node/npm，双击页面即可看 |
| 后端 | Python 3.11 + FastAPI + uvicorn | 单进程，一条命令启动 |
| 数据库 | MySQL 8.0 | 与计划书 5.3 的 ER 设计一致 |
| 算法 | PyTorch（CPU）+ Librosa + scikit-learn | 特征离线算、模型启动加载一次 |

## 目录结构

```
.
├── app/                 # 后端
│   ├── main.py          # FastAPI 入口（建表、加载模型、挂静态资源）
│   ├── config.py        # 配置（读 .env）
│   ├── db.py            # PyMySQL 连接池 + 参数化查询
│   ├── schema.py        # 建库建表 DDL
│   ├── security.py      # PBKDF2 口令哈希 + HMAC 签名令牌
│   ├── deps.py          # 登录态依赖
│   └── routers/         # auth / music / behavior / recommend / admin
├── recsys/              # 算法
│   ├── features.py      # 音频 64 维 + 文本 32 维特征
│   ├── model.py         # MMGate：三模态门控融合网络
│   ├── train.py         # BPR 训练 + 留一法评估
│   └── serving.py       # 在线推荐服务（模型常驻内存）
├── scripts/             # 数据与训练脚本
│   ├── prepare_data.py  # 合成演示集 / 导入真实音频
│   ├── load_tracks.py   # tracks.csv → music 表
│   ├── seed_simulated.py# 模拟用户与交互
│   ├── build_features.py# 批量提特征
│   ├── train.py         # 训练入口
│   └── smoke_test.py    # 端到端冒烟测试（41 项）
├── web/                 # 前端页面
│   ├── index.html       # 首页：推荐列表 + 三模态权重
│   ├── library.html     # 音乐库：搜索 / 筛选 / 详情 / 播放
│   ├── mine.html        # 我的：收藏 / 历史 / 画像
│   ├── compare.html     # 权重对比：自适应 vs 固定等权
│   ├── admin.html       # 后台：歌曲增删改 + 概览
│   └── assets/          # app.css / api.js / player.js / vue 运行时
├── data/                # 数据（音频、特征、模型，均不入库版本管理）
├── docs/                # 数据集说明与设计文档
├── requirements.txt
├── .env                 # 数据库配置（已 gitignore）
└── start.ps1            # 一键启动
```

## 快速开始

前提：装好 Python 3.11 与 MySQL 8.0，并把数据库密码写进 `.env`（参考 `.env.example`）。

```powershell
# 1) 建虚拟环境并装依赖
py -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 2) 生成数据集（默认 600 首合成演示音频 + 元数据）
.venv\Scripts\python.exe scripts\prepare_data.py

# 3) 建库、导入歌曲、造模拟行为
.venv\Scripts\python.exe scripts\load_tracks.py
.venv\Scripts\python.exe scripts\seed_simulated.py

# 4) 提特征 + 训练（约 2~4 分钟）
.venv\Scripts\python.exe scripts\build_features.py
.venv\Scripts\python.exe scripts\train.py

# 5) 启动服务
.venv\Scripts\python.exe -m uvicorn app.main:app --port 8000
```

浏览器打开 <http://127.0.0.1:8000>。

也可以直接运行 `start.ps1`（只启动服务）或 `setup.ps1`（把 1~4 步一次跑完）。

### 演示账号

| 账号 | 密码 | 说明 |
| --- | --- | --- |
| demo1 | 123456 | 偏好「流行」，有 60 条播放历史 |
| demo2 | 123456 | 偏好「电子」，有 60 条播放历史 |
| demo3 | 123456 | 新用户，零历史 —— 用来看冷启动时行为权重怎么掉下去 |
| admin | 123456 | 管理员，可进后台 |

用 demo1 和 demo2 对比首页的三模态权重，能直接看到权重随人变化。

## 数据集：现在用的是哪份

默认走**合成演示集**：脚本用 numpy 合成 600 段 6 秒音频（8 种风格、音色/速度/谐波各不相同），
元数据（歌名、歌手、专辑、风格、描述）也在脚本里生成。

这样处理的理由是：课程阶段手上没有音频素材，而系统必须能完整跑通
「特征提取 → 训练 → 推荐 → 播放」。**报告里必须如实写明这批音频是合成的。**

换成真实数据只需一条命令，业务代码零改动：

```powershell
# 把 FMA-small / GTZAN / 自建歌单放进任意目录，然后：
.venv\Scripts\python.exe scripts\prepare_data.py --mode import --src D:\fma_small\fma_small --meta D:\fma_small\tracks.csv
# 之后重跑 load_tracks --reset / build_features --force / train
```

关于「歌词」：本系统文本模态用的是**曲目元数据 + 风格标签 + 曲目描述**，不含真歌词。
原因见 `docs/dataset.md`（版权、可获得性、以及中文歌词语料的实际困难）。

## 算法说明

三段结构，对应计划书 5.2 的「多模态融合层」：

1. **音频模态**：Librosa 提取 MFCC(20 维均值+标准差) + 色度(12) + 频谱对比度(7) + 谱质心/带宽/滚降 + 零交叉率 + 节奏，池化成 64 维向量。
2. **文本模态**：歌名/歌手/专辑/风格/描述拼成文本，TF-IDF(char 1-2 gram) 后 SVD 降到 32 维。
3. **行为模态**：歌曲隐向量与用户隐向量的 Hadamard 积（GMF 思路），刻画「这个用户和这首歌的交互」。

**融合方式**：门控网络**逐条**算权重——对「用户 × 这一首歌」这一对组合，输入是
`用户隐向量 + 活跃度 + 该曲音频向量 + 该曲文本向量 + 该曲行为向量`，softmax 输出三个权重
`w音频 / w文本 / w行为`，加权求和后送入打分 MLP。
用逐条门控而不是全局一套权重是有实测依据的：全局权重只能拿「用户画像」当输入，在这批数据上
信号太弱，反而比固定等权更差（20 轮时 HR@10 0.157 vs 0.183）；改成逐条之后优势才显现。
用户历史为空时活跃度为 0，门控会把行为权重压低，冷启动自动退化成「看内容」，
这也是计划书里「解决冷启动」的落点。

**训练**：BPR pairwise loss（正样本为用户最后一条交互，负样本随机采样），
外加向均匀权重收缩的 KL 正则（系数 0.05），防止门控塌成「只用某一个模态」。
**早停**：每 5 轮按验证集 NDCG@10 打分，保留最好那一轮的权重。
**评估**：留一法，全库排序后算 HR@10 / NDCG@10，并与「三模态固定等权」对照组比较。

### 实测结果（600 首合成曲目 / 274 个有足够历史的模拟用户 / 最优轮次 25）

| 指标 | 自适应权重（本系统） | 固定等权（对照） |
| --- | --- | --- |
| HR@10（总体） | 0.2555 | 0.2555 |
| NDCG@10（总体） | **0.1508** | 0.1407 |
| 平均权重（音频/文本/行为） | 33% / 36% / 31% | 33% / 33% / 33% |
| 冷启动用户 HR@10（历史 < 5 条，12 人） | 0.0833 | 0.0833 |
| 中等活跃用户 NDCG@10（5~19 条，33 人） | **0.1611** | 0.1442 |
| 活跃用户 NDCG@10（≥ 20 条，229 人） | **0.1559** | 0.1463 |

写法上建议如实陈述：同一批候选里两者「命中率」持平，自适应权重的**排序质量**更好
（NDCG@10 相对提升约 7%），提升主要来自中等活跃用户。随机推荐 HR@10 ≈ 10/600 ≈ 1.7%，
所以 25.6% 大约是随机的 15 倍。轮次影响很大：20 轮欠拟合、25~30 轮最好、60 轮明显过拟合
（HR@10 掉到 0.135），这就是加早停的直接原因。

训练完的指标会写到 `data/models/metrics.json`，也可以在首页或
`GET /api/recommend/metrics` 里看到。

## 自测

服务启动后另开一个终端：

```powershell
.venv\Scripts\python.exe scripts\smoke_test.py
```

脚本只用标准库 urllib，覆盖 41 项检查：健康检查、登录与错误密码、推荐与权重归一化、
自适应/固定对比、播放与收藏、搜索与风格筛选、音频 Range 分片（206 状态码 + 1024 字节）、
后台增删改与权限拦截、六个页面与四个静态资源的可访问性。全部通过时退出码为 0。

## 需求对应

| 需求 | 实现位置 |
| --- | --- |
| REQ001 注册登录 / 改资料 | `app/routers/auth.py` + `web/login.html` |
| REQ002 音乐浏览 | `GET /api/music` + `web/library.html` |
| REQ003 播放 + 记录行为 | `GET /api/audio/{id}`（支持 Range）+ `POST /api/behavior/play` |
| REQ004 收藏与历史 | `app/routers/behavior.py` + `web/mine.html` |
| REQ005 多模态特征提取 | `recsys/features.py` + `scripts/build_features.py` + `music_feature` 表 |
| REQ006 自适应融合推荐 | `recsys/model.py` + `recsys/train.py` + `GET /api/recommend` |
| REQ007 推荐展示 | `web/index.html` |
| REQ008 歌曲搜索 | `GET /api/music?keyword=` |
| REQ009 后台管理 | `app/routers/admin.py` + `web/admin.html` |
| REQ010 推荐点击统计 | `POST /api/recommend/click` + `GET /api/recommend/stats` |

## 接口速查

启动后访问 <http://127.0.0.1:8000/docs> 可以看到完整的 Swagger 文档。

## 已知限制

- 音频不做过媒体解码，直接返回本地文件（计划书 3.2 已排除流媒体解码）。
- 后台新增/修改歌曲后，需要重跑 `build_features.py` 与 `train.py`，再调用 `/api/admin/reload`。
  这是「离线训练 + 在线服务」的必然边界，不是遗漏。
- 交互数据是模拟生成的，模型指标仅代表该模拟分布下的效果。
