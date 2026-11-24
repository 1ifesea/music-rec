"""多模态特征提取：音频模态 + 文本模态。

音频模态：Librosa 提取 MFCC / 色度 / 频谱对比度 / 谱统计量，按帧做均值+标准差池化，
          压成一个固定 64 维向量。这是计划书 5.2 节「MFCC 梅尔频谱音频特征」的落地实现。
文本模态：用 TF-IDF(char n-gram) + SVD 降到 32 维。
          这里刻意没有用 Transformers：
            1) 中文歌词/标签短文本上，句向量模型的收益远小于它 500MB 的模型体积；
            2) 课程演示机器只有 16G 内存，且需要离线可跑。
          如果后续要换回 Transformers，只需替换 text_feature_from_corpus 一个函数。
"""
from __future__ import annotations

import json
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

AUDIO_DIM = 64
TEXT_DIM = 32
SAMPLE_RATE = 22050


# ------------------------------------------------------------------ 音频模态
def estimate_tempo(y: np.ndarray, sr: int, hop: int = 512) -> float:
    """纯 numpy 的节奏估计：谱通量 → 自相关 → 主周期对应的 BPM。

    没有用 librosa.beat.beat_track，因为它在部分 numba 版本上会触发底层崩溃；
    自己算反而更可控，也便于在报告里讲清楚原理。
    """
    mag = np.abs(librosa.stft(y, hop_length=hop))
    flux = np.maximum(0.0, np.diff(mag, axis=1)).sum(axis=0)
    if flux.size < 8:
        return 0.0
    flux = flux - flux.mean()
    ac = np.correlate(flux, flux, mode="full")[flux.size - 1:]
    if ac[0] <= 0:
        return 0.0
    ac = ac / ac[0]

    fps = sr / hop                                     # 每秒帧数
    min_lag = int(fps / (200 / 60))                    # 上限 200 BPM
    max_lag = min(int(fps / (40 / 60)), ac.size - 1)   # 下限 40 BPM
    if max_lag <= min_lag:
        return 0.0
    lag = min_lag + int(np.argmax(ac[min_lag:max_lag]))
    return 60.0 * fps / max(1, lag)


def extract_audio_feature(audio_path: str | Path) -> np.ndarray:
    """返回长度 64 的音频特征向量（L2 归一化）。"""
    y, sr = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)
    if y.size < sr // 4:                      # 太短的片段直接补零，避免特征 NaN
        y = np.pad(y, (0, sr // 4 - y.size))

    # 去掉静音段，避免整段空白音频把统计量拉偏
    y, _ = librosa.effects.trim(y, top_db=30)
    if y.size < sr // 4:
        y, sr = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)          # (20, T)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)            # (12, T)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)    # (7, T)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)    # (1, T)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)  # (1, T)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)      # (1, T)
    zcr = librosa.feature.zero_crossing_rate(y)                 # (1, T)

    mfcc_stat = np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)])   # 40
    rest = np.concatenate([
        chroma.mean(axis=1),                                            # 12
        contrast.mean(axis=1),                                          # 7
        [centroid.mean(), bandwidth.mean(), rolloff.mean()],            # 3
        [zcr.mean()],                                                   # 1
    ])
    tempo = estimate_tempo(y, sr)

    feat = np.concatenate([mfcc_stat, rest, [min(tempo, 220.0) / 200.0]])  # 40+12+7+3+1+1 = 64
    feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
    norm = np.linalg.norm(feat)
    return (feat / norm) if norm > 0 else feat


# ------------------------------------------------------------------ 文本模态
def build_text_corpus(records: list[dict]) -> list[str]:
    """把歌曲的标题、歌手、专辑、风格标签、描述（以及可选歌词）拼成一段文本。

    课程项目里没有稳定的中文歌词来源，所以文本模态用「曲目元数据 + 标签 + 描述」，
    这在工程上是合理取舍：它仍然是文本语义特征，只是语料换成可控的元数据。
    """
    corpus = []
    for r in records:
        parts = [
            str(r.get("music_name") or ""),
            str(r.get("singer") or ""),
            str(r.get("album") or ""),
            str(r.get("style_tag") or ""),
            str(r.get("description") or ""),
        ]
        lyric = (r.get("lyric") or "").strip()
        if lyric:
            parts.append(lyric[:200])
        corpus.append(" ".join(p for p in parts if p))
    return corpus


def fit_text_encoder(corpus: list[str], dim: int = TEXT_DIM):
    """在训练集上拟合 TF-IDF + SVD，返回编码器（后续保存到磁盘）。"""
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 2), max_features=3000, min_df=1)
    tfidf = vectorizer.fit_transform(corpus)
    n_comp = min(dim, max(2, min(tfidf.shape) - 1))
    svd = TruncatedSVD(n_components=n_comp, random_state=2026)
    reduced = svd.fit_transform(tfidf)
    # 维度不够时补零，保证下游维度固定
    if reduced.shape[1] < dim:
        pad = np.zeros((reduced.shape[0], dim - reduced.shape[1]))
        reduced = np.hstack([reduced, pad])
    return vectorizer, svd


def encode_text(corpus: list[str], vectorizer, svd) -> np.ndarray:
    tfidf = vectorizer.transform(corpus)
    reduced = svd.transform(tfidf)
    if reduced.shape[1] < TEXT_DIM:
        reduced = np.hstack([reduced, np.zeros((reduced.shape[0], TEXT_DIM - reduced.shape[1]))])
    return normalize(reduced)


# ------------------------------------------------------------------ 落盘工具
def save_features(audio_feats: np.ndarray, text_feats: np.ndarray, music_ids: list[int], path: Path) -> None:
    np.savez_compressed(path, audio=audio_feats, text=text_feats, music_ids=np.array(music_ids))


def load_features(path: Path):
    data = np.load(path)
    return data["audio"], data["text"], data["music_ids"].tolist()


def to_json(vec: np.ndarray) -> str:
    """向量转 JSON，写入 music_feature 表供文档与审查使用。"""
    return json.dumps([round(float(x), 6) for x in vec], ensure_ascii=False)
