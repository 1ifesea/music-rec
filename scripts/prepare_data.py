"""数据集准备脚本。

两条数据路径，都能跑通整个系统：

1) 默认 --mode synth：离线合成 600 首「可控音色的演示音频」+ 元数据。
   目的不是冒充真实音乐，而是让系统在「还没有音频素材」的情况下也能完整演示
   （特征提取 → 训练 → 推荐 → 播放）。报告里必须如实写明这批音频是合成的。

2) --mode import：导入你自己的真实音频目录（FMA-small / GTZAN / 自建歌单均可），
   自动扫描 wav/mp3/flac 并可选读 CSV 元数据，输出同样格式的 tracks.csv。
   换成真实数据后，重跑 build_features → train 即可，业务代码一行都不用改。

用法：
    python scripts/prepare_data.py                       # 合成演示集
    python scripts/prepare_data.py --count 800           # 指定数量
    python scripts/prepare_data.py --mode import --src D:\\fma_small\\fma_small --meta D:\\fma_small\\tracks.csv
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import settings  # noqa: E402

SR = 22050
DURATION = 6.0

# 八个风格，参数上刻意拉开音色、速度与谐波差异，保证 MFCC 特征可分
GENRE_PROFILE = {
    "流行": dict(tempo=100, root=57, chord=[0, 4, 7], harmonics=6, decay=2.2,
                 noise=0.02, beat=0.5, tremolo=0.15, distort=False,
                 mood=["明亮", "温柔", "轻快"], instr=["钢琴", "吉他", "合成器"]),
    "摇滚": dict(tempo=140, root=52, chord=[0, 3, 7], harmonics=9, decay=1.6,
                 noise=0.06, beat=0.85, tremolo=0.05, distort=True,
                 mood=["激烈", "躁动", "张扬"], instr=["电吉他", "架子鼓", "贝斯"]),
    "民谣": dict(tempo=80, root=55, chord=[0, 4, 7, 9], harmonics=4, decay=3.2,
                 noise=0.01, beat=0.3, tremolo=0.1, distort=False,
                 mood=["安静", "温暖", "叙事"], instr=["木吉他", "口琴", "手鼓"]),
    "电子": dict(tempo=128, root=48, chord=[0, 7, 12], harmonics=5, decay=2.0,
                 noise=0.04, beat=0.95, tremolo=0.3, distort=False,
                 mood=["律动", "未来感", "冷冽"], instr=["合成器", "鼓机", "采样"]),
    "爵士": dict(tempo=92, root=50, chord=[0, 4, 7, 10, 14], harmonics=7, decay=2.6,
                 noise=0.02, beat=0.45, tremolo=0.2, distort=False,
                 mood=["慵懒", "即兴", "醇厚"], instr=["萨克斯", "钢琴", "低音提琴"]),
    "古典": dict(tempo=72, root=60, chord=[0, 4, 7, 12], harmonics=3, decay=3.6,
                 noise=0.005, beat=0.25, tremolo=0.08, distort=False,
                 mood=["庄严", "舒展", "典雅"], instr=["弦乐", "管风琴", "竖琴"]),
    "嘻哈": dict(tempo=90, root=45, chord=[0, 3, 10], harmonics=4, decay=1.4,
                 noise=0.07, beat=1.0, tremolo=0.1, distort=False,
                 mood=["街头", "自信", "重低音"], instr=["808鼓", "说唱采样", "搓盘"]),
    "金属": dict(tempo=160, root=41, chord=[0, 1, 6], harmonics=12, decay=1.2,
                 noise=0.12, beat=0.9, tremolo=0.03, distort=True,
                 mood=["轰鸣", "极速", "黑暗"], instr=["失真吉他", "双踩鼓", "低吼"]),
}

TITLE_HEADS = {
    "流行": ["夏夜", "晚风", "微光", "第七", "半糖", "晴空", "心跳", "旧照片"],
    "摇滚": ["荒野", "逆流", "燃烧", "裂缝", "锈迹", "雷鸣", "脱轨", "花岗岩"],
    "民谣": ["小镇", "三月", "麦田", "巷口", "旧船", "归途", "南山", "纸伞"],
    "电子": ["脉冲", "霓虹", "零度", "回路", "频率", "硅基", "夜航", "比特"],
    "爵士": ["蓝调", "午后", "烟灰", "酒馆", "第七街", "慵懒", "丝绒", "低语"],
    "古典": ["序曲", "奏鸣", "安魂", "晨祷", "冬之", "回旋", "小夜", "赋格"],
    "嘻哈": ["街区", "底线", "押韵", "地心", "手稿", "低位", "反拍", "方言"],
    "金属": ["深渊", "铁幕", "末日", "灰烬", "雷霆", "残响", "黑曜", "断层"],
}
TITLE_TAILS = ["", "之歌", "序曲", "札记", "片段", "回声", "手记", "速写", "进行曲", "小调"]
ARTISTS = ["星野乐队", "南街三人", "青柠计划", "白鲸工作室", "回声房间", "拾光组合",
           "第七日", "海雾乐团", "铜管小组", "夜行电台", "木石声场", "逆光乐队"]


def midi_to_freq(midi: int) -> float:
    return 440.0 * 2 ** ((midi - 69) / 12)


def synth_audio(profile: dict, seed: int) -> np.ndarray:
    """合成一段 6 秒音频：和弦 + 打击 + 噪声，参数由风格决定。"""
    rng = np.random.default_rng(seed)
    n = int(SR * DURATION)
    t = np.arange(n) / SR
    beat = 60.0 / profile["tempo"]

    # 1) 和弦音：按谐波叠加，谐波数量与衰减率决定音色亮度
    sig = np.zeros(n)
    root = profile["root"] + int(rng.integers(-2, 3))
    for note in profile["chord"]:
        f0 = midi_to_freq(root + note)
        for h in range(1, profile["harmonics"] + 1):
            amp = 1.0 / (h ** 1.3)
            phase = rng.uniform(0, 2 * np.pi)
            vibrato = 0.004 * np.sin(2 * np.pi * 4.5 * t + phase)
            sig += amp * np.sin(2 * np.pi * f0 * h * (1 + vibrato) * t + phase)

    # 2) 节拍包络：每个 beat 一次衰减
    env = np.exp(-((t % beat) / (beat * profile["decay"])) ** 1.2)
    env *= 1 - profile["tremolo"] * (0.5 + 0.5 * np.sin(2 * np.pi * 2.2 * t))
    sig *= env

    # 3) 打击：每个 beat 起点的噪声爆发
    if profile["beat"] > 0.3:
        for b in range(int(DURATION / beat) + 1):
            pos = int(b * beat * SR)
            length = int(0.08 * SR)
            if pos + length < n:
                burst = rng.normal(0, 1, length) * np.exp(-np.arange(length) / (0.02 * SR))
                sig[pos:pos + length] += profile["beat"] * 0.35 * burst

    # 4) 底噪，用于区分「干净」与「脏」风格
    sig += rng.normal(0, profile["noise"], n)

    # 5) 失真（摇滚/金属）
    if profile["distort"]:
        sig = np.tanh(sig * 2.5)

    sig = sig / (np.max(np.abs(sig)) + 1e-9) * 0.92
    return sig.astype(np.float32)


def make_track(idx: int, genre: str, profile: dict, rng: random.Random) -> dict:
    head = rng.choice(TITLE_HEADS[genre])
    tail = rng.choice(TITLE_TAILS)
    if tail:
        title = f"{head}{tail}"
    else:
        title = head + rng.choice(["之夜", "的风", "手记", "小记"])
    singer = rng.choice(ARTISTS)
    album = f"{rng.choice(['同名', '第二张', '现场', '精选'])}{rng.choice(['专辑', 'EP', '录音室集'])}"
    mood = "、".join(rng.sample(profile["mood"], 2))
    instr = "、".join(rng.sample(profile["instr"], 2))
    tempo_desc = "轻缓" if profile["tempo"] < 95 else ("中等" if profile["tempo"] < 130 else "快速")
    desc = (f"{genre}风格曲目，情绪{mood}，编曲以{instr}为主，"
            f"节奏{tempo_desc}（约{profile['tempo']}BPM），适合{genre}乐迷日常循环播放。")
    return {
        "music_name": title,
        "singer": singer,
        "album": album,
        "style_tag": genre,
        "description": desc,
        "lyric": "",
        "duration": DURATION,
    }


def build_synth(count: int, out_dir: Path) -> Path:
    """生成合成音频与元数据，返回 tracks.csv 路径。"""
    settings.ensure_dirs()
    out_audio = settings.AUDIO_DIR
    out_audio.mkdir(parents=True, exist_ok=True)
    genre_names = list(GENRE_PROFILE.keys())
    per = max(1, count // len(genre_names))
    rows = []
    music_id = 1
    rng = random.Random(2026)

    for genre in genre_names:
        profile = GENRE_PROFILE[genre]
        for k in range(per):
            meta = make_track(k, genre, profile, rng)
            audio = synth_audio(profile, seed=music_id * 7919 + k)
            path = out_audio / f"{music_id:04d}.wav"
            sf.write(str(path), audio, SR, subtype="PCM_16")
            rows.append({"music_id": music_id, "audio_path": str(path.relative_to(ROOT)).replace("\\", "/"),
                         "duration": DURATION, **meta})
            music_id += 1
            if music_id % 100 == 0:
                print(f"[synth] 已生成 {music_id - 1} 首")

    csv_path = settings.META_DIR / "tracks.csv"
    write_csv(csv_path, rows)
    return csv_path


def scan_import(src: Path, meta_csv: Path | None) -> Path:
    """导入本地真实音频：扫描目录，可选读 CSV 元数据。"""
    exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
    files = sorted([p for p in src.rglob("*") if p.suffix.lower() in exts])
    if not files:
        raise SystemExit(f"在 {src} 下没有找到音频文件")

    meta_map = {}
    if meta_csv and meta_csv.exists():
        with meta_csv.open(encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                key = (row.get("path") or row.get("id") or "").strip()
                if key:
                    meta_map[Path(key).stem] = row

    rows = []
    for i, p in enumerate(files, start=1):
        m = meta_map.get(p.stem, {})
        genre = (m.get("genre") or m.get("style_tag") or "未分类").strip()
        rows.append({
            "music_id": i,
            "audio_path": str(p.relative_to(ROOT)) if ROOT in p.parents else str(p),
            "duration": float(m.get("duration") or 0) or 30.0,
            "music_name": (m.get("title") or m.get("music_name") or p.stem).strip(),
            "singer": (m.get("artist") or m.get("singer") or "未知").strip(),
            "album": (m.get("album") or "").strip(),
            "style_tag": genre,
            "description": (m.get("description") or f"{genre} 风格曲目，来源公开音频数据集。").strip(),
            "lyric": (m.get("lyric") or "").strip(),
        })
    out = settings.META_DIR / "tracks.csv"
    write_csv(out, rows)
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = ["music_id", "audio_path", "duration", "music_name", "singer",
              "album", "style_tag", "description", "lyric"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["synth", "import"], default="synth")
    parser.add_argument("--count", type=int, default=600, help="合成模式下的歌曲数量")
    parser.add_argument("--src", type=str, default="", help="导入模式下的音频目录")
    parser.add_argument("--meta", type=str, default="", help="导入模式下的元数据 CSV")
    args = parser.parse_args()

    if args.mode == "synth":
        path = build_synth(args.count, settings.AUDIO_DIR)
        print(f"[synth] 完成，元数据：{path}")
    else:
        if not args.src:
            raise SystemExit("导入模式必须提供 --src 音频目录")
        meta = Path(args.meta) if args.meta else None
        path = scan_import(Path(args.src), meta)
        print(f"[import] 完成，元数据：{path}")


if __name__ == "__main__":
    main()
