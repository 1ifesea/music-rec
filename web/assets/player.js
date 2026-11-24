/* 播放器：固定在页面底部的播放条。

三件事：
1. 播本地音频（服务端按 Range 返回，可拖动进度）；
2. 播放时上报一行播放历史（REQ003）；
3. 如果这次播放来自推荐列表，同时上报一次推荐点击（REQ010）。
*/
import { api } from "./api.js";

const state = {
  song: null,
  reqId: "",
  position: 0,
  fromRecommend: false,
};

let bar = null;
let audioEl = null;

export function initPlayer() {
  if (bar) return;
  bar = document.createElement("div");
  bar.className = "player";
  bar.innerHTML = `
    <div class="title">
      <div class="song-name" data-title>未在播放</div>
      <div class="song-meta" data-meta>点歌曲右侧的播放按钮开始</div>
    </div>
    <audio controls preload="metadata" data-audio></audio>
  `;
  document.body.appendChild(bar);

  audioEl = bar.querySelector("[data-audio]");
  audioEl.addEventListener("play", onFirstPlay);
}

function onFirstPlay() {
  if (!state.song || state.song.__reported) return;
  state.song.__reported = true;
  api.play(state.song.music_id || state.song.id).catch(() => {});
  if (state.fromRecommend) {
    api.click(state.song.music_id || state.song.id, state.position, state.reqId).catch(() => {});
  }
}

export function playSong(song, opts = {}) {
  initPlayer();
  const id = song.music_id || song.id;
  state.song = { ...song, __reported: false };
  state.reqId = opts.reqId || "";
  state.position = opts.position || 0;
  state.fromRecommend = !!opts.fromRecommend;

  bar.querySelector("[data-title]").textContent = song.music_name || "未命名";
  bar.querySelector("[data-meta]").textContent =
    `${song.singer || "未知"} · ${song.style_tag || ""}`;
  audioEl.src = api.audioUrl(id);
  audioEl.play().catch(() => {
    // 浏览器自动播放限制：用户点击进入时通常没问题，失败就等用户手动点播放
  });
}
