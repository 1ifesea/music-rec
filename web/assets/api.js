/* 统一的前端请求封装：令牌放 localStorage，401 自动跳登录页 */

const TOKEN_KEY = "musicrec_token";
const USER_KEY = "musicrec_user";

export function getToken() {
  return localStorage.getItem(TOKEN_KEY) || "";
}

export function getUser() {
  try {
    return JSON.parse(localStorage.getItem(USER_KEY) || "null");
  } catch {
    return null;
  }
}

export function setSession(token, user) {
  localStorage.setItem(TOKEN_KEY, token);
  localStorage.setItem(USER_KEY, JSON.stringify(user || {}));
}

export function clearSession() {
  localStorage.removeItem(TOKEN_KEY);
  localStorage.removeItem(USER_KEY);
}

export function requireLogin() {
  if (!getToken()) {
    location.href = "/login";
    return false;
  }
  return true;
}

export async function request(path, { method = "GET", body = null, auth = true } = {}) {
  const headers = {};
  if (body !== null) headers["Content-Type"] = "application/json";
  if (auth && getToken()) headers["Authorization"] = "Bearer " + getToken();

  const resp = await fetch(path, {
    method,
    headers,
    body: body === null ? undefined : JSON.stringify(body),
  });

  if (resp.status === 401) {
    clearSession();
    if (!location.pathname.startsWith("/login")) location.href = "/login";
    throw new Error("登录已失效");
  }
  const text = await resp.text();
  const data = text ? JSON.parse(text) : {};
  if (!resp.ok) {
    throw new Error(data.detail || `请求失败（${resp.status}）`);
  }
  return data;
}

export const api = {
  // 用户
  login: (username, password) =>
    request("/api/auth/login", { method: "POST", body: { username, password }, auth: false }),
  register: (username, password, nickname) =>
    request("/api/auth/register", { method: "POST", body: { username, password, nickname }, auth: false }),
  me: () => request("/api/auth/me"),
  updateProfile: (nickname) => request("/api/auth/profile", { method: "PUT", body: { nickname } }),

  // 音乐库
  genres: () => request("/api/music/genres", { auth: false }),
  musicList: (params = {}) => {
    const qs = new URLSearchParams(params).toString();
    return request("/api/music" + (qs ? "?" + qs : ""), { auth: false });
  },
  musicDetail: (id) => request(`/api/music/${id}`, { auth: false }),
  audioUrl: (id) => `/api/audio/${id}`,

  // 行为
  play: (musicId) => request("/api/behavior/play", { method: "POST", body: { music_id: musicId } }),
  history: (limit = 50) => request(`/api/behavior/history?limit=${limit}`),
  favorites: () => request("/api/behavior/favorites"),
  favorite: (musicId) => request("/api/behavior/favorite", { method: "POST", body: { music_id: musicId } }),
  unfavorite: (musicId) => request(`/api/behavior/favorite/${musicId}`, { method: "DELETE" }),
  behaviorStats: () => request("/api/behavior/stats"),

  // 推荐
  recommend: (n = 10, mode = "adaptive") => request(`/api/recommend?n=${n}&mode=${mode}`),
  weights: () => request("/api/recommend/weights"),
  compare: (n = 10) => request(`/api/recommend/compare?n=${n}`),
  click: (musicId, position, reqId) =>
    request("/api/recommend/click", { method: "POST", body: { music_id: musicId, position, req_id: reqId || "" } }),
  metrics: () => request("/api/recommend/metrics", { auth: false }),
  clickStats: () => request("/api/recommend/stats"),

  // 后台
  overview: () => request("/api/admin/overview"),
  adminMusic: () => request("/api/admin/music"),
  adminUsers: () => request("/api/admin/users"),
  createMusic: (payload) => request("/api/admin/music", { method: "POST", body: payload }),
  updateMusic: (id, payload) => request(`/api/admin/music/${id}`, { method: "PUT", body: payload }),
  deleteMusic: (id) => request(`/api/admin/music/${id}`, { method: "DELETE" }),
  reloadModel: () => request("/api/admin/reload", { method: "POST" }),
};
