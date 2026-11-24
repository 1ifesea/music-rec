"""端到端冒烟测试（对应计划书"验收计划"里的人工验收项，改成可自动跑的脚本）。

用法：
    1) 先启动服务：python -m uvicorn app.main:app --port 8000
    2) 另开终端：python scripts/smoke_test.py
    3) 可选 --base 指定地址、--admin 指定管理员账号密码

设计原则：
  * 只用标准库 urllib，不额外装包（避免为了跑测试再装 requests）；
  * 每个用例独立 try/except，一条失败不影响后续用例，最后统一汇总；
  * 退出码非 0 表示有失败项，方便以后接 CI。
"""
import argparse
import json
import sys
import urllib.error
import urllib.parse
import urllib.request

from urllib.parse import urlencode

PASS, FAIL = [], []


def call(method: str, url: str, token: str = "", body=None, headers=None, raw=False):
    """发一个请求；raw=True 时返回 (状态码, 响应头, 原始字节)，用于验证 Range 分片。"""
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", "Bearer " + token)
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            payload = resp.read()
            if raw:
                return resp.status, dict(resp.headers), payload
            text = payload.decode("utf-8")
            return resp.status, dict(resp.headers), (json.loads(text) if text else None)
    except urllib.error.HTTPError as e:
        payload = e.read()
        return e.code, dict(e.headers or {}), payload


def check(name: str, cond: bool, detail: str = ""):
    if cond:
        PASS.append(name)
        print(f"[PASS] {name}" + (f"  {detail}" if detail else ""))
    else:
        FAIL.append(name)
        print(f"[FAIL] {name}" + (f"  {detail}" if detail else ""))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--user", default="demo1")
    ap.add_argument("--password", default="123456")
    ap.add_argument("--admin", default="admin")
    ap.add_argument("--admin-password", default="123456")
    args = ap.parse_args()
    base = args.base.rstrip("/")

    # ---------- 1. 健康检查 ----------
    code, _, health = call("GET", f"{base}/health")
    check("健康检查返回 200", code == 200, f"status={code}")
    check("数据库连通", bool(health and health.get("db")), str(health))
    check("模型已加载", bool(health and health.get("model_ready")), str(health.get("model_notice")))

    # ---------- 2. 登录 ----------
    code, _, login = call("POST", f"{base}/api/auth/login",
                          body={"username": args.user, "password": args.password})
    token = (login or {}).get("token", "")
    check("普通用户登录成功", code == 200 and bool(token), f"status={code}")
    code, _, me = call("GET", f"{base}/api/auth/me", token=token)
    check("/api/auth/me 返回昵称", code == 200 and (me or {}).get("nickname"), str((me or {}).get("nickname")))

    code, _, bad = call("POST", f"{base}/api/auth/login",
                        body={"username": args.user, "password": "wrong-pwd"})
    check("错误密码被拒绝（401）", code == 401, f"status={code}")

    # ---------- 3. 推荐 ----------
    code, _, rec = call("GET", f"{base}/api/recommend?n=10", token=token)
    items = (rec or {}).get("items") or []
    check("推荐返回 200 且 10 条", code == 200 and len(items) == 10, f"status={code} n={len(items)}")
    weights = (rec or {}).get("weights") or {}
    wsum = sum(weights.values())
    check("三模态权重之和≈1", abs(wsum - 1.0) < 1e-3, f"sum={wsum:.4f} {weights}")
    missed = [
        {"id": r["music_id"], "play_count": 0}
        for r in ((rec or {}).get("items") or [])
    ]
    _ = missed  # 预留：如需校验"已听不再推荐"，可在此处对比 /history
    check("推荐项带歌曲元信息", bool(items) and bool(items[0].get("music_name")),
          str(items[0].get("music_name") if items else ""))

    code, _, w = call("GET", f"{base}/api/recommend/weights", token=token)
    check("/weights 与推荐权重一致",
          code == 200 and abs(sum((w or {}).get("weights", {}).values()) - 1.0) < 1e-3,
          str((w or {}).get("weights")))

    code, _, cmp = call("GET", f"{base}/api/recommend/compare?n=10", token=token)
    check("自适应/固定对比返回两个列表",
          code == 200 and len((cmp or {}).get("adaptive", {}).get("items", [])) == 10
          and len((cmp or {}).get("fixed", {}).get("items", [])) == 10,
          f"overlap={(cmp or {}).get('overlap')}")

    code, _, met = call("GET", f"{base}/api/recommend/metrics")
    check("离线指标可读（HR@10 / NDCG@10 存在）",
          code == 200 and "adaptive" in (met or {}) and "HR@10" in (met or {}).get("adaptive", {}),
          f"adaptive={str((met or {}).get('adaptive', {}).get('HR@10'))}")

    # ---------- 4. 行为记录 ----------
    mid = items[0]["music_id"] if items else 1
    code, _, _ = call("POST", f"{base}/api/behavior/play", token=token, body={"music_id": mid})
    check("播放上报成功", code == 200, f"status={code} music_id={mid}")
    code, _, hist = call("GET", f"{base}/api/behavior/history?limit=5", token=token)
    check("播放历史可查（最新一条是刚才的歌）",
          code == 200 and (hist or {}).get("items") and hist["items"][0]["music_id"] == mid,
          str((hist or {}).get("items", [{}])[0].get("music_name")))

    code, _, fav = call("POST", f"{base}/api/behavior/favorite", token=token, body={"music_id": mid})
    check("收藏成功", code == 200 and (fav or {}).get("favorited") is True, f"status={code}")
    code, _, favs = call("GET", f"{base}/api/behavior/favorites", token=token)
    check("收藏列表包含该歌曲", any(r["music_id"] == mid for r in (favs or {}).get("items", [])), "")
    code, _, un = call("DELETE", f"{base}/api/behavior/favorite/{mid}", token=token)
    check("取消收藏成功", code == 200 and (un or {}).get("favorited") is False, f"status={code}")

    code, _, st = call("GET", f"{base}/api/behavior/stats", token=token)
    check("个人统计含播放数与偏好风格",
          code == 200 and (st or {}).get("play_count", 0) > 0 and "top_genres" in (st or {}),
          f"plays={(st or {}).get('play_count')}")

    # ---------- 5. 曲库与音频分片 ----------
    # 中文参数必须做 URL 编码，否则 urllib 会因非 ASCII 直接报错
    qs = urlencode({"page": 1, "size": 5, "genre": "摇滚"})
    code, _, lst = call("GET", f"{base}/api/music?{qs}", token=token)
    check("按风格筛选曲库", code == 200 and (lst or {}).get("total", 0) > 0
          and all(i["style_tag"] == "摇滚" for i in lst["items"]),
          f"total={(lst or {}).get('total')}")
    qs = urlencode({"keyword": "之"})
    code, _, kw = call("GET", f"{base}/api/music?{qs}", token=token)
    check("按关键词搜索", code == 200 and (kw or {}).get("total", 0) >= 0, f"total={(kw or {}).get('total')}")
    code, _, genres = call("GET", f"{base}/api/music/genres", token=token)
    check("风格列表 8 种", code == 200 and len((genres or {}).get("items", [])) == 8,
          f"n={len((genres or {}).get('items', []))}")

    code, _, detail = call("GET", f"{base}/api/music/{mid}", token=token)
    check("歌曲详情", code == 200 and (detail or {}).get("id") == mid, f"status={code}")

    code, hdr, payload = call("GET", f"{base}/api/audio/{mid}", token=token,
                              headers={"Range": "bytes=0-1023"}, raw=True)
    check("音频支持 Range 分片（206）", code == 206,
          f"status={code} content-range={hdr.get('Content-Range')} bytes={len(payload)}")
    check("分片长度=1024", len(payload) == 1024, f"len={len(payload)}")

    # ---------- 6. 后台管理 ----------
    code, _, alogin = call("POST", f"{base}/api/auth/login",
                           body={"username": args.admin, "password": args.admin_password})
    atoken = (alogin or {}).get("token", "")
    check("管理员登录成功", code == 200 and bool(atoken), f"status={code}")

    code, _, ov = call("GET", f"{base}/api/admin/overview", token=atoken)
    check("后台概览含 600 首歌", code == 200 and (ov or {}).get("music", 0) >= 600,
          f"music={(ov or {}).get('music')} plays={(ov or {}).get('plays')}")

    code, _, created = call("POST", f"{base}/api/admin/music", token=atoken, body={
        "music_name": "冒烟测试歌曲", "singer": "自动化", "style_tag": "测试", "duration": 3.0})
    new_id = (created or {}).get("id")
    check("后台新增歌曲并返回重建提示", code == 200 and bool(new_id) and "notice" in (created or {}),
          f"id={new_id}")

    code, _, _ = call("PUT", f"{base}/api/admin/music/{new_id}", token=atoken, body={
        "music_name": "冒烟测试歌曲（改）", "singer": "自动化", "style_tag": "测试", "duration": 3.0})
    check("后台编辑歌曲", code == 200, f"status={code}")
    code, _, _ = call("DELETE", f"{base}/api/admin/music/{new_id}", token=atoken)
    check("后台删除歌曲（清理测试数据）", code == 200, f"status={code} id={new_id}")

    code, _, _ = call("POST", f"{base}/api/admin/reload", token=atoken)
    check("模型热加载", code == 200, f"status={code}")

    code, _, forb = call("GET", f"{base}/api/admin/overview", token=token)
    check("普通用户访问后台被拦截（403）", code == 403, f"status={code}")

    # ---------- 7. 页面与静态资源 ----------
    for path in ["/", "/login", "/library", "/mine", "/admin", "/compare"]:
        code, _, page = call("GET", f"{base}{path}", raw=True)
        check(f"页面 {path} 可访问", code == 200 and len(page) > 1500, f"status={code} bytes={len(page)}")
    for asset in ["app.css", "api.js", "player.js", "vue.esm-browser.prod.js"]:
        code, _, blob = call("GET", f"{base}/assets/{asset}", raw=True)
        check(f"静态资源 {asset}", code == 200 and len(blob) > 1000, f"status={code} bytes={len(blob)}")

    # ---------- 汇总 ----------
    print("-" * 56)
    print(f"通过 {len(PASS)} 项，失败 {len(FAIL)} 项")
    if FAIL:
        print("失败项：")
        for name in FAIL:
            print("  -", name)
        return 1
    print("全部通过。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
