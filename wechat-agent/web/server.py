"""Web UI backend for WeChat Auto-Reply Agent.

Provides:
- Start/stop agent control
- Real-time log streaming (SSE)
- Configuration read/write
- Visual chat-region calibration
"""

import os
import sys
import json
import time
import queue
import threading
import io
import base64

# Add parent dir so we can import src modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, render_template, request, jsonify, Response, send_file
from PIL import Image

from src.capture import find_wechat_window, get_window_rect, capture_region
from src.ocr import extract_text

app = Flask(__name__)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(BASE, "config.json")

# Agent state
_agent_thread = None
_agent_stop = threading.Event()
_log_queue = queue.Queue(maxsize=200)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _push_log(msg):
    """Push a log message to the SSE queue (non-blocking)."""
    ts = time.strftime("%H:%M:%S")
    try:
        _log_queue.put_nowait(f"[{ts}] {msg}")
    except queue.Full:
        pass  # drop oldest log if full


def _load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_config(cfg):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=4)


# ---------------------------------------------------------------------------
# Agent runner (background thread)
# ---------------------------------------------------------------------------

def _run_agent(config):
    """Run the agent loop in a background thread."""
    from src.capture import find_wechat_window, get_window_rect, capture_region
    from src.ocr import extract_text
    from src.responder import generate_reply
    from src.inputter import send_message_direct
    from src.safety import RateLimiter, in_allowed_window
    from src.chat_type import detect_chat_type

    _push_log("Agent 启动中...")

    # Window check
    result = find_wechat_window(config["wechat"]["window_title"])
    if not result:
        _push_log("错误: 未找到微信窗口")
        return
    hwnd, title = result
    _push_log(f"找到窗口: {title}")

    # Load personality
    personality_file = os.path.join(BASE, config["personality_file"])
    if os.path.exists(personality_file):
        with open(personality_file, "r", encoding="utf-8") as f:
            personality = f.read().strip()
    else:
        personality = "用户没有提供风格描述"

    knowledge_path = os.path.join(BASE, "knowledge.txt")
    if os.path.exists(knowledge_path):
        with open(knowledge_path, "r", encoding="utf-8") as f:
            knowledge = f.read().strip()
        if knowledge:
            personality += "\n\n## 额外背景知识\n" + knowledge

    friends_dir = config.get("friends_dir", "personality/friends")
    if not os.path.isabs(friends_dir):
        friends_dir = os.path.join(BASE, friends_dir)

    # Safety
    safety_cfg = config["safety"]
    rate_limiter = RateLimiter(
        max_per_hour=safety_cfg["max_replies_per_hour"],
        cooldown_min=safety_cfg.get("cooldown_min", 5),
        cooldown_max=safety_cfg.get("cooldown_max", 10),
        burst_limit=safety_cfg.get("burst_limit", 3),
        burst_cooldown=safety_cfg.get("burst_cooldown", 30),
    )

    def get_chat_region(hwnd, region_config):
        left, top, right, bottom = get_window_rect(hwnd)
        win_w = right - left
        win_h = bottom - top
        mode = region_config.get("mode", "pixel")
        if mode == "percent":
            chat_left = left + int(win_w * region_config["x_percent"])
            chat_top = top + int(win_h * region_config["y_percent"])
            chat_width = int(win_w * region_config["w_percent"])
            chat_height = int(win_h * region_config["h_percent"])
        else:
            chat_left = left + region_config["x_offset"]
            chat_top = top + region_config["y_offset"]
            chat_width = region_config["width"]
            chat_height = region_config["height"]
        return {
            "left": chat_left,
            "top": chat_top,
            "width": min(chat_width, right - chat_left),
            "height": min(chat_height, bottom - chat_top),
        }

    chat_region = get_chat_region(hwnd, config["wechat"]["chat_region"])
    last_text = ""
    sent_messages = []
    last_replied_hash = ""
    stable_chat_type = "group"
    private_streak = 0
    iteration = 0

    _push_log("Agent 开始运行")

    # We inline the agent loop — identical logic to agent.py
    import re as _re_mod
    import hashlib as _hashlib
    import random as _random

    def text_hash(t):
        return _hashlib.md5(t.encode("utf-8", errors="ignore")).hexdigest()

    while not _agent_stop.is_set():
        iteration += 1

        # Time-of-day check
        if not in_allowed_window(safety_cfg["allowed_start_hour"],
                                 safety_cfg["allowed_end_hour"]):
            time.sleep(5)
            continue

        # Capture
        try:
            img = capture_region(chat_region)
        except Exception as e:
            _push_log(f"截图失败: {e}")
            time.sleep(2)
            continue

        # OCR
        ocr_cfg = config["ocr"]
        current_text = extract_text(img, lang=ocr_cfg["lang"],
                                     tesseract_path=ocr_cfg.get("tesseract_path"))

        if not current_text:
            time.sleep(config["safety"].get("scan_interval_seconds", 3))
            continue

        # Detect new messages (simplified)
        len_diff = abs(len(current_text) - len(last_text))
        if len_diff <= 10:
            last_text = current_text
            time.sleep(config["safety"].get("scan_interval_seconds", 3))
            continue

        _push_log(f"检测到新消息 (len_diff={len_diff})")

        # Own message filter
        if sent_messages and any(msg in current_text and msg not in last_text
                                  for msg in sent_messages[-3:]):
            last_text = current_text
            time.sleep(config["safety"].get("scan_interval_seconds", 3))
            continue

        # Rate limit
        if not rate_limiter.allowed():
            last_text = current_text
            time.sleep(1)
            continue

        # Extract latest messages (last 6 lines)
        def extract_latest(raw, max_lines=6):
            lines = raw.strip().split("\n")
            filtered = []
            for line in lines:
                s = line.strip()
                if not s:
                    continue
                is_own = any(msg in s for msg in sent_messages)
                if not is_own:
                    filtered.append(s)
            recent = filtered[-max_lines:] if len(filtered) > max_lines else filtered
            return "\n".join(recent)

        clean_text = extract_latest(current_text)

        # Dedup
        lines = [l for l in clean_text.strip().split("\n") if l.strip()]
        if lines:
            latest_hash = _hashlib.md5(lines[-1].encode("utf-8", errors="ignore")).hexdigest()
            if latest_hash == last_replied_hash:
                last_text = current_text
                time.sleep(config["safety"].get("scan_interval_seconds", 3))
                continue
        else:
            latest_hash = ""

        # Chat type detection
        chat_info = detect_chat_type(current_text, title)
        raw_type = chat_info["type"]

        if raw_type == "group":
            private_streak = 0
            if stable_chat_type != "group":
                _push_log("切换回群聊模式")
            stable_chat_type = "group"
        else:
            private_streak += 1
            if stable_chat_type == "group" and private_streak >= 3:
                _push_log("切换为私聊模式")
                stable_chat_type = "private"

        chat_type = stable_chat_type
        friend_name = chat_info.get("friend_name") if chat_type == "private" else None

        # Load friend info
        friend_info = ""
        if chat_type == "private" and friend_name:
            friend_path = os.path.join(friends_dir, f"{friend_name}.txt")
            if os.path.exists(friend_path):
                with open(friend_path, "r", encoding="utf-8") as f:
                    friend_info = f.read().strip()
                if friend_info:
                    friend_info = f"\n\n## {friend_name} 的资料\n{friend_info}"
                    _push_log(f"私聊模式 — 好友: {friend_name} (已加载资料)")
            else:
                _push_log(f"私聊模式 — 好友: {friend_name}")

        # Build context
        context_parts = []
        if sent_messages:
            last_reply = sent_messages[-1]
            context_parts.append(
                f'[系统提示] 你上一次回复的内容是："{last_reply}"。'
                f"注意：不要重复这个话题，除非对方追问。"
            )
        if chat_type == "group":
            context_parts.append("[消息格式] 每条消息标注了发送者，如 [张三]: 你好")
        else:
            context_parts.append(f"[消息格式] 这是你和 {friend_name or '对方'} 的私聊")
        context_parts.append(clean_text)

        # AI call
        try:
            _push_log("咨询 AI...")
            llm_cfg = config["llm"]
            result = generate_reply(
                conversation_text="\n".join(context_parts),
                personality_text=personality,
                api_key=llm_cfg["api_key"],
                model=llm_cfg["model"],
                max_tokens=llm_cfg["max_tokens"],
                temperature=llm_cfg["temperature"],
                provider=llm_cfg.get("provider", "deepseek"),
                chat_type=chat_type,
                friend_name=friend_name or "",
                friend_info=friend_info,
            )
        except Exception as e:
            _push_log(f"AI 调用失败: {e}")
            last_text = current_text
            time.sleep(3)
            continue

        if not result.get("should_reply"):
            _push_log("AI 判断无需回复")
            last_text = current_text
            time.sleep(config["safety"].get("scan_interval_seconds", 3))
            continue

        reply_text = result.get("message", "").strip()
        if not reply_text:
            last_text = current_text
            time.sleep(config["safety"].get("scan_interval_seconds", 3))
            continue

        delay = result.get("delay_seconds", 3)
        _push_log(f'准备回复: "{reply_text}" ({delay}s)')

        # Send
        try:
            time.sleep(delay)
            input_x = chat_region["left"] + chat_region["width"] // 2
            input_y = chat_region["top"] + chat_region["height"] + 50
            send_message_direct(hwnd, reply_text, input_x, input_y,
                                wpm=config["typing"]["wpm"],
                                typo_rate=config["typing"]["typo_rate"],
                                pause_rate=config["typing"]["pause_rate"])
            _push_log("已发送!")
            rate_limiter.record()
            sent_messages.append(reply_text)
            last_replied_hash = latest_hash
        except Exception as e:
            _push_log(f"发送失败: {e}")

        last_text = current_text
        time.sleep(config["safety"].get("scan_interval_seconds", 3))

    _push_log("Agent 已停止")


# ---------------------------------------------------------------------------
# Routes — Page
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


# ---------------------------------------------------------------------------
# Routes — Agent control
# ---------------------------------------------------------------------------

@app.route("/api/start", methods=["POST"])
def api_start():
    global _agent_thread, _agent_stop
    if _agent_thread and _agent_thread.is_alive():
        return jsonify({"ok": False, "error": "Agent 已在运行"})

    config = _load_config()
    if not config["llm"]["api_key"] or config["llm"]["api_key"].startswith("YOUR_"):
        return jsonify({"ok": False, "error": "请先配置 API Key"})

    _agent_stop.clear()
    _agent_thread = threading.Thread(target=_run_agent, args=(config,), daemon=True)
    _agent_thread.start()
    return jsonify({"ok": True, "message": "Agent 已启动"})


@app.route("/api/stop", methods=["POST"])
def api_stop():
    global _agent_stop
    _agent_stop.set()
    _push_log("正在停止...")
    return jsonify({"ok": True, "message": "停止信号已发送"})


@app.route("/api/status")
def api_status():
    running = _agent_thread is not None and _agent_thread.is_alive()
    return jsonify({"running": running})


# ---------------------------------------------------------------------------
# Routes — Logs (Server-Sent Events)
# ---------------------------------------------------------------------------

@app.route("/api/logs")
def api_logs():
    def stream():
        while True:
            try:
                msg = _log_queue.get(timeout=1)
                yield f"data: {msg}\n\n"
            except queue.Empty:
                yield ": keepalive\n\n"
    return Response(stream(), mimetype="text/event-stream")


@app.route("/api/logs/recent")
def api_logs_recent():
    """Return recent log entries as JSON (for initial load)."""
    logs = []
    while not _log_queue.empty():
        try:
            logs.append(_log_queue.get_nowait())
        except queue.Empty:
            break
    # Put them back
    for msg in logs:
        try:
            _log_queue.put_nowait(msg)
        except queue.Full:
            pass
    return jsonify({"logs": logs[-50:]})  # last 50


# ---------------------------------------------------------------------------
# Routes — Configuration
# ---------------------------------------------------------------------------

@app.route("/api/config", methods=["GET"])
def api_get_config():
    cfg = _load_config()
    # Mask API key
    cfg["llm"]["api_key"] = _mask_key(cfg["llm"]["api_key"])
    return jsonify(cfg)


@app.route("/api/config", methods=["POST"])
def api_save_config():
    data = request.get_json()
    current = _load_config()

    # Merge incoming fields
    if "api_key" in data.get("llm", {}):
        new_key = data["llm"]["api_key"]
        if not new_key.startswith("***"):  # not masked
            current["llm"]["api_key"] = new_key
    if "model" in data.get("llm", {}):
        current["llm"]["model"] = data["llm"]["model"]
    if "provider" in data.get("llm", {}):
        current["llm"]["provider"] = data["llm"]["provider"]
    if "personality_file" in data:
        current["personality_file"] = data["personality_file"]

    # Chat region
    if "chat_region" in data.get("wechat", {}):
        current["wechat"]["chat_region"] = data["wechat"]["chat_region"]

    _save_config(current)
    return jsonify({"ok": True})


def _mask_key(key):
    if not key or key.startswith("YOUR_"):
        return key
    if len(key) <= 8:
        return "***"
    return key[:4] + "***" + key[-4:]


# ---------------------------------------------------------------------------
# Routes — Calibration
# ---------------------------------------------------------------------------

@app.route("/api/calibrate/screenshot", methods=["POST"])
def api_calibrate_screenshot():
    """Capture the WeChat window and return as base64 image."""
    config = _load_config()
    result = find_wechat_window(config["wechat"]["window_title"])
    if not result:
        return jsonify({"ok": False, "error": "未找到微信窗口"})

    hwnd, title = result
    left, top, right, bottom = get_window_rect(hwnd)
    win_w = right - left
    win_h = bottom - top

    # Capture the entire WeChat window
    region = {"left": left, "top": top, "width": win_w, "height": win_h}
    img = capture_region(region)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()

    return jsonify({
        "ok": True,
        "image": f"data:image/png;base64,{b64}",
        "window_title": title,
        "win_width": win_w,
        "win_height": win_h,
        "current_region": config["wechat"]["chat_region"],
    })


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"  WeChat Agent Web UI")
    print(f"  打开浏览器访问 http://127.0.0.1:5050")
    print(f"  按 Ctrl+C 停止")
    app.run(host="127.0.0.1", port=5050, debug=False, threaded=True)


if __name__ == "__main__":
    main()
