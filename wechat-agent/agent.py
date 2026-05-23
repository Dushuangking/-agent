#!/usr/bin/env python3
"""
WeChat Auto-Reply Agent.

Uses screenshot + OCR to read messages, Claude API to generate replies,
and simulated keyboard input to send them. Designed for low-risk,
occasional use with friends.

Usage:
    python agent.py                    # Start with default config
    python agent.py --once             # Check once then exit
    python agent.py --dry-run          # No typing, just show what would be sent
    python agent.py --calibrate        # Show chat area to help configure region
"""

import os
import sys
import json
import time
import random
import hashlib
import argparse

from src.capture import find_wechat_window, get_window_rect, focus_window, capture_region
from src.ocr import extract_text
from src.responder import generate_reply
from src.inputter import send_message_direct
from src.safety import RateLimiter, in_allowed_window, human_delay
from src.chat_type import detect_chat_type

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(path="config.json"):
    """Load configuration from JSON file."""
    base = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base, path)

    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # Resolve relative paths
    if config.get('personality_file'):
        config['personality_file'] = os.path.join(
            base, config['personality_file'])

    # API key: config file > environment variable
    if not config['llm']['api_key']:
        config['llm']['api_key'] = config['llm']['api_key'] or os.environ.get('DEEPSEEK_API_KEY')

    return config


def load_personality(path):
    """Load personality text from file."""
    if not os.path.exists(path):
        return "用户没有提供风格描述，请用自然、随意的中文回复。"
    with open(path, 'r', encoding='utf-8') as f:
        return f.read().strip()


def load_knowledge(path):
    """Load knowledge base from file (optional)."""
    if not os.path.exists(path):
        return ""
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    if not content:
        return ""
    return "\n\n## 额外背景知识（你需要知道的）\n" + content


def load_friend_info(friend_name, friends_dir):
    """Load a friend's personality file from the friends directory.

    Returns the file content if found, empty string otherwise.
    """
    if not friend_name:
        return ""
    file_path = os.path.join(friends_dir, f'{friend_name}.txt')
    if not os.path.exists(file_path):
        return ""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    if not content:
        return ""
    return f"\n\n## {friend_name} 的资料（你了解的信息）\n{content}"


# ---------------------------------------------------------------------------
# Chat region calculation
# ---------------------------------------------------------------------------

def get_chat_region(hwnd, region_config):
    """Calculate the screen region for the chat message area.

    Supports two modes:
    - 'percent': calculate from window dimensions (survives resize)
    - 'pixel' (default): use absolute offsets
    """
    left, top, right, bottom = get_window_rect(hwnd)
    win_w = right - left
    win_h = bottom - top

    mode = region_config.get('mode', 'pixel')
    if mode == 'percent':
        chat_left = left + int(win_w * region_config['x_percent'])
        chat_top = top + int(win_h * region_config['y_percent'])
        chat_width = int(win_w * region_config['w_percent'])
        chat_height = int(win_h * region_config['h_percent'])
    else:
        chat_left = left + region_config['x_offset']
        chat_top = top + region_config['y_offset']
        chat_width = region_config['width']
        chat_height = region_config['height']

    return {
        'left': chat_left,
        'top': chat_top,
        'width': min(chat_width, right - chat_left),
        'height': min(chat_height, bottom - chat_top),
    }


def get_input_position(hwnd, region_config):
    """Estimate the input box click position (bottom of chat area)."""
    left, top, right, bottom = get_window_rect(hwnd)
    win_w = right - left
    win_h = bottom - top

    mode = region_config.get('mode', 'pixel')
    if mode == 'percent':
        input_x = left + int(win_w * (region_config['x_percent'] + region_config['w_percent'] / 2))
        input_y = top + int(win_h * (region_config['y_percent'] + region_config['h_percent'] + 0.03))
    else:
        input_x = left + region_config['x_offset'] + region_config['width'] // 2
        input_y = top + region_config['y_offset'] + region_config['height'] + 50
    return input_x, input_y


# ---------------------------------------------------------------------------
# Text change detection
# ---------------------------------------------------------------------------

def text_hash(text):
    """Short hash for comparing OCR results."""
    return hashlib.md5(text.encode('utf-8', errors='ignore')).hexdigest()


def is_different(text_a, text_b, threshold=0.85):
    """
    Check if two OCR texts are meaningfully different.
    Uses fuzzy comparison to tolerate OCR noise.
    """
    if not text_a or not text_b:
        return True

    # Significant length change = real new content
    len_diff = abs(len(text_a) - len(text_b))
    if len_diff > 50:
        return True

    # Normalize: collapse whitespace and lowercase for comparison
    import re
    def normalize(t):
        return re.sub(r'\s+', ' ', t).strip().lower()

    norm_a = normalize(text_a)
    norm_b = normalize(text_b)

    # If one is nearly a subset of the other, just new content appended
    if len(norm_b) > len(norm_a) and len(norm_a) > 10:
        if norm_a[:int(len(norm_a) * 0.8)] in norm_b:
            return True  # old text found inside new text = real new message

    # Jaccard on character trigrams (more robust than line-based for OCR noise)
    def trigrams(s):
        return {s[i:i+3] for i in range(len(s) - 2)}

    tri_a = trigrams(norm_a)
    tri_b = trigrams(norm_b)
    if not tri_a or not tri_b:
        return len_diff > 10

    union = len(tri_a | tri_b)
    intersection = len(tri_a & tri_b)
    jaccard = intersection / union if union > 0 else 0

    return jaccard < threshold


# ---------------------------------------------------------------------------
# Calibration mode
# ---------------------------------------------------------------------------

def calibrate(config):
    """Help the user calibrate chat region by showing a screenshot."""
    result = find_wechat_window(config['wechat']['window_title'])
    if not result:
        print("[!] 未找到微信窗口，请确保微信已打开")
        return

    hwnd, title = result
    print(f"[*] 找到窗口: {title}")

    # Gentle focus attempt
    focus_window(hwnd)
    time.sleep(0.2)

    region = get_chat_region(hwnd, config['wechat']['chat_region'])
    img = capture_region(region)
    calib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'calibrate.png')
    img.save(calib_path)
    print(f"[*] 截图已保存到: {calib_path}")
    print(f"[*] 当前区域: {region}")
    print("[*] 请打开截图检查是否覆盖了聊天消息区域，如有偏差请调整 config.json 中的 chat_region")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_loop(config, personality, friends_dir, once=False, dry_run=False):
    """Main agent loop."""
    result = find_wechat_window(config['wechat']['window_title'])
    if not result:
        print("[!] 未找到微信窗口，请确保微信已打开")
        sys.exit(1)
    hwnd, title = result

    print(f"[*] 找到窗口: {title}")
    print(f"[*] 个性文件: {config.get('personality_file', 'N/A')}")
    print(f"[*] 模式: {'仅一次' if once else '持续监控'}{' (Dry-run)' if dry_run else ''}")
    print()

    safety_cfg = config['safety']
    rate_limiter = RateLimiter(
        max_per_hour=safety_cfg['max_replies_per_hour'],
        cooldown_min=safety_cfg.get('cooldown_min', 5),
        cooldown_max=safety_cfg.get('cooldown_max', 10),
        burst_limit=safety_cfg.get('burst_limit', 3),
        burst_cooldown=safety_cfg.get('burst_cooldown', 30),
    )

    chat_region = get_chat_region(hwnd, config['wechat']['chat_region'])
    last_text = ""
    sent_messages = []  # all replies we've sent, to filter them from OCR
    baseline_text = ""  # OCR snapshot right after last send — used to compute delta
    last_replied_hash = ""  # dedup: skip if the latest message hasn't changed
    # Sticky chat-type detection — prevents flapping from noise
    stable_chat_type = "group"  # assume group by default (safer)
    private_streak = 0

    def extract_latest_messages(raw_text, max_lines=6):
        """Return only the last N lines of conversation for the LLM.

        This ensures the AI focuses on the newest messages instead of
        re-reading the entire chat history every time.
        """
        lines = raw_text.strip().split('\n')
        # Filter out our own replies
        filtered = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            is_own = False
            for msg in sent_messages:
                if msg in stripped:
                    is_own = True
                    break
            if not is_own:
                filtered.append(stripped)

        # Only keep the last few lines — the latest messages
        recent = filtered[-max_lines:] if len(filtered) > max_lines else filtered
        return '\n'.join(recent)

    def latest_message_hash(text):
        """Hash the last meaningful line for dedup."""
        lines = [l for l in text.strip().split('\n') if l.strip()]
        if not lines:
            return ""
        return hashlib.md5(lines[-1].encode('utf-8', errors='ignore')).hexdigest()

    def next_tick(msg=None):
        """Sleep for the scan interval, then check if we should stop (--once)."""
        if msg:
            print(f"  [{iteration}] {msg}")
        if once:
            print("[*] 单次检查完成")
            return True  # signal to break
        jitter = random.uniform(-1.5, 1.5)
        time.sleep(max(1, safety_cfg['scan_interval_seconds'] + jitter))
        return False

    iteration = 0
    while True:
        iteration += 1

        # Time-of-day check
        if not in_allowed_window(safety_cfg['allowed_start_hour'],
                                 safety_cfg['allowed_end_hour']):
            if next_tick("不在允许时段，等待..."):
                break
            continue

        # Capture (no focus needed — using mss which captures screen directly)
        try:
            img = capture_region(chat_region)
        except Exception as e:
            if next_tick(f"截图失败: {e}"):
                break
            continue

        # OCR
        ocr_cfg = config['ocr']
        current_text = extract_text(img, lang=ocr_cfg['lang'],
                                    tesseract_path=ocr_cfg.get('tesseract_path'))

        if not current_text:
            if next_tick("OCR 为空，跳过"):
                break
            continue

        # Detect new messages
        if not is_different(current_text, last_text):
            last_text = current_text
            if next_tick():
                break
            continue

        # New text appeared — but skip if it's just our own replies
        if sent_messages and any(msg in current_text and msg not in last_text for msg in sent_messages[-3:]):
            last_text = current_text
            if next_tick("检测到自己的回复出现，跳过"):
                break
            continue

        print(f"  [{iteration}] 检测到新消息...")
        # Save latest capture for debugging
        img.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'last_capture.png'))

        # Rate limit check
        if not rate_limiter.allowed():
            wait = rate_limiter.next_available_in()
            last_text = current_text
            if next_tick(f"频率限制，需等待 {wait}s"):
                break
            continue

        # AI decide
        if not config['llm']['api_key']:
            last_text = current_text
            if next_tick("未配置 LLM API Key，跳过回复"):
                break
            continue

        try:
            # Only look at the latest messages (last 6 lines max)
            clean_text = extract_latest_messages(current_text)

            # Dedup: skip if the latest message is the same one we last replied to
            latest_hash = latest_message_hash(clean_text)
            if latest_hash and latest_hash == last_replied_hash:
                last_text = current_text
                if next_tick("最新消息未变化，跳过"):
                    break
                continue

            # Detect chat type with hysteresis (prevents flapping)
            chat_info = detect_chat_type(current_text, title)
            raw_type = chat_info['type']

            if raw_type == 'group':
                private_streak = 0
                if stable_chat_type != 'group':
                    print(f"  [{iteration}] 检测到多个发送人 → 切换回群聊模式")
                stable_chat_type = 'group'
            else:  # private
                private_streak += 1
                if stable_chat_type == 'group' and private_streak < 3:
                    # Stay in group mode — a single private-looking frame
                    # could just mean only 1 person sent recent messages
                    pass
                else:
                    if stable_chat_type != 'private':
                        print(f"  [{iteration}] 持续检测为私聊 → 切换为私聊模式")
                    stable_chat_type = 'private'

            chat_type = stable_chat_type
            friend_name = chat_info.get('friend_name') if chat_type == 'private' else None
            if chat_type == 'private' and friend_name:
                friend_info = load_friend_info(friend_name, friends_dir)
                print(f"  [{iteration}] 私聊模式，好友: {friend_name}" +
                      (f" (已加载资料)" if friend_info else " (无资料文件)"))
            else:
                friend_info = ""

            print(f"  [{iteration}] 咨询 AI...")
            print(f"  [{iteration}] 最新消息: {clean_text[:120]}...")

            # Build context: tell the AI what it last said so it won't repeat
            context_parts = []
            if sent_messages:
                last_reply = sent_messages[-1]
                context_parts.append(f'[系统提示] 你上一次回复的内容是："{last_reply}"。'
                                     f'注意：不要重复这个话题，除非对方追问。')
            if chat_type == 'group':
                context_parts.append(f'[消息格式] 每条消息标注了发送者，如 [张三]: 你好')
            else:
                context_parts.append(f'[消息格式] 这是你和 {friend_name or "对方"} 的私聊')
            context_parts.append(clean_text)

            result = generate_reply(
                conversation_text='\n'.join(context_parts),
                personality_text=personality,
                api_key=config['llm']['api_key'],
                model=config['llm']['model'],
                max_tokens=config['llm']['max_tokens'],
                temperature=config['llm']['temperature'],
                provider=config['llm'].get('provider', 'deepseek'),
                chat_type=chat_type,
                friend_name=friend_name or "",
                friend_info=friend_info,
            )
        except Exception as e:
            last_text = current_text
            if next_tick(f"AI 调用失败: {e}"):
                break
            continue

        if not result.get('should_reply'):
            last_text = current_text
            if dry_run:
                print(f"  [{iteration}] OCR内容预览: {current_text[:150]}...")
            print(f"  [{iteration}] AI 判断无需回复 (OCR: {current_text[:80]}...)")
            if next_tick():
                break
            continue

        reply_text = result.get('message', '').strip()
        if not reply_text:
            last_text = current_text
            if next_tick("AI 回复为空"):
                break
            continue

        delay = result.get('delay_seconds', 3)
        print(f"  [{iteration}] 准备回复: \"{reply_text}\" (思考 {delay}s)")

        # Send
        if dry_run:
            print(f"  [{iteration}] [DRY-RUN] 未实际发送")
            rate_limiter.record()
            sent_messages.append(reply_text)
            last_replied_hash = latest_hash
            baseline_text = current_text
        else:
            time.sleep(delay)

            try:
                input_x, input_y = get_input_position(hwnd, config['wechat']['chat_region'])
                typing_cfg = config['typing']
                send_message_direct(hwnd, reply_text, input_x, input_y,
                                    wpm=typing_cfg['wpm'],
                                    typo_rate=typing_cfg['typo_rate'],
                                    pause_rate=typing_cfg['pause_rate'])

                print(f"  [{iteration}] 已发送!")
                rate_limiter.record()
                sent_messages.append(reply_text)
                last_replied_hash = latest_hash
                baseline_text = current_text

            except Exception as e:
                print(f"  [!] 发送失败: {e}")

        last_text = current_text

        if next_tick():
            break


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='WeChat Auto-Reply Agent')
    parser.add_argument('--once', action='store_true', help='Check once then exit')
    parser.add_argument('--dry-run', action='store_true', help='No typing, just print what would happen')
    parser.add_argument('--calibrate', action='store_true', help='Take a screenshot to verify chat region')
    parser.add_argument('--config', default='config.json', help='Config file path')
    args = parser.parse_args()

    base = os.path.dirname(os.path.abspath(__file__))
    config = load_config(args.config)
    personality = load_personality(config['personality_file'])
    # Optional knowledge base
    knowledge_path = os.path.join(base, 'knowledge.txt')
    personality += load_knowledge(knowledge_path)
    # Friends personality directory
    friends_dir = config.get('friends_dir', 'personality/friends')
    if not os.path.isabs(friends_dir):
        friends_dir = os.path.join(base, friends_dir)

    if args.calibrate:
        calibrate(config)
        return

    # Validate setup
    if not config['llm']['api_key']:
        print("[!] 警告: 未配置 LLM API Key")
        print("    在 config.json 中设置 llm.api_key，或设置环境变量 DEEPSEEK_API_KEY")
        if not args.dry_run:
            print("    当前将以 dry-run 模式运行")
            args.dry_run = True

    print("=" * 50)
    print("  WeChat Auto-Reply Agent")
    print("  按 Ctrl+C 停止")
    print("=" * 50)

    try:
        run_loop(config, personality, friends_dir, once=args.once, dry_run=args.dry_run)
    except KeyboardInterrupt:
        print("\n[*] 已停止")


if __name__ == '__main__':
    main()
