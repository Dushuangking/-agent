"""Detect whether the current WeChat window is a group chat or private chat."""

import re


def _looks_like_person_name(name):
    """Heuristic: does this string look like a person's name vs a group name?

    Person names are typically short (1-4 chars Chinese, 2-15 chars Latin).
    Group names tend to be longer, contain punctuation, emoji, or keywords.
    """
    if not name:
        return False
    # Group-name indicators: long, contains emoji, special chars, group keywords
    if any(kw in name for kw in ['群', '讨论', '交流', '通知', '工作', '项目',
                                   '班', '组', '队', '室', '部', '会',
                                   '朋友', '家人', '兄弟', '姐妹']):
        return False
    # Person names are short and clean
    if re.match(r'^[一-鿿]{1,4}$', name):  # Chinese name
        return True
    if re.match(r'^[A-Za-z][A-Za-z\s\.]{1,14}$', name):  # Latin name
        return True
    return False


def detect_chat_type(ocr_text, window_title=""):
    """Analyze OCR output and window title to determine chat type.

    Detection signals (in priority order):
    1. Window title: "好友名 - 微信" → private, "群名 - 微信" → group
    2. OCR sender count: 2+ unique [Name]: patterns → group
    3. Default: assume private chat

    Returns:
        dict: {
            'type': 'group' | 'private',
            'friend_name': str or None,
        }
    """
    # --- Signal 1: Window title ---
    if window_title and window_title != "微信":
        # Try "XXX - 微信" or "XXX – 微信" format
        m = re.match(r'^(.+?)\s*[-–—]\s*微信', window_title)
        if m:
            extracted = m.group(1).strip()
            if _looks_like_person_name(extracted):
                return {'type': 'private', 'friend_name': extracted}
            else:
                return {'type': 'group', 'friend_name': None}
        # Title is something else (not "微信" and not "X - 微信")
        # Could be a popped-out window with just the name
        if _looks_like_person_name(window_title):
            return {'type': 'private', 'friend_name': window_title}
        if len(window_title) > 1:
            return {'type': 'group', 'friend_name': None}

    # --- Signal 2: OCR sender name count ---
    if ocr_text:
        senders = set()
        for match in re.finditer(r'\[([^\]]+)\]:', ocr_text):
            name = match.group(1).strip()
            if name:
                senders.add(name)

        if len(senders) >= 2:
            return {'type': 'group', 'friend_name': None}
        if len(senders) == 1:
            return {'type': 'private', 'friend_name': list(senders)[0]}

    # --- Signal 3: Default to private ---
    return {'type': 'private', 'friend_name': None}
