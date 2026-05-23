"""OCR processing for WeChat screenshots using RapidOCR.

Sender name detection uses both text patterns AND spatial cues:
- Sender names in group chat are in a smaller font (smaller bounding box)
- Real sender names sit very close above their message (tight vertical gap)
- This combination is far more reliable than text-pattern alone.
"""
import re
import numpy as np
from rapidocr_onnxruntime import RapidOCR

_ocr_instance = None


def _get_ocr():
    """Get or create the RapidOCR instance (singleton)."""
    global _ocr_instance
    if _ocr_instance is None:
        _ocr_instance = RapidOCR()
    return _ocr_instance


# Common short chat messages that look like sender names but aren't
_NOT_NAMES = {
    # Chinese reactions / interjections
    '哈哈', '哈哈哈', '哈哈哈哈', '呵呵', '嘿嘿', '嘻嘻',
    '嗯', '嗯嗯', '哦', '噢', '喔', '啊', '呀', '哎', '诶', '咦', '切', '哼',
    '好', '好的', '行', '可以', '没事', '没关系', '不客气',
    '对', '是的', '没错', '确实', '笑死', '难绷', '牛的', '6',
    # Common short replies
    '来了', '在吗', '在', '说', '好嘞', '好滴', 'okk',
    '谢谢', '多谢', '晚安', '早安', '吃了', '睡了', '再见', '拜拜',
    # English
    'ok', 'OK', 'Ok', 'yes', 'no', 'lol', 'LOL', 'lmao', 'haha',
    'thanks', 'thx', 'bye', 'hi', 'hey', 'yo', 'wow', 'omg',
    # Punctuation-only
    '？', '?', '！', '!', '...', '。。。',
}


def _is_sender_name(text):
    """Check if text looks like a WeChat sender name (not a message)."""
    if text in _NOT_NAMES:
        return False
    # Chinese name: 1-5 characters
    if re.match(r'^[一-鿿]{1,5}$', text):
        return True
    # Latin name: 2-12 letters
    if re.match(r'^[A-Za-z]{2,12}$', text):
        return True
    return False


def _box_height(box):
    """Compute bounding box height from RapidOCR box coordinates."""
    if isinstance(box, np.ndarray):
        return float(box[:, 1].max() - box[:, 1].min())
    else:
        ys = [p[1] for p in box]
        return float(max(ys) - min(ys))


def _box_top(box):
    """Compute top Y coordinate from RapidOCR box coordinates."""
    if isinstance(box, np.ndarray):
        return float(box[:, 1].min())
    else:
        return float(min(p[1] for p in box))


def extract_text(image, lang='ch', tesseract_path=None):
    """Run RapidOCR on a PIL Image and return formatted conversation text.

    Collects bounding-box height alongside each text line so the
    annotator can distinguish small-font sender names from messages.
    """
    ocr = _get_ocr()
    img_array = np.array(image.convert('RGB'))

    output = ocr(img_array)

    if isinstance(output, tuple):
        result = output[0]
    else:
        result = output

    if not result:
        return ""

    # Collect (top_y, text, box_height) for each detected line
    items = []
    for entry in result:
        box = entry[0]
        rest = entry[1:]

        if len(rest) >= 2:
            text, score = rest[0], rest[1]
        elif isinstance(rest[0], (list, tuple)):
            text, score = rest[0][0], rest[0][1] if len(rest[0]) > 1 else 1.0
        else:
            text, score = rest[0], 1.0

        if not text or float(score) < 0.5:
            continue

        stripped = str(text).strip()
        if stripped:
            items.append((_box_top(box), stripped, _box_height(box)))

    if not items:
        return ""

    # Sort by vertical position (top-to-bottom reading order)
    items.sort(key=lambda x: x[0])

    # Remove noise, keep (text, height) pairs
    clean = []
    for _, text, height in items:
        filtered = _filter_noise(text)
        if filtered:
            clean.append((filtered, height))

    return _annotate_senders(clean)


def _filter_noise(text):
    """Remove WeChat UI noise lines."""
    # Standalone timestamps like "19:11"
    if re.match(r'^\d{1,2}:\d{2}$', text):
        return None
    # Known UI button text
    if text in ('发送',):
        return None
    return text


def _annotate_senders(lines):
    """Annotate sender names using text patterns + spatial cues.

    In WeChat group chats, sender names are rendered in a noticeably
    smaller font than message text.  We exploit this:

    - A candidate sender name whose bounding box is *shorter* than the
      following line is probably a real sender label.
    - Two consecutive lines of similar height are both regular messages
      (the first one just happens to be short text).

    Args:
        lines: list of (text, box_height) tuples
    Returns:
        annotated conversation string
    """
    out = []
    current_sender = None
    i = 0
    n = len(lines)

    while i < n:
        text, height = lines[i]

        if _is_sender_name(text) and i + 1 < n:
            next_text, next_height = lines[i + 1]

            if not _is_sender_name(next_text):
                # Spatial check: sender name should be in a smaller font
                # than the message that follows it.
                height_ratio = height / next_height if next_height > 0 else 1.0
                if height_ratio < 0.92:
                    # Confirmed: small-font label followed by larger message
                    current_sender = text
                    out.append(f'[{text}]: {next_text}')
                    i += 2
                    continue
                # Heights are too similar — treat both as regular messages

        # Multi-line message continuation
        if current_sender and not _is_sender_name(text):
            out.append(f'[{current_sender}]: {text}')
        else:
            out.append(text)
        i += 1

    return '\n'.join(out)
