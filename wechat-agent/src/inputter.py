"""Human-like keyboard input simulation."""
import time
import random
import pyautogui
import pyperclip

pyautogui.FAILSAFE = True


def click_at(x, y):
    """Click at screen coordinates with natural movement."""
    pyautogui.moveTo(x, y, duration=random.uniform(0.15, 0.4),
                     tween=pyautogui.easeInOutQuad)
    time.sleep(random.uniform(0.03, 0.12))
    pyautogui.click()


def type_like_human(text, wpm=180, typo_rate=0.02, pause_rate=0.08):
    """
    Type text with human-like rhythm, occasional typos, and thinking pauses.
    For ASCII text only — Chinese is handled via clipboard paste.
    """
    chars_per_second = (wpm * 5) / 60
    base_delay = 1.0 / chars_per_second

    for char in text:
        delay = random.gauss(base_delay, base_delay * 0.35)
        delay = max(0.02, min(delay, 0.35))

        if random.random() < pause_rate:
            delay += random.uniform(0.4, 2.0)

        if random.random() < typo_rate:
            nearby = _nearby_key(char)
            if nearby:
                pyautogui.write(nearby, interval=random.uniform(0.02, 0.08))
                time.sleep(random.uniform(0.1, 0.3))
                pyautogui.press('backspace')
                time.sleep(random.uniform(0.04, 0.12))

        pyautogui.write(char, interval=0)
        time.sleep(delay)


def _contains_chinese(text):
    """Check if text contains any CJK characters."""
    for ch in text:
        if '一' <= ch <= '鿿' or '㐀' <= ch <= '䶿':
            return True
    return False


def send_message(text, wpm=180, typo_rate=0.02, pause_rate=0.08):
    """Send a message via pyautogui — uses clipboard paste for Chinese, types ASCII directly."""
    if _contains_chinese(text):
        pyperclip.copy(text)
        time.sleep(random.uniform(0.05, 0.15))
        pyautogui.hotkey('ctrl', 'v')
    else:
        type_like_human(text, wpm=wpm, typo_rate=typo_rate, pause_rate=pause_rate)

    time.sleep(random.uniform(0.15, 0.5))
    pyautogui.press('enter')


def send_message_direct(hwnd, text, input_x, input_y,
                        wpm=180, typo_rate=0.02, pause_rate=0.08):
    """Send via PostMessage click + pyautogui paste.

    Click is sent via PostMessage (no mouse move, no focus steal).
    Ctrl+V and Enter use pyautogui because modifier keys need real
    keyboard state — a standalone hotkey is indistinguishable from
    a human pasting.
    """
    import win32gui
    import win32con

    pyperclip.copy(text)
    time.sleep(random.uniform(0.06, 0.15))

    # Click input area via PostMessage — doesn't move mouse
    client_pos = win32gui.ScreenToClient(hwnd, (input_x, input_y))
    lparam = (client_pos[1] << 16) | client_pos[0]
    win32gui.PostMessage(hwnd, win32con.WM_LBUTTONDOWN, win32con.MK_LBUTTON, lparam)
    time.sleep(random.uniform(0.03, 0.08))
    win32gui.PostMessage(hwnd, win32con.WM_LBUTTONUP, 0, lparam)
    time.sleep(random.uniform(0.06, 0.12))

    # Paste + Enter via pyautogui — modifier combos need real keyboard state
    pyautogui.hotkey('ctrl', 'v')
    time.sleep(random.uniform(0.1, 0.3))
    pyautogui.press('enter')


_KEYBOARD_NEIGHBORS = {
    'a': 'q', 'b': 'v', 'c': 'x', 'd': 's', 'e': 'w', 'f': 'd',
    'g': 'f', 'h': 'g', 'i': 'u', 'j': 'h', 'k': 'j', 'l': 'k',
    'm': 'n', 'n': 'm', 'o': 'i', 'p': 'o', 'q': 'w', 'r': 'e',
    's': 'a', 't': 'r', 'u': 'y', 'v': 'c', 'w': 'q', 'x': 'z',
    'y': 't', 'z': 'x',
}


def _nearby_key(char):
    """Return a keyboard-adjacent key for simulating typos, or None."""
    return _KEYBOARD_NEIGHBORS.get(char.lower())
