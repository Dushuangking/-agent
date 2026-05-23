"""Window finding and screen capture for WeChat."""
import ctypes
from ctypes import wintypes
import win32gui
import win32con
import win32process
import mss
from PIL import Image

# Windows API for getting process name
_kernel32 = ctypes.windll.kernel32
_psapi = ctypes.windll.psapi


def _get_process_name(hwnd):
    """Get the executable name for a window's process."""
    _, pid = win32process.GetWindowThreadProcessId(hwnd)
    if not pid:
        return ""

    process_handle = _kernel32.OpenProcess(0x0400 | 0x0010, False, pid)
    if not process_handle:
        return ""

    try:
        name_buffer = ctypes.create_unicode_buffer(260)
        size = wintypes.DWORD(260)
        if _psapi.GetModuleBaseNameW(process_handle, None, name_buffer, size):
            return name_buffer.value.lower()
    finally:
        _kernel32.CloseHandle(process_handle)

    return ""


def find_wechat_window(title="微信"):
    """Find the WeChat main window handle."""
    def callback(hwnd, windows):
        if not win32gui.IsWindowVisible(hwnd):
            return True
        window_text = win32gui.GetWindowText(hwnd)
        if title not in window_text:
            return True
        rect = win32gui.GetWindowRect(hwnd)
        w = rect[2] - rect[0]
        h = rect[3] - rect[1]
        if w < 400 or h < 300:
            return True
        # Verify it's actually WeChat process
        proc = _get_process_name(hwnd)
        if 'wechat' not in proc and 'weixin' not in proc:
            return True
        windows.append((rect[0], rect[1], rect[2], rect[3], hwnd, window_text))
        return True

    windows = []
    win32gui.EnumWindows(callback, windows)
    if not windows:
        return None
    # Pick the largest window (main window, not chat popup)
    windows.sort(key=lambda x: (x[2] - x[0]) * (x[3] - x[1]), reverse=True)
    return (windows[0][4], windows[0][5])  # (hwnd, title)


def get_window_rect(hwnd):
    """Get window rectangle as (left, top, right, bottom)."""
    return win32gui.GetWindowRect(hwnd)


def focus_window(hwnd):
    """Bring the WeChat window to foreground (best-effort)."""
    win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
    win32gui.SetForegroundWindow(hwnd)


def capture_region(region):
    """
    Capture a screen region.
    region: {'left': x, 'top': y, 'width': w, 'height': h}
    Returns PIL Image.
    """
    with mss.mss() as sct:
        screenshot = sct.grab(region)
        return Image.frombytes('RGB', screenshot.size, screenshot.bgra, 'raw', 'BGRX')


def is_window_foreground(hwnd):
    """Check if the given window is currently in the foreground."""
    return win32gui.GetForegroundWindow() == hwnd
