"""Safety controls: rate limiting and time-of-day restrictions."""
import time
import random
from datetime import datetime, time as dtime
from collections import deque


class RateLimiter:
    def __init__(self, max_per_hour=8, cooldown_min=5, cooldown_max=10,
                 burst_limit=3, burst_cooldown=30):
        self.max_per_hour = max_per_hour
        self.cooldown_min = cooldown_min
        self.cooldown_max = cooldown_max
        self.burst_limit = burst_limit
        self.burst_cooldown = burst_cooldown
        self.timestamps = deque()
        self.last_reply_at = 0
        self._current_cooldown = 0
        self._burst_count = 0

    def _pick_cooldown(self):
        """Random cooldown between min and max. After burst, use burst_cooldown."""
        if self._burst_count >= self.burst_limit:
            return self.burst_cooldown
        return random.uniform(self.cooldown_min, self.cooldown_max)

    def allowed(self):
        now = time.time()

        # Check if still in cooldown window
        if self._current_cooldown > 0:
            if now - self.last_reply_at < self._current_cooldown:
                return False

        # Check burst — if 3+ replies happened within a short window
        recent_window = now - 120  # last 2 minutes
        recent_replies = sum(1 for ts in self.timestamps if ts > recent_window)
        if recent_replies >= self.burst_limit:
            # Reset burst count after burst_cooldown since last reply
            if now - self.last_reply_at < self.burst_cooldown:
                self._burst_count = self.burst_limit  # force burst cooldown
                return False
            else:
                self._burst_count = 0

        # Hourly limit
        cutoff = now - 3600
        while self.timestamps and self.timestamps[0] < cutoff:
            self.timestamps.popleft()

        if len(self.timestamps) >= self.max_per_hour:
            return False

        return True

    def record(self):
        now = time.time()
        self.timestamps.append(now)
        self.last_reply_at = now

        # Track consecutive replies
        recent_window = now - 60
        self._burst_count = sum(1 for ts in self.timestamps if ts > recent_window)

        # Pick next random cooldown
        self._current_cooldown = self._pick_cooldown()

    def next_available_in(self):
        """Seconds until next reply is allowed, or 0 if already allowed."""
        now = time.time()
        if self._current_cooldown > 0:
            remaining = self._current_cooldown - (now - self.last_reply_at)
            if remaining > 0:
                return int(remaining)
        cutoff = now - 3600
        while self.timestamps and self.timestamps[0] < cutoff:
            self.timestamps.popleft()
        if len(self.timestamps) >= self.max_per_hour:
            if self.timestamps:
                return int(self.timestamps[0] + 3600 - now)
        return 0


def in_allowed_window(start_hour=7, end_hour=25):
    """
    Check if current time is within allowed hours.
    end_hour > 24 means past midnight, e.g. 25 = 1:00 AM.
    """
    now = datetime.now().time()
    start = dtime(start_hour % 24, 0)
    end = dtime(end_hour % 24, 0)

    if end_hour > 24:
        end = dtime(end_hour - 24, 0)
        # Include the full hour: e.g. end_hour=25 means allowed through 01:59
        end = end.replace(minute=59, second=59)
        if now >= start or now <= end:
            return True
        return False

    if start <= end:
        return start <= now <= end
    else:
        # start > end means cross-midnight (e.g. 22:00-02:00)
        end = end.replace(minute=59, second=59)
        return now >= start or now <= end


def human_delay(min_s=3, max_s=20):
    """Random delay with a gamma-ish distribution: usually shorter but occasionally longer."""
    delay = random.uniform(min_s, max_s)
    if random.random() < 0.3:
        delay += random.uniform(2, 8)
    return delay
