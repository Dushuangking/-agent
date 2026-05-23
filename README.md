An intelligent WeChat auto-reply agent that uses screenshot + OCR to read messages and LLM APIs to generate human-like replies. Designed for low-risk, occasional use in friend group chats.

## Features

- **Screenshot + OCR**: Captures the WeChat chat area and extracts text via RapidOCR
- **AI-Powered Replies**: Uses DeepSeek or Claude API to generate natural, context-aware replies
- **Group & Private Chat Detection**: Automatically switches between group and private chat modes, with per-friend personality support
- **Web UI**: Browser-based control panel with start/stop, settings, visual calibration, and real-time logs
- **Human-like Typing**: Simulated keystrokes with configurable WPM, typos, and pauses
- **Safety Controls**: Rate limiting, time-of-day restrictions, burst protection, cooldown timers

## Requirements

- Python 3.10+
- Windows (uses win32gui, pywinauto for WeChat window interaction)

Install dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Configuration

Edit `config.json`:

```json
{
    "llm": {
        "provider": "deepseek",
        "api_key": "YOUR_DEEPSEEK_API_KEY_HERE",
        "model": "deepseek-chat"
    }
}
```

Set your DeepSeek API key (get one at https://platform.deepseek.com) or use an environment variable:

```bash
set DEEPSEEK_API_KEY=your-key-here
```

### 2. Calibrate Chat Region

The agent needs to know where the chat messages appear in the WeChat window. Use the Web UI for visual calibration:

```bash
python web/server.py
```

Open `http://127.0.0.1:5050` in your browser, click "可视校准" in the Chat Region card, and drag to select the message area.

### 3. Customize Personality

Edit `personality/default.txt` to describe your persona — nickname, interests, speaking style, relationship with friends, etc.

Optionally add friend-specific personalities in `personality/friends/<friend_name>.txt`. The agent will automatically load the matching file during private chats.

### 4. Add Knowledge (Optional)

Edit `knowledge.txt` to add domain knowledge the AI can draw from (football, gaming, memes, etc.).

### 5. Run

**Web UI (recommended):**

```bash
python web/server.py
# Open http://127.0.0.1:5050
```

**Command line:**

```bash
python agent.py              # Continuous monitoring
python agent.py --once       # Check once then exit
python agent.py --dry-run    # Preview mode (no actual typing)
python agent.py --calibrate  # Screenshot to verify chat region
```

## Configuration Reference

| Section | Key | Description |
|---------|-----|-------------|
| `wechat.window_title` | `"微信"` | WeChat window title to detect |
| `wechat.chat_region` | `mode: "percent"` | Chat area as % of window (survives resize) |
| `ocr.lang` | `"ch"` | OCR language |
| `llm.provider` | `"deepseek"` or `"claude"` | LLM provider |
| `llm.model` | `"deepseek-chat"` | Model name |
| `llm.temperature` | `0.85` | Reply creativity (0-1) |
| `safety.max_replies_per_hour` | `120` | Hourly reply cap |
| `safety.allowed_start_hour` | `7` | Earliest hour to reply |
| `safety.allowed_end_hour` | `26` | Latest hour (26 = 2am next day) |
| `typing.wpm` | `320` | Typing speed (words per minute) |
| `typing.typo_rate` | `0.01` | Random typo probability |

## Project Structure

```
wechat-agent/
  agent.py              # CLI entry point
  config.json           # Configuration
  knowledge.txt         # Domain knowledge for AI
  requirements.txt      # Python dependencies
  personality/
    default.txt         # Base personality description
    friends/
      _template.txt     # Friend personality template
  src/
    capture.py          # Window detection & screenshot
    ocr.py              # OCR text extraction
    responder.py        # AI reply generation (DeepSeek/Claude)
    inputter.py         # Simulated keyboard input
    safety.py           # Rate limiter & time restrictions
    chat_type.py        # Group vs private chat detection
  web/
    server.py           # Flask backend
    templates/
      index.html        # Web UI frontend
```

## How It Works

1. **Capture**: Screenshots the WeChat chat region at configurable intervals
2. **OCR**: Extracts text using RapidOCR, with sender name annotation via spatial analysis (font size differences)
3. **Detect**: Identifies new messages, filters out own replies, detects group vs private chat
4. **Decide**: Sends recent messages + personality + knowledge to the LLM; the LLM decides whether to reply and what to say
5. **Send**: Simulates human typing through the WeChat input box with realistic timing

## Safety Features

- Rate limiting with burst protection
- Configurable active hours (no replies at 3am)
- Random delays between replies
- Cooldown after each reply
- Own-message filter prevents self-reply loops

## Disclaimer

This tool is for educational and entertainment purposes. Use responsibly and in accordance with WeChat's terms of service. The authors are not responsible for any account restrictions or other consequences resulting from use of this software.
