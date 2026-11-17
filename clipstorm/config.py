"""Where things are kept, and the settings saved from the dashboard (data/settings.json)."""
import json
import os
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("CLIPSTORM_DATA") or ROOT / "data").resolve()
JOBS = DATA / "jobs"
MODELS = DATA / "models"
FONTS = Path(__file__).resolve().parent / "fonts"
WEB = ROOT / "web"
HOST = os.environ.get("CLIPSTORM_HOST", "127.0.0.1")
PORT = int(os.environ.get("CLIPSTORM_PORT", "8787"))
PASSWORD = os.environ.get("CLIPSTORM_PASSWORD", "")

for d in (DATA, JOBS, MODELS):
    d.mkdir(parents=True, exist_ok=True)

DEFAULTS = {
    # the AI that picks the moments and writes hooks and titles
    "ai_provider": "anthropic",
    "ai_key": "",
    "ai_model": "",
    "ai_base_url": "",
    # speech to text: local (free, this machine) | groq | openai
    "asr_provider": "local",
    "asr_key": "",
    "asr_local_model": "base",
    "asr_language": "",          # "" = detect; or "en", "hi", "ur" ...
    # clips
    "min_len": 20,
    "max_len": 55,
    "max_clips": 6,              # per finished video
    "min_score": 70,             # live: only clip moments the AI scores at least this
    "layout": "smart",           # smart (follow the face) | zoom | blur | crop
    "watermark": "",             # your channel name, drawn small on every short
    "edits": True,               # jump cuts, punch-in zooms and the progress bar
    "style": "bounce",           # bounce | hype | clean
    "captions": True,
    "caption_script": "roman",   # roman: Hindi/Urdu etc. written in English letters | original
    "quality": 1080,             # 1080 or 720 (faster)
    # live
    "live_check_min": 3,         # ask the AI at least this often while people are talking
    "live_per_hour": 8,          # most clips per hour from one stream
    "buffer_min": 10,            # minutes of stream kept to cut from
    "agents": 3,                 # AI agents that judge each stretch side by side (Hype, Story, Hook, Funny)
    "force_min": 0,              # live: no short for this many minutes, take the best moment anyway (0 = never)
    "wait_for_live": True,       # link not live yet: keep checking instead of giving up
    # downloads
    "max_height": 1080,
    "cookies_file": "",          # a cookies.txt, if YouTube asks to sign in
    "cookies_browser": "",       # or: read the sign-in from a browser on this machine (chrome, firefox, edge, brave)
    # YouTube posting (optional)
    "yt_client_id": "",
    "yt_client_secret": "",
    "yt_auto_upload": False,
    "yt_privacy": "private",     # private | unlisted | public
    "yt_tokens": {},
    "yt_channel": "",
}
SECRET_KEYS = ("ai_key", "asr_key", "yt_client_secret")
_PATH = DATA / "settings.json"
_lock = threading.Lock()
_cache = None


def get() -> dict:
    global _cache
    with _lock:
        if _cache is None:
            saved = {}
            if _PATH.exists():
                try:
                    saved = json.loads(_PATH.read_text(encoding="utf-8"))
                except Exception:  # noqa: BLE001 - a broken file should not stop the app from opening
                    saved = {}
            _cache = {**DEFAULTS, **{k: v for k, v in saved.items() if k in DEFAULTS}}
            # keys may also come from the environment, so nothing has to be typed on a server
            env = {"ai_key": "CLIPSTORM_AI_KEY", "ai_provider": "CLIPSTORM_AI_PROVIDER", "ai_model": "CLIPSTORM_AI_MODEL"}
            for k, name in env.items():
                if not _cache[k] and os.environ.get(name):
                    _cache[k] = os.environ[name]
        return dict(_cache)


def update(changes: dict) -> dict:
    global _cache
    cur = get()
    for k, v in changes.items():
        if k not in DEFAULTS:
            continue
        if k in SECRET_KEYS and (v is None or v == "" or set(str(v)) <= {"•"}):
            continue                                  # an untouched masked field keeps the saved secret
        want = type(DEFAULTS[k])
        try:
            if want is bool:
                v = v if isinstance(v, bool) else str(v).lower() in ("1", "true", "yes", "on")
            elif want is int:
                v = int(float(v))
            elif want is str:
                v = str(v).strip()
        except (TypeError, ValueError):
            continue
        cur[k] = v
    cur["min_len"] = max(8, min(120, cur["min_len"]))
    cur["max_len"] = max(cur["min_len"] + 5, min(180, cur["max_len"]))
    cur["max_clips"] = max(1, min(30, cur["max_clips"]))
    cur["buffer_min"] = max(4, min(60, cur["buffer_min"]))
    cur["live_check_min"] = max(1, min(30, cur["live_check_min"]))
    cur["agents"] = max(1, min(4, cur["agents"]))
    cur["force_min"] = max(0, min(240, cur["force_min"]))
    with _lock:
        _cache = cur
        tmp = _PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(cur, indent=1), encoding="utf-8")
        os.chmod(tmp, 0o600)
        tmp.replace(_PATH)
    return dict(cur)


def clear_secret(key: str):
    global _cache
    cur = get()
    cur[key] = DEFAULTS[key]
    with _lock:
        _cache = cur
        _PATH.write_text(json.dumps(cur, indent=1), encoding="utf-8")


def public() -> dict:
    """Settings for the page: secrets are never sent back, only whether they are set."""
    s = get()
    out = {k: v for k, v in s.items() if k not in SECRET_KEYS and k != "yt_tokens"}
    for k in SECRET_KEYS:
        out[k] = "••••••••" if s[k] else ""
    out["yt_connected"] = bool(s["yt_tokens"].get("refresh_token"))
    return out
