"""Getting video in: yt-dlp for links, ffmpeg/ffprobe for everything on disk."""
import json
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np

from . import config

SR = 16000


class MediaError(Exception):
    pass


def tools_missing() -> list:
    return [t for t in ("ffmpeg", "ffprobe") if not shutil.which(t)]


def run(cmd, timeout=3600, cwd=None) -> str:
    p = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, errors="replace", timeout=timeout, cwd=cwd)
    if p.returncode != 0:
        raise MediaError((p.stderr or p.stdout or "ffmpeg failed").strip()[-600:])
    return p.stdout


def probe(path) -> dict:
    out = run(["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", path], 120)
    d = json.loads(out)
    v = next((s for s in d.get("streams", []) if s.get("codec_type") == "video"), None)
    a = next((s for s in d.get("streams", []) if s.get("codec_type") == "audio"), None)
    if not v:
        raise MediaError("This file has no video in it.")
    w, h = int(v["width"]), int(v["height"])
    rot = str((v.get("tags") or {}).get("rotate") or next(
        (sd.get("rotation") for sd in v.get("side_data_list", []) if "rotation" in sd), 0))
    if rot.lstrip("-") in ("90", "270"):
        w, h = h, w
    return {"width": w, "height": h, "duration": float(d.get("format", {}).get("duration") or 0), "has_audio": bool(a)}


def pcm(path, start=None, dur=None) -> np.ndarray:
    """The audio as 16 kHz mono int16. An unreadable or silent source gives an empty array."""
    cmd = ["ffmpeg", "-v", "error"]
    if start is not None:
        cmd += ["-ss", f"{start:.3f}"]
    if dur is not None:
        cmd += ["-t", f"{dur:.3f}"]
    cmd += ["-i", str(path), "-vn", "-ac", "1", "-ar", str(SR), "-f", "s16le", "pipe:1"]
    p = subprocess.run(cmd, capture_output=True, timeout=3600)
    data = p.stdout[: len(p.stdout) // 2 * 2]
    return np.frombuffer(data, dtype=np.int16)


def loudness(audio: np.ndarray) -> list:
    """One loudness figure (dB) per second."""
    n = len(audio) // SR
    if n == 0:
        return []
    x = audio[: n * SR].astype(np.float32).reshape(n, SR) / 32768.0
    return [round(float(v), 1) for v in 20 * np.log10(np.sqrt((x ** 2).mean(axis=1)) + 1e-5)]


# ---------------------------------------------------------------- links
class _Quiet:
    def debug(self, m): pass
    def info(self, m): pass
    def warning(self, m): pass
    def error(self, m): pass


BROWSERS = ("chrome", "chromium", "brave", "edge", "firefox")
_signin = {"t": 0.0, "found": {}}


def browser_signin(fresh=False) -> dict:
    """Which browsers on this machine are signed in to YouTube: {browser: True/False}. Only cookie NAMES are looked at;
    nothing is read out or stored. Cached for a minute."""
    import time
    if not fresh and time.time() - _signin["t"] < 60:
        return _signin["found"]
    from yt_dlp.cookies import extract_cookies_from_browser
    found = {}
    for b in BROWSERS:
        try:
            jar = extract_cookies_from_browser(b, logger=_Quiet())
        except Exception:  # noqa: BLE001 - not installed, or its cookies cannot be read
            continue
        names = {c.name for c in jar if c.domain.endswith("youtube.com")}
        found[b] = bool(names & {"LOGIN_INFO", "SAPISID", "__Secure-3PAPISID"})
    _signin.update(t=time.time(), found=found)
    return found


def _ydl_opts(extra=None, url="") -> dict:
    s = config.get()
    o = {"quiet": True, "no_warnings": True, "noplaylist": True, "logger": _Quiet(), "socket_timeout": 30}
    ck = s["cookies_file"].strip()
    if ck and Path(ck).is_file():
        o["cookiefile"] = ck
    elif s["cookies_browser"] in BROWSERS:
        o["cookiesfrombrowser"] = (s["cookies_browser"],)
    elif is_youtube(url):                       # nothing chosen: use any browser here that is signed in to YouTube
        signed = [b for b, ok in browser_signin().items() if ok]
        if signed:
            o["cookiesfrombrowser"] = (signed[0],)
    o.update(extra or {})
    return o


def is_youtube(url: str) -> bool:
    return bool(re.match(r"^https?://([\w-]+\.)?(youtube\.com|youtu\.be)/", url or "", re.I))


def youtube_help() -> str:
    """What to do when YouTube asks for a sign-in, based on what is really on this machine."""
    s = config.get()
    if s["cookies_file"].strip() and Path(s["cookies_file"].strip()).is_file():
        return ("YouTube refused the video even with your cookies.txt. The file may be old or from a signed-out "
                "browser: export it again while signed in to YouTube.")
    found = browser_signin(fresh=True)
    if any(found.values()):
        return ("YouTube refused this computer even with your browser's YouTube sign-in. Open youtube.com in that "
                "browser, play any video once, then try again. If it keeps happening, YouTube is blocking this "
                "network: upload the video file instead.")
    names = ", ".join(b.title() for b in found) or "no browser"
    return ("YouTube wants this computer to be signed in before it hands over videos, and no browser here is signed "
            f"in to YouTube (found: {names}). Open youtube.com in Chrome on this computer, sign in with any Google "
            "account, then try the link again. ClipStorm will use that sign-in by itself. Twitch, Kick and "
            "uploaded files work without this.")


def friendly(e, url="") -> str:
    m = re.sub(r"\x1b\[[0-9;]*m", "", str(e))
    m = re.sub(r"^ERROR:\s*(\[[^\]]+\]\s*[\w-]+:\s*)?", "", m).strip()
    low = m.lower()
    if "sign in to confirm" in low or "not a bot" in low:
        return youtube_help()
    if "private video" in low:
        return "This video is private."
    if "unsupported url" in low:
        return "That link is not a video page ClipStorm can read."
    return m[-300:] or "The link could not be opened."


def normalize(url: str) -> str:
    url = (url or "").strip()
    if not url:
        raise MediaError("Paste a link first.")
    if not re.match(r"^https?://", url, re.I):
        url = "https://" + url
    if not re.match(r"^https?://[^/\s]+\.[^/\s]+", url, re.I):
        raise MediaError("That does not look like a link.")
    # a YouTube channel page: look at its live stream
    m = re.match(r"^(https?://(?:www\.)?youtube\.com/(?:@[^/?#]+|channel/[^/?#]+|c/[^/?#]+))/?(?:[?#].*)?$", url)
    return m.group(1) + "/live" if m else url


def inspect(url: str) -> dict:
    """What is behind this link: a finished video, or a stream that is live right now."""
    import yt_dlp
    try:
        with yt_dlp.YoutubeDL(_ydl_opts({"format": "b[height<=720]/bv*[height<=720]+ba/b"}, url)) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as e:  # noqa: BLE001
        low = str(e).lower()
        waiting = any(k in low for k in ("not currently live", "is offline", "live event will begin", "premieres in",
                                         "will begin in", "no video formats", "does not have a live"))
        return {"ok": False, "waiting": waiting, "error": friendly(e)}
    if not info:
        return {"ok": False, "waiting": False, "error": "Nothing was found at that link."}
    title = info.get("title") or ""
    if "(live)" in title and info.get("description"):        # Twitch keeps the real title here
        title = str(info["description"]).splitlines()[0][:120]
    fmts = info.get("requested_formats") or [info]
    return {"ok": True, "live": bool(info.get("is_live")), "title": re.sub(r"\s+\d{4}-\d\d-\d\d \d\d:\d\d$", "", title),
            "uploader": info.get("uploader") or info.get("channel") or "", "duration": float(info.get("duration") or 0),
            "urls": [f["url"] for f in fmts if f.get("url")], "page": info.get("webpage_url") or url}


def download(url: str, dest: Path, progress=None) -> Path:
    import yt_dlp
    h = int(config.get()["max_height"])

    def hook(d):
        if progress and d.get("status") == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            if total:
                progress(d.get("downloaded_bytes", 0) / total)

    opts = _ydl_opts({"format": f"bv*[height<={h}][vcodec^=avc]+ba[ext=m4a]/bv*[height<={h}]+ba/b[height<={h}]/b",
                      "merge_output_format": "mp4", "outtmpl": str(dest / "source.%(ext)s"), "progress_hooks": [hook],
                      "retries": 5, "fragment_retries": 5, "concurrent_fragment_downloads": 4}, url)
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.extract_info(url, download=True)
    except Exception as e:  # noqa: BLE001
        raise MediaError(friendly(e)) from e
    files = sorted((f for f in dest.glob("source.*") if f.suffix not in (".part", ".ytdl")), key=lambda f: -f.stat().st_size)
    if not files:
        raise MediaError("The download finished but no video file was written.")
    return files[0]
