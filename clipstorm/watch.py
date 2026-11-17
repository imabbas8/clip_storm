"""The watchlist: channels you follow. Every two minutes each one is checked, and a channel that has gone live
starts being recorded and clipped by itself. Kept in data/channels.json."""
import json
import threading
import time
import uuid

from . import config, jobs, live, media

_PATH = config.DATA / "channels.json"
_lock = threading.RLock()
STATUS: dict = {}            # channel id -> {"state": live|offline|recording|busy|error, "checked": time, "note": text}
EVERY = 120
MAX_LIVE = 3                 # streams recorded at once from the watchlist; links you paste yourself are not counted


def load() -> list:
    with _lock:
        try:
            return json.loads(_PATH.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return []


def _save(chs):
    with _lock:
        _PATH.write_text(json.dumps(chs, indent=1), encoding="utf-8")


def add(url, name="") -> dict:
    url = media.normalize(url)
    chs = load()
    if any(c["url"].lower() == url.lower() for c in chs):
        raise ValueError("That channel is already on the list.")
    if len(chs) >= 50:
        raise ValueError("The watchlist is full (50 channels).")
    guess = url.rstrip("/").removesuffix("/live").split("/")[-1].lstrip("@")
    ch = {"id": uuid.uuid4().hex[:8], "url": url, "name": (name or guess).strip()[:60], "auto": True}
    _save(chs + [ch])
    threading.Thread(target=check, args=(ch,), daemon=True).start()
    return ch


def update(cid, auto):
    chs = load()
    for c in chs:
        if c["id"] == cid:
            c["auto"] = bool(auto)
    _save(chs)


def remove(cid):
    _save([c for c in load() if c["id"] != cid])
    STATUS.pop(cid, None)


def _recording(url):
    return next((s for s in live.SESSIONS.values() if s.job["source"].get("url", "").lower() == url.lower()
                 and not s.stop_flag.is_set()), None)


def check(ch):
    """Look at one channel now. Starts a recording if it is live and set to record by itself."""
    def note(state, text=""):
        STATUS[ch["id"]] = {"state": state, "checked": time.time(), "note": text}

    if _recording(ch["url"]):
        return note("recording", "Being recorded")
    info = media.inspect(ch["url"])
    if not info["ok"]:
        return note("offline" if info.get("waiting") else "error", "" if info.get("waiting") else info["error"])
    if not info["live"]:
        return note("offline")
    if not ch.get("auto"):
        return note("live", info.get("title", ""))
    if len(live.SESSIONS) >= MAX_LIVE:
        return note("busy", f"Live, but {MAX_LIVE} streams are already being recorded")
    job = jobs.create("live", {"type": "url", "url": ch["url"]}, info.get("title") or ch["name"])
    job["uploader"] = info.get("uploader") or ch["name"]
    jobs.set_stage(job, "Connecting to the stream", status="live")
    jobs.log(job, f"{ch['name']} went live: recording started by itself")
    live.start(job)
    note("recording", info.get("title", ""))


def _loop():
    time.sleep(15)
    while True:
        t0 = time.time()
        for ch in load():
            try:
                check(ch)
            except Exception as e:  # noqa: BLE001
                STATUS[ch["id"]] = {"state": "error", "checked": time.time(), "note": str(e)[:150]}
        time.sleep(max(20, EVERY - (time.time() - t0)))


def start():
    threading.Thread(target=_loop, daemon=True).start()


def summary() -> list:
    return [dict(c, **STATUS.get(c["id"], {"state": "unknown", "checked": 0, "note": ""})) for c in load()]
