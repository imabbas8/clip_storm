"""Jobs: one per link or uploaded file. Each lives in its own folder (data/jobs/<id>) with a job.json, so a
restart loses nothing. Finished videos run one at a time; every clip is rendered by one shared worker."""
import json
import queue
import shutil
import threading
import time
import uuid
from pathlib import Path

import numpy as np

from . import config, media, asr, scout, render

JOBS: dict = {}
_lock = threading.RLock()
_video_q: "queue.Queue" = queue.Queue()
_render_q: "queue.Queue" = queue.Queue()
_started = False


def job_dir(jid) -> Path:
    return config.JOBS / jid


def save(job):
    with _lock:
        d = job_dir(job["id"])
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / "job.tmp"
        tmp.write_text(json.dumps(job), encoding="utf-8")
        tmp.replace(d / "job.json")


def create(kind, source, title="") -> dict:
    s = config.get()
    job = {"id": time.strftime("%m%d") + uuid.uuid4().hex[:8], "kind": kind, "source": source, "title": title,
           "uploader": "", "status": "queued", "stage": "Waiting to start", "progress": 0.0, "error": "", "note": "",
           "created": time.time(), "duration": 0.0, "clips": [], "tokens": 0, "live": None, "log": [],
           "options": {k: s[k] for k in ("min_len", "max_len", "max_clips", "min_score", "layout", "style", "captions",
                                         "caption_script", "quality", "edits", "live_check_min", "live_per_hour",
                                         "buffer_min", "agents", "force_min")}}
    with _lock:
        JOBS[job["id"]] = job
    save(job)
    return job


def log(job, text):
    job["log"].append({"t": round(time.time()), "m": str(text)[:300]})
    del job["log"][:-40]


def set_stage(job, stage, progress=None, status=None):
    job["stage"] = stage
    if progress is not None:
        job["progress"] = round(float(progress), 3)
    if status:
        job["status"] = status
    save(job)


def stopped(job) -> bool:
    return bool(job.get("stop"))


def delete(jid):
    with _lock:
        job = JOBS.pop(jid, None)
    if job:
        job["stop"] = True
    shutil.rmtree(job_dir(jid), ignore_errors=True)


# ---------------------------------------------------------------- rendering (one worker for everything)
def new_clip(job, c) -> dict:
    o = job["options"]
    n = job["made"] = job.get("made", len(job["clips"])) + 1          # never reused, even after a clip is deleted
    c.update(id=f"clip{n:03d}", status="queued", error="", made_at=time.time(), layout=o["layout"], style=o["style"],
             captions=o["captions"], quality=o["quality"], edits=o["edits"], youtube=None)
    with _lock:
        job["clips"].append(c)
    save(job)
    return c


def submit_render(fn):
    _render_q.put(fn)


def render_one(job, clip, src, words, src_offset=0.0):
    """Make one clip. Never raises: a failed clip is marked and the next one still gets made."""
    if stopped(job) and job["kind"] != "live":
        return
    clip["status"] = "rendering"
    save(job)
    try:
        if "width" not in job:
            meta = media.probe(src)
            job.update(width=meta["width"], height=meta["height"], has_audio=meta["has_audio"])
        inside = [w for w in words if clip["start"] - 0.5 <= w["s"] <= clip["end"] + 0.5]
        if clip.get("captions", True) and job["options"].get("caption_script") == "roman":
            inside = scout.romanize(inside, job.get("title", ""))
        out = render.render_clip(Path(src), job_dir(job["id"]) / "clips", clip["id"], clip, inside, job["width"],
                                 job["height"], src_offset=src_offset, has_audio=job.get("has_audio", True))
        clip.update(out, status="done", text=" ".join(w["w"] for w in inside)[:1500])
        log(job, f"Clip ready: {clip.get('title') or clip['id']} ({out['duration']} s, {out['cuts']} cuts)")
    except Exception as e:  # noqa: BLE001
        clip.update(status="error", error=str(e)[-300:])
        log(job, f"Clip failed: {str(e)[-200:]}")
    save(job)
    if clip["status"] == "done" and config.get()["yt_auto_upload"]:
        from . import youtube
        youtube.upload_clip(job, clip)


def reedit(job, clip, changes: dict):
    """Change a short and make it again: its words on screen, where it starts and ends, how it looks."""
    d = job_dir(job["id"])
    if clip.get("src"):                                   # a live clip: its own saved piece of the stream
        src, lo, hi = d / "clips" / clip["src"], clip["src_offset"] + 0.1, clip["src_end"] - 0.1
        wfile, offset = d / "clips" / f"{clip['id']}.words.json", clip["src_offset"]
    else:
        src, lo, hi = Path(job.get("file") or ""), 0.0, float(job.get("duration") or 0)
        wfile, offset = d / "words.json", 0.0
    if not src.is_file() or not wfile.is_file():
        raise ValueError("The original video of this short is no longer on disk, so it cannot be edited again.")
    for k, limit in (("hook", 60), ("title", 100), ("description", 300)):
        if k in changes:
            clip[k] = " ".join(str(changes[k]).split())[:limit]
    if "hook" in changes:
        clip["hook"] = clip["hook"].upper()
    for k in ("captions", "edits"):
        if k in changes:
            clip[k] = bool(changes[k])
    if changes.get("style") in render.STYLES:
        clip["style"] = changes["style"]
    if changes.get("layout") in render.LAYOUTS:
        clip["layout"] = changes["layout"]
    try:
        s = max(lo, float(changes.get("start", clip["start"])))
        e = min(hi, float(changes.get("end", clip["end"])))
    except (TypeError, ValueError):
        raise ValueError("Start and end must be numbers (seconds).") from None
    if e - s < 3:
        raise ValueError("A short must be at least 3 seconds long.")
    if e - s > 180:
        raise ValueError("A short can be at most 3 minutes long.")
    clip.update(start=round(s, 2), end=round(e, 2), status="queued", error="", youtube=None, edited=True)
    save(job)
    words = json.loads(wfile.read_text(encoding="utf-8"))
    submit_render(lambda: render_one(job, clip, src, words, src_offset=offset))


def delete_clip(job, cid):
    d = job_dir(job["id"]) / "clips"
    with _lock:
        job["clips"] = [c for c in job["clips"] if c["id"] != cid]
    for f in d.glob(f"{cid}.*"):
        f.unlink(missing_ok=True)
    save(job)


def _render_worker():
    while True:
        fn = _render_q.get()
        try:
            fn()
        except Exception:  # noqa: BLE001
            pass


# ---------------------------------------------------------------- a finished video, start to end
def _process(job):
    d = job_dir(job["id"])
    o = job["options"]
    src = job["source"]
    if src["type"] == "url":
        set_stage(job, "Downloading the video", 0.02, "working")
        path = media.download(src["url"], d, lambda p: set_stage(job, "Downloading the video", 0.02 + 0.2 * p))
    else:
        path = Path(src["path"])
    job["file"] = str(path)
    meta = media.probe(path)
    job.update(width=meta["width"], height=meta["height"], duration=meta["duration"], has_audio=meta["has_audio"])
    if stopped(job):
        return
    set_stage(job, "Listening to the audio", 0.24, "working")
    audio = media.pcm(path) if meta["has_audio"] else np.zeros(0, dtype=np.int16)
    db = media.loudness(audio)
    set_stage(job, "Writing down what is said", 0.26)
    t0 = time.time()

    def heard(p):
        """Show how far the speech-to-text has got, and roughly how long it still needs."""
        text = f"Writing down what is said: {round(p * 100)}%"
        if 0.04 < p < 1:
            left = (time.time() - t0) / p * (1 - p)
            text += f", about {max(1, round(left / 60))} min left"
        set_stage(job, text, 0.26 + 0.34 * p)

    words, lang = asr.transcribe_long(audio, heard,
                                      lambda: stopped(job))
    del audio
    job["language"] = lang
    (d / "words.json").write_text(json.dumps(words), encoding="utf-8")
    if stopped(job):
        return
    set_stage(job, "Finding the best moments", 0.62)
    clips, info = scout.find_clips(words, db, meta["duration"], job.get("title", ""), job.get("uploader", ""),
                                   o["max_clips"], o["min_len"], o["max_len"], log=lambda m: log(job, m),
                                   progress=lambda p: set_stage(job, "Finding the best moments", 0.62 + 0.1 * p),
                                   agents=o.get("agents", 1))
    job["tokens"] += info["tokens"]
    job["note"] = info["note"]
    if not clips:
        set_stage(job, "No clips found", 1.0, "done")
        return
    made = [new_clip(job, c) for c in clips]
    for i, c in enumerate(made):
        if stopped(job):
            break
        set_stage(job, f"Editing clip {i + 1} of {len(made)}", 0.72 + 0.28 * i / len(made))
        done = threading.Event()
        submit_render(lambda c=c: (render_one(job, c, path, words), done.set()))
        done.wait()
    ok = sum(1 for c in job["clips"] if c["status"] == "done")
    set_stage(job, f"Done: {ok} short{'s' if ok != 1 else ''} ready", 1.0, "done")


def _video_worker():
    while True:
        job = _video_q.get()
        if job["id"] not in JOBS or stopped(job):
            continue
        try:
            _process(job)
            if stopped(job) and job["status"] != "done":
                set_stage(job, "Stopped", job["progress"], "stopped")
        except Exception as e:  # noqa: BLE001
            job["error"] = str(e)[-400:]
            set_stage(job, "Something went wrong", job["progress"], "error")


def enqueue(job):
    _video_q.put(job)


def start():
    """Load what is on disk and start the workers. Jobs cut off by a restart are marked, not silently lost."""
    global _started
    if _started:
        return
    _started = True
    for f in sorted(config.JOBS.glob("*/job.json")):
        try:
            job = json.loads(f.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        if job.get("status") in ("queued", "working", "live", "waiting"):
            job.update(status="stopped", stage="Stopped when ClipStorm was closed")
            for c in job.get("clips", []):
                if c.get("status") in ("queued", "rendering"):
                    c.update(status="error", error="ClipStorm was closed before this clip was made.")
        JOBS[job["id"]] = job
    threading.Thread(target=_video_worker, daemon=True).start()
    threading.Thread(target=_render_worker, daemon=True).start()


def summary() -> list:
    with _lock:
        jobs = sorted(JOBS.values(), key=lambda j: -j["created"])
    return [{k: v for k, v in j.items() if k not in ("file", "stop")} for j in jobs]
