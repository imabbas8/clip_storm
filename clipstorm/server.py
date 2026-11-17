"""The web server: one page, a small JSON API, and the clip files. Run with:  python -m clipstorm.server"""
import base64
import re
import secrets
import shutil
import threading
import time
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, Response

from . import ai, asr, config, jobs, live, media, render, watch, youtube

app = FastAPI(title="ClipStorm", docs_url=None, redoc_url=None, openapi_url=None)
VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi", ".ts", ".flv"}


@app.middleware("http")
async def gate(request: Request, call_next):
    """With CLIPSTORM_PASSWORD set, everything asks for it. Needed whenever other people can reach this address."""
    if config.PASSWORD and not request.url.path.startswith("/oauth/"):
        got = request.headers.get("authorization", "")
        ok = False
        if got.lower().startswith("basic "):
            try:
                ok = secrets.compare_digest(base64.b64decode(got[6:]).decode().split(":", 1)[1], config.PASSWORD)
            except Exception:  # noqa: BLE001
                ok = False
        if not ok:
            return Response("Password needed.", 401, {"WWW-Authenticate": 'Basic realm="ClipStorm"'})
    if request.method not in ("GET", "HEAD"):          # a page on another site must not be able to press our buttons
        origin = request.headers.get("origin")
        if origin and origin.split("//", 1)[-1] != request.headers.get("host"):
            return JSONResponse({"detail": "Request came from another site."}, 403)
    return await call_next(request)


def _job(jid):
    job = jobs.JOBS.get(jid)
    if not job:
        raise HTTPException(404, "That job no longer exists.")
    return job


@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse((config.WEB / "index.html").read_text(encoding="utf-8"), headers={"Cache-Control": "no-store"})


@app.get("/api/state")
def state():
    return {"settings": config.public(), "jobs": jobs.summary(), "ai_ready": ai.ready(),
            "providers": ai.catalog(), "missing": media.tools_missing(), "channels": watch.summary(),
            "styles": {k: v["label"] for k, v in render.STYLES.items()}, "layouts": render.LAYOUTS}


@app.post("/api/settings")
async def save_settings(request: Request):
    config.update(await request.json())
    return {"settings": config.public(), "ai_ready": ai.ready()}


@app.post("/api/ai/test")
async def ai_test(request: Request):
    body = await request.json()
    out = ai.test(body)
    if out["ok"]:
        config.update(body)                    # a test that passes is also a save: one button, nothing to forget
    return out


@app.post("/api/ai/models")
async def ai_models(request: Request):
    return ai.models(await request.json())


@app.post("/api/asr/test")
async def asr_test(request: Request):
    body = await request.json()
    out = asr.test(body)
    if out["ok"]:
        config.update(body)
    return out


@app.post("/api/start")
async def start(request: Request):
    """Paste a link: a finished video is downloaded and clipped, a live stream is watched until it ends."""
    body = await request.json()
    if media.tools_missing():
        raise HTTPException(400, "ffmpeg is not installed on this machine. Install it, then try again.")
    try:
        url = media.normalize(body.get("url", ""))
    except media.MediaError as e:
        raise HTTPException(400, str(e)) from e
    info = media.inspect(url)
    want_live = body.get("mode") == "live"
    if not info["ok"] and not (info.get("waiting") and (want_live or config.get()["wait_for_live"])):
        raise HTTPException(400, info["error"])
    if not info["ok"] or info["live"] or want_live:
        job = jobs.create("live", {"type": "url", "url": url}, info.get("title") or url)
        job["uploader"] = info.get("uploader", "")
        jobs.set_stage(job, "Connecting to the stream", status="live" if info["ok"] else "waiting")
        live.start(job)
    else:
        job = jobs.create("video", {"type": "url", "url": info.get("page") or url}, info["title"])
        job.update(uploader=info["uploader"], duration=info["duration"])
        jobs.save(job)
        jobs.enqueue(job)
    return {"id": job["id"], "kind": job["kind"]}


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    name = re.sub(r"[^\w.\- ]", "_", Path(file.filename or "video.mp4").name)[-120:]
    if Path(name).suffix.lower() not in VIDEO_EXT:
        raise HTTPException(400, "That is not a video file ClipStorm knows (mp4, mov, mkv, webm).")
    job = jobs.create("video", {"type": "file", "path": ""}, Path(name).stem)
    dest = jobs.job_dir(job["id"]) / ("source" + Path(name).suffix.lower())
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f, 1024 * 1024)
    job["source"]["path"] = str(dest)
    jobs.save(job)
    jobs.enqueue(job)
    return {"id": job["id"], "kind": "video"}


@app.post("/api/jobs/{jid}/stop")
def stop(jid: str):
    job = _job(jid)
    s = live.get(jid)
    if s:
        s.stop()
    else:
        job["stop"] = True
        if job["status"] in ("queued", "waiting"):
            jobs.set_stage(job, "Stopped", status="stopped")
    return {"ok": True}


@app.post("/api/jobs/{jid}/clip")
async def clip_now(jid: str, request: Request):
    s = live.get(jid)
    if not s:
        raise HTTPException(400, "This stream is not being recorded any more.")
    try:
        s.clip_last(max(10, min(180, int((await request.json()).get("seconds", 45)))))
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    return {"ok": True}


@app.delete("/api/jobs/{jid}")
def remove(jid: str):
    _job(jid)
    s = live.get(jid)
    if s:
        s.stop()
        time.sleep(1.5)
    jobs.delete(jid)
    return {"ok": True}


@app.get("/media/{jid}/{name}")
def clip_file(jid: str, name: str, download: int = 0):
    if not re.fullmatch(r"[A-Za-z0-9]+", jid) or not re.fullmatch(r"clip\d{3}\.(mp4|jpg)", name):
        raise HTTPException(404)
    path = config.JOBS / jid / "clips" / name
    if not path.is_file():
        raise HTTPException(404)
    if download:
        job = jobs.JOBS.get(jid) or {}
        clip = next((c for c in job.get("clips", []) if c["id"] == name[:7]), {})
        nice = re.sub(r"[^\w\- ]", "", clip.get("title") or name[:7]).strip()[:60] or name[:7]
        return FileResponse(path, filename=f"{nice}.mp4")
    return FileResponse(path)


@app.post("/api/jobs/{jid}/clips/{cid}/edit")
async def clip_edit(jid: str, cid: str, request: Request):
    job = _job(jid)
    clip = next((c for c in job["clips"] if c["id"] == cid), None)
    if not clip:
        raise HTTPException(404, "That short no longer exists.")
    if clip["status"] in ("queued", "rendering"):
        raise HTTPException(400, "This short is still being made.")
    try:
        jobs.reedit(job, clip, await request.json())
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    return {"ok": True}


@app.delete("/api/jobs/{jid}/clips/{cid}")
def clip_delete(jid: str, cid: str):
    if not re.fullmatch(r"clip\d{3}", cid):
        raise HTTPException(404)
    jobs.delete_clip(_job(jid), cid)
    return {"ok": True}


# ---------------------------------------------------------------- the watchlist
@app.post("/api/channels")
async def channel_add(request: Request):
    body = await request.json()
    try:
        return watch.add(body.get("url", ""), body.get("name", ""))
    except (ValueError, media.MediaError) as e:
        raise HTTPException(400, str(e)) from e


@app.post("/api/channels/{cid}")
async def channel_update(cid: str, request: Request):
    body = await request.json()
    if "auto" in body:
        watch.update(cid, body["auto"])
    if body.get("check"):
        ch = next((c for c in watch.load() if c["id"] == cid), None)
        if ch:
            threading.Thread(target=watch.check, args=(ch,), daemon=True).start()
    return {"ok": True}


@app.delete("/api/channels/{cid}")
def channel_remove(cid: str):
    watch.remove(cid)
    return {"ok": True}


# ---------------------------------------------------------------- YouTube
@app.get("/oauth/youtube/start")
def yt_start(request: Request):
    try:
        return RedirectResponse(youtube.auth_url(str(request.base_url).rstrip("/") + "/oauth/youtube/callback"))
    except youtube.YTError as e:
        raise HTTPException(400, str(e)) from e


@app.get("/oauth/youtube/callback")
def yt_callback(code: str = "", state: str = "", error: str = ""):
    try:
        if error or not code:
            raise youtube.YTError("Google sign-in was cancelled.")
        youtube.finish(code, state)
    except youtube.YTError as e:
        return HTMLResponse(f"<p style='font:16px sans-serif;padding:40px'>{str(e)}<br><br><a href='/'>Back to ClipStorm</a></p>", 400)
    return RedirectResponse("/#settings")


@app.post("/api/youtube/signin")
def yt_signin():
    """Is a browser on this machine signed in to YouTube? Only cookie names are checked."""
    found = media.browser_signin(fresh=True)
    signed = [b for b, ok in found.items() if ok]
    if signed:
        return {"ok": True, "message": f"Signed in to YouTube in: {', '.join(b.title() for b in signed)}. "
                                         "ClipStorm will use it for YouTube links."}
    return {"ok": False, "message": media.youtube_help()}


@app.get("/api/youtube/redirect")
def yt_redirect(request: Request):
    return {"uri": str(request.base_url).rstrip("/") + "/oauth/youtube/callback"}


@app.post("/api/youtube/test")
def yt_test():
    return youtube.test()


@app.post("/api/youtube/disconnect")
def yt_disconnect():
    youtube.disconnect()
    return {"ok": True}


@app.post("/api/jobs/{jid}/clips/{cid}/youtube")
def yt_upload(jid: str, cid: str):
    job = _job(jid)
    clip = next((c for c in job["clips"] if c["id"] == cid and c["status"] == "done"), None)
    if not clip:
        raise HTTPException(404, "That clip is not ready.")
    if not config.get()["yt_tokens"].get("refresh_token"):
        raise HTTPException(400, "Connect YouTube first (Settings, YouTube).")
    if (clip.get("youtube") or {}).get("status") == "uploading":
        raise HTTPException(400, "This clip is already being posted.")
    threading.Thread(target=youtube.upload_clip, args=(job, clip), daemon=True).start()
    return {"ok": True}


def main():
    import uvicorn
    jobs.start()
    watch.start()
    where = f"http://{'127.0.0.1' if config.HOST in ('0.0.0.0', '::') else config.HOST}:{config.PORT}"
    print(f"\n  ClipStorm is running. Open {where}\n")
    if config.HOST not in ("127.0.0.1", "localhost") and not config.PASSWORD:
        print("  Careful: other computers can reach this address and no password is set (CLIPSTORM_PASSWORD).\n")
    uvicorn.run(app, host=config.HOST, port=config.PORT, log_level="warning")


if __name__ == "__main__":
    main()
