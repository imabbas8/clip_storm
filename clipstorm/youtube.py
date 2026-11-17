"""Posting finished shorts to your own YouTube channel (optional).

You make one "Web application" OAuth client in Google Cloud, paste its id and secret in Settings, YouTube, and
press Connect. ClipStorm keeps the refresh token in data/settings.json on this machine and nowhere else."""
import secrets
import time
import urllib.parse

import httpx

from . import config, jobs

AUTH = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN = "https://oauth2.googleapis.com/token"
SCOPES = "https://www.googleapis.com/auth/youtube.upload https://www.googleapis.com/auth/youtube.readonly"
_states: dict = {}


class YTError(Exception):
    pass


def auth_url(redirect_uri: str) -> str:
    s = config.get()
    if not s["yt_client_id"] or not s["yt_client_secret"]:
        raise YTError("Save the Google client id and secret first.")
    state = secrets.token_urlsafe(24)
    _states[state] = (time.time(), redirect_uri)
    q = {"client_id": s["yt_client_id"], "redirect_uri": redirect_uri, "response_type": "code", "scope": SCOPES,
         "access_type": "offline", "prompt": "consent", "state": state}
    return AUTH + "?" + urllib.parse.urlencode(q)


def finish(code: str, state: str) -> str:
    made, redirect_uri = _states.pop(state, (0, ""))
    if not redirect_uri or time.time() - made > 900:
        raise YTError("That sign-in took too long or was not started here. Press Connect again.")
    s = config.get()
    r = httpx.post(TOKEN, data={"code": code, "client_id": s["yt_client_id"], "client_secret": s["yt_client_secret"],
                                "redirect_uri": redirect_uri, "grant_type": "authorization_code"}, timeout=30)
    if r.status_code != 200:
        raise YTError(f"Google refused the sign-in: {r.text[:200]}")
    d = r.json()
    if not d.get("refresh_token"):
        raise YTError("Google did not give a lasting sign-in. Remove ClipStorm at myaccount.google.com/permissions and connect again.")
    config.update({"yt_tokens": {"refresh_token": d["refresh_token"], "access_token": d.get("access_token", ""),
                                 "expires": time.time() + int(d.get("expires_in", 3000)) - 60}})
    return test()["message"]


def _token() -> str:
    s = config.get()
    t = s["yt_tokens"]
    if not t.get("refresh_token"):
        raise YTError("YouTube is not connected yet.")
    if t.get("access_token") and t.get("expires", 0) > time.time():
        return t["access_token"]
    r = httpx.post(TOKEN, data={"refresh_token": t["refresh_token"], "client_id": s["yt_client_id"],
                                "client_secret": s["yt_client_secret"], "grant_type": "refresh_token"}, timeout=30)
    if r.status_code != 200:
        raise YTError("Google no longer accepts the saved sign-in. Connect YouTube again.")
    d = r.json()
    t.update(access_token=d["access_token"], expires=time.time() + int(d.get("expires_in", 3000)) - 60)
    config.update({"yt_tokens": t})
    return t["access_token"]


def test() -> dict:
    try:
        r = httpx.get("https://www.googleapis.com/youtube/v3/channels?part=snippet&mine=true",
                      headers={"Authorization": f"Bearer {_token()}"}, timeout=30)
    except YTError as e:
        return {"ok": False, "message": str(e)}
    except httpx.HTTPError as e:
        return {"ok": False, "message": f"Could not reach YouTube: {type(e).__name__}"}
    if r.status_code != 200:
        return {"ok": False, "message": f"YouTube answered HTTP {r.status_code}: {r.text[:200]}"}
    items = r.json().get("items") or []
    if not items:
        return {"ok": False, "message": "This Google account has no YouTube channel yet. Create one, then connect again."}
    name = items[0]["snippet"]["title"]
    config.update({"yt_channel": name})
    return {"ok": True, "message": f"Connected to the YouTube channel \"{name}\"."}


def disconnect():
    config.update({"yt_tokens": {}, "yt_channel": "", "yt_auto_upload": False})


def upload_clip(job, clip) -> dict:
    """Send one finished short. The result is written on the clip, so the page can show the link or the reason."""
    path = jobs.job_dir(job["id"]) / "clips" / clip["file"]
    s = config.get()
    clip["youtube"] = {"status": "uploading"}
    jobs.save(job)
    try:
        tags = [t for t in clip.get("hashtags", []) if t][:8]
        title = (clip.get("title") or clip.get("hook") or "Short").strip()[:90]
        if "#shorts" not in title.lower() and len(title) <= 92:
            title += " #shorts"
        body = {"snippet": {"title": title[:100], "categoryId": "24", "tags": tags,
                            "description": ((clip.get("description") or "") + "\n\n" + " ".join("#" + t for t in tags)).strip()[:4900]},
                "status": {"privacyStatus": s["yt_privacy"] if s["yt_privacy"] in ("private", "unlisted", "public") else "private",
                           "selfDeclaredMadeForKids": False}}
        head = {"Authorization": f"Bearer {_token()}"}
        size = path.stat().st_size
        r = httpx.post("https://www.googleapis.com/upload/youtube/v3/videos?uploadType=resumable&part=snippet,status",
                       headers={**head, "X-Upload-Content-Type": "video/mp4", "X-Upload-Content-Length": str(size)},
                       json=body, timeout=60)
        if r.status_code != 200 or "location" not in r.headers:
            raise YTError(f"YouTube refused the upload (HTTP {r.status_code}): {r.text[:200]}")
        with open(path, "rb") as f:
            r = httpx.put(r.headers["location"], headers={**head, "Content-Type": "video/mp4"}, content=f.read(), timeout=900)
        if r.status_code not in (200, 201):
            raise YTError(f"The upload stopped (HTTP {r.status_code}): {r.text[:200]}")
        vid = r.json()["id"]
        clip["youtube"] = {"status": "done", "url": f"https://youtube.com/shorts/{vid}", "privacy": body["status"]["privacyStatus"]}
        jobs.log(job, f"Posted to YouTube ({body['status']['privacyStatus']}): {title}")
    except Exception as e:  # noqa: BLE001
        clip["youtube"] = {"status": "error", "error": str(e)[:300]}
    jobs.save(job)
    return clip["youtube"]
