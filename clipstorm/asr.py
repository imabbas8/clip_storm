"""Speech to text with a time for every word.

local   faster-whisper on this machine: free, private, slower on a small computer
groq    Groq's hosted Whisper: very fast, has a free tier
openai  OpenAI's hosted Whisper

Every backend returns the same thing: [{"w": word, "s": start, "e": end}, ...] and the language it heard."""
import io
import threading
import wave

import httpx
import numpy as np

from . import config

SR = 16000
REMOTE = {"groq": ("https://api.groq.com/openai/v1", "whisper-large-v3-turbo"),
          "openai": ("https://api.openai.com/v1", "whisper-1")}
_model = {"name": None, "obj": None}
_lock = threading.Lock()          # one transcription at a time: a small machine has nothing to gain from two


class ASRError(Exception):
    pass


def _key(s) -> str:
    if s["asr_key"].strip():
        return s["asr_key"].strip()
    return s["ai_key"].strip() if s["ai_provider"] == s["asr_provider"] else ""      # same service: one key does both


def _local(name):
    if _model["name"] != name:
        from faster_whisper import WhisperModel
        import os
        _model["obj"] = WhisperModel(name, device="cpu", compute_type="int8", cpu_threads=max(1, (os.cpu_count() or 2)),
                                     download_root=str(config.MODELS))
        _model["name"] = name
    return _model["obj"]


def _transcribe_local(audio, s):
    m = _local(s["asr_local_model"] or "base")
    segs, info = m.transcribe(audio.astype(np.float32) / 32768.0, language=s["asr_language"] or None,
                              word_timestamps=True, vad_filter=True, beam_size=1,
                              condition_on_previous_text=False)
    words = []
    for seg in segs:
        for w in seg.words or []:
            t = w.word.strip()
            if t:
                words.append({"w": t, "s": round(float(w.start), 2), "e": round(float(w.end), 2)})
    return words, info.language or ""


def _wav_bytes(audio) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(SR)
        f.writeframes(audio.astype(np.int16).tobytes())
    return buf.getvalue()


def _transcribe_remote(audio, s):
    base, model = REMOTE[s["asr_provider"]]
    key = _key(s)
    if not key:
        raise ASRError(f"Add a {s['asr_provider'].title()} key for speech to text (Settings, Speech).")
    data = {"model": model, "response_format": "verbose_json", "timestamp_granularities[]": "word"}
    if s["asr_language"]:
        data["language"] = s["asr_language"]
    try:
        r = httpx.post(f"{base}/audio/transcriptions", headers={"Authorization": f"Bearer {key}"}, data=data,
                       files={"file": ("audio.wav", _wav_bytes(audio), "audio/wav")}, timeout=300)
    except httpx.HTTPError as e:
        raise ASRError(f"Could not reach the speech service: {type(e).__name__}") from e
    if r.status_code in (401, 403):
        raise ASRError("The speech service refused the key. Check it in Settings, Speech.")
    if r.status_code != 200:
        raise ASRError(f"The speech service answered HTTP {r.status_code}: {r.text[:200]}")
    d = r.json()
    words = [{"w": str(w.get("word", "")).strip(), "s": round(float(w.get("start", 0)), 2), "e": round(float(w.get("end", 0)), 2)}
             for w in d.get("words") or [] if str(w.get("word", "")).strip()]
    if not words:                         # a service that only gave sentences: spread each one's words over its time
        for seg in d.get("segments") or []:
            parts = str(seg.get("text", "")).split()
            a, b = float(seg.get("start", 0)), float(seg.get("end", 0))
            step = (b - a) / max(1, len(parts))
            words += [{"w": p, "s": round(a + i * step, 2), "e": round(a + (i + 1) * step, 2)} for i, p in enumerate(parts)]
    return words, str(d.get("language") or "")[:2].lower()


def transcribe(audio: np.ndarray, settings=None):
    """Words of one piece of audio (16 kHz mono int16), times counted from its start. Returns (words, language)."""
    s = settings or config.get()
    if len(audio) < SR // 2:
        return [], ""
    with _lock:
        if s["asr_provider"] in REMOTE:
            return _transcribe_remote(audio, s)
        return _transcribe_local(audio, s)


def quiet_cut(audio: np.ndarray, lo: float, hi: float) -> int:
    """The quietest point between lo and hi seconds, as a sample index, so a piece never ends inside a word."""
    a, b = int(lo * SR), min(len(audio), int(hi * SR))
    if b <= a:
        return len(audio)
    hop = 800
    x = audio[a:b].astype(np.float32)
    n = len(x) // hop
    if n < 2:
        return b
    e = (x[: n * hop].reshape(n, hop) ** 2).mean(axis=1)
    return a + int(np.argmin(e)) * hop + hop // 2


def transcribe_long(audio: np.ndarray, progress=None, stop=None):
    """A whole video, piece by piece, so progress can be shown and a long file never sits in memory twice."""
    s = config.get()
    piece = 600 if s["asr_provider"] in REMOTE else 30          # hosted Whisper takes big pieces; local stays responsive
    words, lang, pos, total = [], "", 0, len(audio)
    while pos < total:
        if stop and stop():
            break
        rest = audio[pos:]
        cut = len(rest) if len(rest) <= piece * SR else quiet_cut(rest, piece * 0.8, piece)
        got, l = transcribe(rest[:cut], s)
        off = pos / SR
        words += [{"w": w["w"], "s": round(w["s"] + off, 2), "e": round(w["e"] + off, 2)} for w in got]
        lang = lang or l
        pos += cut
        if progress:
            progress(pos / total)
    return words, lang


def test(over=None) -> dict:
    s = {**config.get(), **{k: v for k, v in (over or {}).items() if v not in (None, "") and set(str(v)) != {"•"}}}
    t = np.arange(SR * 2) / SR
    tone = (np.sin(2 * np.pi * 220 * t) * 3000).astype(np.int16)      # two seconds of hum: no words, but a full round trip
    try:
        transcribe(tone, s)
    except ASRError as e:
        return {"ok": False, "message": str(e)}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "message": f"Speech to text failed: {str(e)[:200]}"}
    where = ("on this machine with the " + (s["asr_local_model"] or "base") + " model") if s["asr_provider"] not in REMOTE \
        else f"through {s['asr_provider'].title()}"
    return {"ok": True, "message": f"Speech to text works {where}."}
