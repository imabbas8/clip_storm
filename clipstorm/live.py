"""Live streams: record into a rolling buffer, write down what is said as it happens, and cut a short whenever
something worth clipping happens, for as long as the stream runs.

One clock for everything. The stream is saved in pieces of a few seconds; each finished piece's own audio is
what gets transcribed and measured, so a word's time and the picture it belongs to can never drift apart.

The AI is asked in three situations: on a schedule while people are talking, right after the stream suddenly
gets louder or chat busier, and when someone says "clip that"."""
import collections
import json
import re
import socket
import ssl
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

from . import ai, asr, config, jobs, media, scout

SESSIONS: dict = {}
SR = 16000
SEG_SEC = 6
PAD = 8                    # seconds kept either side of a clip, so it can be trimmed again later
MAX_HOURS = 12
SPIKE_BEFORE, SPIKE_AFTER, SPIKE_GAP = 45, 18, 60     # stream sent around a jump; least time between two jumps
SAID_BEFORE, SAID_AFTER, SAID_GAP = 60, 12, 45
JUMP = 0.6                                            # 60% louder or busier than the last three minutes
CLIP_WORDS = re.compile(r"\bclip\s+(it|that|this)\b", re.I)


def resolve(url):
    """The stream's playable address, or why there is none yet."""
    info = media.inspect(url)
    if info["ok"] and not info["live"]:
        return {"ok": False, "waiting": True, "error": "This link is not live right now."}
    if info["ok"]:
        info["input"] = ["-reconnect", "1", "-reconnect_streamed", "1", "-reconnect_delay_max", "10"]
    return info


class Session:
    def __init__(self, job, resolver=resolve):
        self.job, self.o, self.resolver = job, job["options"], resolver
        self.dir = jobs.job_dir(job["id"])
        self.buf = self.dir / "buffer"
        self.buf.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.stop_flag = threading.Event()
        self.ended = False                 # recording is over (stream ended, stopped, or failed)
        self.proc = None
        self.part = 0
        self.rows_seen = {}
        self.segments = []                 # {"path", "s", "e"} on the session clock
        self.clock = 0.0                   # seconds of stream received so far
        self.pending = np.zeros(0, dtype=np.int16)   # audio not yet transcribed
        self.pending_at = 0.0              # session time where `pending` starts
        self.tail = np.zeros(0, dtype=np.int16)      # under a second of audio waiting for its loudness figure
        self.db = []                       # loudness per second
        self.words = []
        self.t_done = 0.0                  # transcript is complete up to here
        self.chat = collections.Counter()
        self.chat_total, self.chat_on = 0, False
        self.last_check = 0.0
        self.spike = self.said = None
        self.last_spike = self.last_said = -999.0
        self.manual = []
        self.verdicts = []
        self.checks = 0
        self.started_at = time.time()
        self.last_clip_at = time.time()
        self.last_save = 0.0
        self.state = "connecting"

    # ---------------------------------------------------------------- recording
    def start(self):
        SESSIONS[self.job["id"]] = self
        threading.Thread(target=self._record_safe, daemon=True).start()
        threading.Thread(target=self._analyze, daemon=True).start()
        m = re.match(r"^https?://(?:www\.)?twitch\.tv/([A-Za-z0-9_]+)", self.job["source"].get("url", ""))
        if m:
            threading.Thread(target=self._chat, args=(m.group(1).lower(),), daemon=True).start()

    def stop(self, note="Stopped by you"):
        self.note = note
        self.job["stop"] = True
        self.stop_flag.set()
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()

    def _record_safe(self):
        try:
            self._record()
        except Exception as e:  # noqa: BLE001
            self.job["error"] = f"Recording stopped: {str(e)[-200:]}"
        finally:
            self.ended = True

    def _record(self):
        fails = 0
        url = self.job["source"].get("url", "")
        while not self.stop_flag.is_set():
            info = self.resolver(url)
            if not info["ok"]:
                if self.clock == 0 and info.get("waiting") and config.get()["wait_for_live"]:
                    self.state = "waiting"                      # not live yet: keep looking, without giving up
                    jobs.set_stage(self.job, "Waiting for the stream to go live", status="waiting")
                    self.stop_flag.wait(45)
                    continue
                fails += 1
                if self.clock == 0 or fails >= 3:
                    if self.clock == 0:
                        self.job["error"] = info["error"]
                    return
                self.stop_flag.wait(10)
                continue
            if not self.job.get("title") or self.job["title"] == url:
                self.job.update(title=info.get("title") or url, uploader=info.get("uploader", ""))
            self.part += 1
            p = f"p{self.part:02d}"
            cmd = ["ffmpeg", "-v", "error"]
            for u in info["urls"]:
                cmd += list(info.get("input", [])) + ["-i", u]
            a_in = "1:a:0" if len(info["urls"]) > 1 else "0:a:0"
            cmd += ["-map", "0:v:0", "-map", a_in, "-c", "copy", "-f", "segment", "-segment_time", str(SEG_SEC),
                    "-reset_timestamps", "1", "-segment_format", "mpegts", "-segment_list", f"{p}.csv",
                    "-segment_list_type", "csv", f"{p}_%05d.ts"]
            t0 = time.time()
            with open(self.dir / "ffmpeg.log", "ab") as errlog:
                self.proc = subprocess.Popen(cmd, cwd=self.buf, stdout=subprocess.DEVNULL, stderr=errlog)
                while self.proc.poll() is None:
                    if self.stop_flag.wait(1):
                        self.proc.terminate()
                        try:
                            self.proc.wait(8)
                        except subprocess.TimeoutExpired:
                            self.proc.kill()
                    if time.time() - self.started_at > MAX_HOURS * 3600:
                        self.stop(f"Stopped after {MAX_HOURS} hours, the longest one recording runs")
            fails = 0 if time.time() - t0 > 30 else fails + 1    # ran a while: a normal hiccup, connect again
            if fails >= 3:
                return
            self.stop_flag.wait(3)

    def _scan(self):
        """Take in the pieces ffmpeg has finished: their audio joins the transcript queue and the loudness record."""
        for i in range(1, self.part + 1):
            f = self.buf / f"p{i:02d}.csv"
            if not f.exists():
                continue
            rows = [r.split(",") for r in f.read_text(encoding="utf-8", errors="ignore").splitlines() if r.count(",") >= 2]
            for row in rows[self.rows_seen.get(i, 0):]:
                path = self.buf / row[0]
                audio = media.pcm(path) if path.exists() else np.zeros(0, dtype=np.int16)
                if not len(audio):
                    continue
                with self.lock:
                    s = self.clock
                    self.clock += len(audio) / SR
                    self.segments.append({"path": path, "s": s, "e": self.clock})
                    self.pending = np.concatenate([self.pending, audio])
                    both = np.concatenate([self.tail, audio])
                    whole = len(both) // SR
                    self.db += media.loudness(both[: whole * SR])
                    self.tail = both[whole * SR:]
            self.rows_seen[i] = len(rows)
        keep_from = self.clock - self.o["buffer_min"] * 60
        with self.lock:
            while self.segments and self.segments[0]["e"] < keep_from:
                self.segments.pop(0)["path"].unlink(missing_ok=True)

    # ---------------------------------------------------------------- chat (Twitch only, read-only, no account)
    def _chat(self, channel):
        while not self.stop_flag.is_set() and not self.ended:
            try:
                raw = socket.create_connection(("irc.chat.twitch.tv", 6697), timeout=20)
                s = ssl.create_default_context().wrap_socket(raw, server_hostname="irc.chat.twitch.tv")
                s.sendall(f"NICK justinfan{int(time.time()) % 90000 + 10000}\r\nJOIN #{channel}\r\n".encode())
                s.settimeout(30)
                rest, self.chat_on = b"", True
                while not self.stop_flag.is_set() and not self.ended:
                    try:
                        data = s.recv(8192)
                    except socket.timeout:
                        continue
                    if not data:
                        break
                    *lines, rest = (rest + data).split(b"\r\n")
                    for line in lines:
                        if line.startswith(b"PING"):
                            s.sendall(b"PONG :tmi.twitch.tv\r\n")
                        elif b" PRIVMSG #" in line:
                            with self.lock:
                                self.chat[int(self.clock)] += 1
                                self.chat_total += 1
                s.close()
            except Exception:  # noqa: BLE001
                pass
            self.chat_on = False
            self.stop_flag.wait(5)

    def _chat_busy(self, s, e):
        if not self.chat_total or self.clock < 60:
            return False
        avg = self.chat_total / self.clock
        a, b = int(s), int(e) + 6
        return sum(self.chat.get(t, 0) for t in range(a, b + 1)) / (b - a + 1) > max(3 * avg, 1.0)

    # ---------------------------------------------------------------- transcript
    def _transcribe_step(self) -> bool:
        with self.lock:
            n = len(self.pending)
            hurry = self.ended or bool(self.manual)
            if n < SR * (1 if hurry else 12):
                return False
            audio, base = self.pending, self.pending_at
        cut = n
        if n > SR * 28:                       # fallen behind: a bigger bite, still ending in a quiet spot
            cut = asr.quiet_cut(audio, 20, 28)
        elif not hurry:
            cut = asr.quiet_cut(audio, 7, n / SR)
        try:
            words, lang = asr.transcribe(audio[:cut])
        except Exception as e:  # noqa: BLE001 - the audio is dropped, the stream goes on
            words, lang = [], ""
            self._verdict("Speech to text failed: " + str(e)[:120])
        words = [{"w": w["w"], "s": round(w["s"] + base, 2), "e": round(w["e"] + base, 2)} for w in words]
        with self.lock:
            self.pending = self.pending[cut:]
            self.pending_at += cut / SR
            self.words += words
            self.t_done = self.pending_at
        if lang and not self.job.get("language"):
            self.job["language"] = lang
        for w, nxt in zip(words, words[1:] + [{"w": ""}]):
            if CLIP_WORDS.search(f"{w['w']} {nxt['w']}") and w["s"] - self.last_said > SAID_GAP and not self.said:
                self.said = {"t": w["s"], "why": 'Someone said "clip that"'}
                self.last_said = w["s"]
        return True

    # ---------------------------------------------------------------- deciding
    def _watch_spikes(self):
        """Compare the last five seconds with the three minutes before them. A jump is only marked here; it is
        looked at once the words around it have been written down."""
        with self.lock:
            n = len(self.db)
            if n < 60:
                return
            db = np.array(self.db[max(0, n - 190):], dtype=np.float32)
            chat = [self.chat.get(t, 0) for t in range(max(0, n - 190), n)] if self.chat_total else None
        amp = 10 ** (db / 20)
        why = []
        jump = float(amp[-5:].mean()) / max(float(amp[:-10].mean()), 1e-6) - 1
        if jump >= JUMP:
            why.append(f"loudness +{round(jump * 100)}%")
        if chat:
            base = sum(chat[:-10]) / max(1, len(chat) - 10)
            now = sum(chat[-8:]) / 8
            if now >= max(1.5, base * (1 + JUMP)):
                why.append("chat got busy")
        if why and not self.spike and n - self.last_spike >= SPIKE_GAP:
            self.spike = {"t": float(n - 3), "why": "Jump: " + ", ".join(why)}
            self.last_spike = n

    def _verdict(self, text):
        self.verdicts.append({"t": round(self.t_done), "m": text})
        del self.verdicts[:-8]

    def _clips_last_hour(self):
        return sum(1 for c in self.job["clips"] if time.time() - c.get("made_at", 0) < 3600 and not c.get("manual"))

    def _check(self, window=None, why="Regular check"):
        o = self.o
        now = self.t_done
        ws = max(0.0, window[0]) if window else max(0.0, now - o["live_check_min"] * 60 - 60)
        now = min(now, window[1]) if window else now
        self.last_check = self.t_done
        win = [w for w in self.words if ws <= w["s"] <= now]
        if len(win) < 8:
            return self._verdict(f"{why}: nobody talking, skipped")
        if self._clips_last_hour() >= o["live_per_hour"]:
            return self._verdict(f"{why}: already {o['live_per_hour']} clips this hour, skipped")
        with self.lock:
            z0 = max(0, int(now) - 600)
            z = scout.zscore(self.db[z0: int(now) + 1])
        if not ai.ready():                         # no AI: a jump in loudness is itself the reason to clip
            if not window:
                return
            picks = scout.by_loudness(self.db[z0: int(now) + 1], [dict(w, s=w["s"] - z0, e=w["e"] - z0) for w in win
                                                                  ], 1, o["min_len"], o["max_len"], now - z0)
            for c in picks:
                c.update(start=round(c["start"] + z0, 2), end=round(c["end"] + z0, 2), hook="")
                if not scout._overlaps(c["start"], c["end"], self.job["clips"]):
                    self._add(c, f"{why}: clipped by loudness")
            return
        force = int(o.get("force_min") or 0)       # off unless asked for: a quiet stream should give no shorts, not weak ones
        must = bool(force) and not window and time.time() - self.last_clip_at >= force * 60
        try:
            cands, tok = scout.live_check(win, z, z0, now, self._chat_busy if self.chat_total else None,
                                          self.job.get("title", ""), self.job.get("uploader", ""), o["min_len"],
                                          o["max_len"], self.job["clips"], must=must, agents=o.get("agents", 1))
        except Exception as e:  # noqa: BLE001
            return self._verdict(f"{why}: AI error, {str(e)[:120]}")
        self.checks += 1
        self.job["tokens"] += tok
        need = 50 if must else o["min_score"]
        good = [c for c in cands if c["score"] >= need]
        if not good:
            best = max((c["score"] for c in cands), default=0)
            return self._verdict(f"{why}: nothing clip-worthy" + (f" (best scored {best})" if best else ""))
        for c in good[:2]:
            who = (" + ".join(c["agents"]) + " agreed, ") if len(c.get("agents", [])) > 1 else \
                  (c["agents"][0] + " agent, ") if c.get("agents") else ""
            self._add(c, f"{why}: clipped, {who}viral score {c['score']}")

    def _add(self, c, verdict):
        clip = jobs.new_clip(self.job, c)
        self.last_clip_at = time.time()
        self._verdict(verdict)
        jobs.log(self.job, verdict)
        jobs.submit_render(lambda: self._render(clip))

    def clip_last(self, seconds):
        """The Clip now button: the last N seconds, whatever the AI thinks."""
        if self.clock < 5:
            raise ValueError("Nothing has been recorded yet.")
        self.manual.append({"start": max(0.0, self.clock - seconds), "end": self.clock})

    def _manual_step(self):
        for req in [m for m in self.manual if m["end"] <= self.t_done + 0.5 or self.ended]:
            self.manual.remove(req)
            win = [w for w in self.words if req["start"] <= w["s"] <= req["end"]]
            c = None
            if ai.ready() and len(win) >= 8:
                try:
                    got, tok = scout.live_check(win, None, 0, req["end"], None, self.job.get("title", ""),
                                                self.job.get("uploader", ""), 5, int(req["end"] - req["start"]) + 5, (),
                                                must=True, n=1)
                    self.job["tokens"] += tok
                    c = got[0] if got else None
                except Exception:  # noqa: BLE001
                    c = None
            c = c or {"score": 0, "category": "other", "hook": "", "title": " ".join(w["w"] for w in win[:9])[:70],
                      "description": "", "hashtags": ["shorts"], "highlight_words": [], "reason": ""}
            c.update(start=round(req["start"], 2), end=round(req["end"], 2), manual=True, reason="Clipped by hand.")
            self._add(c, "Clipped by hand")

    # ---------------------------------------------------------------- making the clip
    def _render(self, clip):
        deadline = time.time() + 45
        while time.time() < deadline and not self.ended:           # wait for the stream to reach the clip's end
            with self.lock:
                if self.segments and self.segments[-1]["e"] >= clip["end"] + 0.5:
                    break
            time.sleep(1)
        with self.lock:
            segs = [s for s in self.segments if s["e"] > clip["start"] - PAD and s["s"] < clip["end"] + PAD and s["path"].exists()]
            words = list(self.words)
        if not segs or segs[0]["s"] > clip["start"] + 2:
            clip.update(status="error", error="That part of the stream is no longer in the buffer.")
            return jobs.save(self.job)
        cdir = self.dir / "clips"
        cdir.mkdir(exist_ok=True)
        lst, src = cdir / f"{clip['id']}.txt", cdir / f"{clip['id']}.src.ts"
        lst.write_text("".join(f"file '{s['path'].as_posix()}'\n" for s in segs), encoding="utf-8")
        try:
            media.run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", src], 300)
            clip["start"] = max(clip["start"], segs[0]["s"])
            clip["end"] = min(clip["end"], segs[-1]["e"])
            near = [w for w in words if segs[0]["s"] - 1 <= w["s"] <= segs[-1]["e"] + 1]
            (cdir / f"{clip['id']}.words.json").write_text(json.dumps(near), encoding="utf-8")
            clip.update(src=src.name, src_offset=round(segs[0]["s"], 3), src_end=round(segs[-1]["e"], 3))
            jobs.render_one(self.job, clip, src, near, src_offset=segs[0]["s"])
        except Exception as e:  # noqa: BLE001
            clip.update(status="error", error=str(e)[-300:])
            jobs.save(self.job)
        finally:
            lst.unlink(missing_ok=True)

    # ---------------------------------------------------------------- the loop
    def _publish(self, force=False):
        lines = scout.make_lines([w for w in self.words if w["s"] >= self.t_done - 40])[-4:]
        with self.lock:
            buf = round(self.segments[-1]["e"] - self.segments[0]["s"]) if self.segments else 0
            self.job["duration"] = self.clock
            self.job["live"] = {"elapsed": round(self.clock), "buffer_sec": buf, "lines": [l["text"] for l in lines],
                                "level": [round(x) for x in self.db[-60:]], "checks": self.checks,
                                "words": len(self.words), "verdicts": list(reversed(self.verdicts)),
                                "chat": self.chat_on, "behind": round(self.clock - self.t_done),
                                "hour": self._clips_last_hour(), "ai": ai.ready()}
        if force or time.time() - self.last_save > 3:
            self.last_save = time.time()
            jobs.save(self.job)

    def _analyze(self):
        started = False
        while True:
            busy = False
            try:
                self._scan()
                busy = self._transcribe_step()
                if not started and self.clock > 0:
                    started = True
                    self.state = "live"
                    jobs.set_stage(self.job, "Watching the stream", status="live")
                    jobs.log(self.job, "Recording started")
                self._manual_step()
                if started and not self.stop_flag.is_set():
                    self._watch_spikes()
                    sp, said, every = self.spike, self.said, self.o["live_check_min"] * 60
                    flush = self.ended and not busy
                    if said and (self.t_done >= said["t"] + SAID_AFTER or flush):
                        self.said = None
                        self._check((said["t"] - SAID_BEFORE, said["t"] + SAID_AFTER), said["why"])
                    elif sp and (self.t_done >= sp["t"] + SPIKE_AFTER or flush):
                        self.spike = None
                        self._check((sp["t"] - SPIKE_BEFORE, sp["t"] + SPIKE_AFTER), sp["why"])
                    elif self.t_done - self.last_check >= every or (flush and self.t_done - self.last_check > 45):
                        self._check()
                self._publish()
            except Exception as e:  # noqa: BLE001
                self._verdict("Problem: " + str(e)[:140])
            if self.ended and not busy and len(self.pending) < SR and not self.manual:
                self._scan()
                if len(self.pending) < SR:
                    break
            if not busy:
                time.sleep(1.5)
        failed = bool(self.job.get("error")) and not started
        note = getattr(self, "note", "The stream ended")
        done = threading.Event()
        jobs.submit_render(done.set)                # runs after every clip already waiting to be made
        done.wait()
        for f in self.buf.glob("*"):
            f.unlink(missing_ok=True)
        n = sum(1 for c in self.job["clips"] if c["status"] == "done")
        self._publish(True)
        jobs.set_stage(self.job, "Could not record" if failed else f"{note}: {n} short{'s' if n != 1 else ''} made",
                       1.0, "error" if failed else "done")
        SESSIONS.pop(self.job["id"], None)


def start(job, resolver=resolve) -> Session:
    s = Session(job, resolver)
    s.start()
    return s


def get(jid):
    return SESSIONS.get(jid)
