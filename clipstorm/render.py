"""Turns a time range into an edited vertical short.

The edit, in order: dead air is cut out (jump cuts), the picture punches in and out from shot to shot, the hook
sits on top as a title, every word is captioned as it is said with the punch words in colour, a progress bar
runs along the bottom, and the sound is levelled to the loudness the platforms expect."""
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from . import config

PW, PH = 1080, 1920        # the canvas captions are laid out on; the video itself may be rendered smaller
FPS = 30
# ASS colours are BBGGRR
STYLES = {
    "bounce": {"label": "Bounce", "font": "Poppins ExtraBold", "size": 118, "base": "FFFFFF", "active": "00F0FF",
               "strong": "5CFF7A", "outline": 8, "upper": True, "per": 3, "chars": 16},
    "hype": {"label": "Hype", "font": "Anton", "size": 150, "base": "FFFFFF", "active": "3C3CFF",
             "strong": "00D7FF", "outline": 8, "upper": True, "per": 2, "chars": 14},
    "clean": {"label": "Clean", "font": "Poppins ExtraBold", "size": 88, "base": "FFFFFF", "active": "FFD76B",
              "strong": "FFFFFF", "outline": 5, "upper": False, "per": 4, "chars": 22},
}
LAYOUTS = {"smart": "Smart: follow the face, fill the screen", "zoom": "Zoomed in, blurred fill", "blur": "Whole frame, blurred fill", "crop": "Fill the screen (centre crop)"}
BAR_COLOR = "0xFFE600"
MAX_SHOTS = 16


def _ts(t):
    t = max(0.0, t)
    return f"{int(t // 3600)}:{int(t % 3600 // 60):02d}:{t % 60:05.2f}"


def _clean(text):
    return re.sub(r"[\x00-\x1f\x7f]+", " ", re.sub(r"[{}\\]", "", str(text))).strip()


def _key(word):
    return re.sub(r"[^\w']", "", word.lower())


# ---------------------------------------------------------------- the cut: which parts of the range are kept
def envelope(audio, hop=0.05) -> np.ndarray:
    """Loudness (dB) of 16 kHz audio in steps of `hop` seconds: fine enough to tell a breath from a silence."""
    n = int(16000 * hop)
    k = len(audio) // n
    if k == 0:
        return np.zeros(0, dtype=np.float32)
    x = audio[: k * n].astype(np.float32).reshape(k, n) / 32768.0
    return (20 * np.log10(np.sqrt((x ** 2).mean(axis=1)) + 1e-5)).astype(np.float32)


def plan_cuts(start, end, words, env=None, hop=0.05, jump_cuts=True) -> list:
    """The pieces of [start, end] that stay, as (a, b) in source time. `env` is the clip's own loudness, counted
    from `start`. A pause between words is cut only when it is also quiet: a silent speaker goes, a wordless but
    loud moment (gameplay, a crowd, music, a crash) stays."""
    inside = [w for w in words if w["e"] > start and w["s"] < end]
    if not jump_cuts or len(inside) < 4 or env is None or not len(env):
        return [(start, end)]

    def level(a, b):
        seg = env[max(0, int((a - start) / hop)): max(0, int((b - start) / hop)) + 1]
        return float(np.max(seg)) if len(seg) else -100.0

    quiet_below = float(np.median([level(w["s"], w["e"]) for w in inside])) - 10.0
    keep, a = [], start
    if inside[0]["s"] - start > 0.6 and level(start, inside[0]["s"] - 0.3) < quiet_below:
        a = inside[0]["s"] - 0.25
    for prev, nxt in zip(inside, inside[1:]):
        gap = nxt["s"] - prev["e"]
        if gap > 0.75 and level(prev["e"] + 0.2, nxt["s"] - 0.2) < quiet_below:
            b = prev["e"] + 0.18
            if b - a > 0.3:
                keep.append((a, b))
            a = nxt["s"] - 0.15
    keep.append((a, end))
    keep = [(max(start, x), min(end, y)) for x, y in keep if y - x > 0.2]
    total = sum(y - x for x, y in keep)
    if not keep or total < 0.6 * (end - start) or total < 6:       # cutting that much would gut the clip: leave it whole
        return [(start, end)]
    return keep


def plan_shots(keep, words, min_shot=2.2) -> list:
    """Split the kept pieces at sentence ends into shots, each a little closer or wider than the one before.
    Returns [(a, b, zoom)]."""
    marks = ".?!।؟"
    shots = []
    for a, b in keep:
        cuts = [w["e"] + 0.05 for w in words if a + min_shot < w["e"] < b - min_shot and w["w"][-1:] in marks]
        last = a
        for c in cuts:
            if c - last >= min_shot:
                shots.append((last, c))
                last = c
        shots.append((last, b))
    while len(shots) > MAX_SHOTS:                    # too many: join the shortest neighbours that touch
        i = min(range(len(shots) - 1), key=lambda i: (shots[i][1] - shots[i][0]) + (shots[i + 1][1] - shots[i + 1][0])
                if abs(shots[i][1] - shots[i + 1][0]) < 0.01 else 1e9)
        if abs(shots[i][1] - shots[i + 1][0]) >= 0.01:
            break
        shots[i:i + 2] = [(shots[i][0], shots[i + 1][1])]
    return [(a, b, 1.0 if i % 2 == 0 else 1.12) for i, (a, b) in enumerate(shots)]


def remap(words, shots) -> list:
    """Caption words moved onto the edited timeline (after the cuts)."""
    out, base = [], 0.0
    for a, b, _ in shots:
        for w in words:
            if a - 0.05 <= w["s"] < b:
                out.append(dict(w, s=round(base + max(0.0, w["s"] - a), 2), e=round(base + min(b, w["e"]) - a, 2)))
        base += b - a
    return out


# ---------------------------------------------------------------- following the face
_detector = {"obj": None, "size": None}


def face_track(src, ss, dur, src_w, src_h) -> list:
    """Where the main face is, once a second: a list of x positions (0..1 across the frame), None where no face
    was seen. One pass over small frames; an empty list means face tracking is not available on this machine."""
    try:
        import cv2
    except ImportError:
        return []
    model = Path(__file__).resolve().parent / "models" / "face_yunet.onnx"
    if not model.exists() or not hasattr(cv2, "FaceDetectorYN"):
        return []
    w = 640
    h = int(round(w * src_h / src_w / 2)) * 2
    p = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{ss:.3f}", "-t", f"{dur:.3f}", "-i", str(src), "-vf",
                        f"fps=2,scale={w}:{h}", "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"],
                       capture_output=True, timeout=600)
    n = len(p.stdout) // (w * h * 3)
    if n == 0:
        return []
    frames = np.frombuffer(p.stdout[: n * w * h * 3], dtype=np.uint8).reshape(n, h, w, 3)
    if _detector["size"] != (w, h):
        _detector["obj"] = cv2.FaceDetectorYN.create(str(model), "", (w, h), 0.7, 0.3, 50)
        _detector["size"] = (w, h)
    out = []
    for f in frames:
        _, faces = _detector["obj"].detect(np.ascontiguousarray(f))
        if faces is None or not len(faces):
            out.append(None)
            continue
        big = max(faces, key=lambda b: b[2] * b[3])           # the largest face is nearly always the one talking
        out.append(None if big[2] < w * 0.04 else float((big[0] + big[2] / 2) / w))
    return out


def split_by_focus(shots, track, start, step=0.5, max_shots=24) -> list:
    """Give every shot the face position to hold, and cut a shot in two where the camera itself cut: the face
    jumps across the frame, or appears or disappears. Returns [(a, b, zoom, focus or None)]."""
    out = []
    for a, b, z in shots:
        i0, i1 = max(0, int((a - start) / step)), max(0, int((b - start) / step))
        seg = track[i0: i1 + 1]
        runs, cur = [], []                                   # runs of samples that belong to one camera angle
        for k, x in enumerate(seg):
            ref = [v for v in cur if v is not None]
            med = float(np.median(ref)) if ref else None
            same = (not cur or (x is None and med is None) or (x is not None and med is not None and abs(x - med) < 0.16)
                    or (x is None and sum(v is None for v in cur[-2:]) < 2) or (med is None and len(cur) < 3))
            if not same and len(cur) >= 3:                   # an angle lasts at least a second and a half
                runs.append((k - len(cur), cur))
                cur = []
            cur.append(x)
        runs.append((len(seg) - len(cur), cur))
        if len(runs) > 1 and len(runs[-1][1]) < 3:           # a sliver at the end joins the angle before it
            k, tail = runs.pop()
            runs[-1] = (runs[-1][0], runs[-1][1] + tail)
        for j, (k, vals) in enumerate(runs):
            ra = a if j == 0 else start + (i0 + k) * step
            rb = b if j == len(runs) - 1 else start + (i0 + runs[j + 1][0]) * step
            seen = [v for v in vals if v is not None]
            focus = float(np.median(seen)) if len(seen) >= max(1, 0.5 * len(vals)) else None
            if rb - ra > 0.25:
                out.append((ra, rb, z, focus))
    return out if len(out) <= max_shots else [(a, b, z, None) for a, b, z in shots]


# ---------------------------------------------------------------- captions and hook
def geometry(src_w, src_h, layout):
    """(layout, height of the sharp picture, hook y, caption y) on the 1080x1920 canvas."""
    if src_h >= src_w:
        layout = "crop"                                # already vertical: fill the screen
    if layout == "smart":
        return "smart", PH, 300, 1400
    if layout == "blur":
        fg_h = min(PH, int(round(PW * src_h / src_w / 2)) * 2)
    elif layout == "zoom":
        fg_h = 860
    else:
        return "crop", PH, 300, 1400
    top = (PH - fg_h) // 2
    return layout, fg_h, max(240, top - 170), min(1650, top + fg_h + 190)


def _groups(words, per, chars):
    out, cur = [], []
    for w in words:
        if cur and (len(cur) >= per or sum(len(x["w"]) + 1 for x in cur) + len(w["w"]) > chars
                    or w["s"] - cur[-1]["e"] > 0.45 or cur[-1]["w"][-1:] in ".?!।؟"):
            out.append(cur)
            cur = []
        cur.append(w)
    if cur:
        out.append(cur)
    return out


def build_ass(words, dur, hook, style_name, highlight, hook_y, cap_y, watermark="", hook_dur=None) -> str:
    """`words` are already on the edited timeline, 0 = first frame of the short."""
    st = STYLES.get(style_name, STYLES["bounce"])
    strong = {_key(h) for h in highlight if _key(h)}
    head = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {PW}
PlayResY: {PH}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,{st['font']},{st['size']},&H00{st['base']},&H00{st['base']},&H00000000,&H96000000,0,0,0,0,100,100,0,0,1,{st['outline']},3,5,40,40,0,1
Style: Hook,Poppins ExtraBold,82,&H00141010,&H00141010,&H00FFFFFF,&H00FFFFFF,0,0,0,0,100,100,0,0,3,18,0,5,90,90,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    ev = []
    hook = _clean(hook)
    if hook:
        size = 82 if len(hook) <= 22 else max(52, int(82 * 22 / len(hook)) + 14)
        ev.append(f"Dialogue: 5,{_ts(0)},{_ts(min(dur, hook_dur or dur))},Hook,,0,0,0,,{{\\an5\\pos({PW // 2},{hook_y})\\fs{size}"
                  f"\\fscx70\\fscy70\\t(0,140,\\fscx104\\fscy104)\\t(140,240,\\fscx100\\fscy100)}}{hook}")
    mark = _clean(watermark)[:40]
    if mark:                                               # your channel name, small and see-through, above the bar
        ev.append(f"Dialogue: 1,{_ts(0)},{_ts(dur)},Cap,,0,0,0,,{{\\an5\\pos({PW // 2},{PH - 70})\\fs44\\bord3"
                  f"\\1a&H55&\\3a&H88&}}{mark}")
    ws = sorted(({**w, "w": _clean(w["w"])} for w in words if 0 <= w["s"] < dur), key=lambda w: w["s"])
    ws = [w for w in ws if w["w"]]
    for i, w in enumerate(ws[:-1]):
        w["e"] = max(w["s"], min(w["e"], ws[i + 1]["s"]))
    groups = _groups(ws, st["per"], st["chars"])
    for gi, g in enumerate(groups):
        g_start = g[0]["s"]
        g_end = min(g[-1]["e"] + 0.3, groups[gi + 1][0]["s"] if gi + 1 < len(groups) else dur)
        if g_end - g_start < 0.03:
            continue
        texts = [re.sub(r"[.,;:]+$", "", x["w"]) for x in g]
        texts = [t.upper() if st["upper"] else t for t in texts]
        n_chars = sum(len(t) for t in texts) + len(texts) - 1
        size = st["size"] if n_chars <= st["chars"] else max(40, int(st["size"] * st["chars"] / n_chars))
        for i, w in enumerate(g):
            a = g_start if i == 0 else w["s"]
            b = g[i + 1]["s"] if i + 1 < len(g) else g_end
            if b - a < 0.02:
                continue
            parts = []
            for j, t in enumerate(texts):
                base = st["strong"] if _key(g[j]["w"]) in strong else st["base"]
                if j == i:        # the word being said: its own colour and a touch bigger
                    parts.append(f"{{\\c&H{st['active']}&\\fscx108\\fscy108}}{t}{{\\c&H{st['base']}&\\fscx100\\fscy100}}")
                else:
                    parts.append(f"{{\\c&H{base}&}}{t}")
            pop = "\\fscx82\\fscy82\\t(0,90,\\fscx100\\fscy100)" if i == 0 else ""
            ev.append(f"Dialogue: 2,{_ts(a)},{_ts(b)},Cap,,0,0,0,,{{\\an5\\pos({PW // 2},{cap_y})\\fs{size}{pop}}}"
                      + " ".join(parts))
    return head + "\n".join(ev) + "\n"


# ---------------------------------------------------------------- ffmpeg
def _shot_chain(i, layout, fg_h, W, H, zoom, focus, src_w, src_h) -> str:
    """The picture for one shot, from [t{i}] (already trimmed) to [v{i}]."""
    if layout == "smart" and focus is not None:
        ch = int(src_h / zoom / 2) * 2                       # a 9:16 window on the source, centred on the face
        cw = min(src_w, int(ch * 9 / 16 / 2) * 2)
        x = int(max(0, min(src_w - cw, focus * src_w - cw / 2)))
        return f"[t{i}]crop={cw}:{ch}:{x}:{(src_h - ch) // 2},scale={W}:{H},setsar=1[v{i}];"
    z = f",scale={int(W * zoom / 2) * 2}:{int(H * zoom / 2) * 2},crop={W}:{H}" if zoom != 1.0 else ""
    if layout == "crop":
        return f"[t{i}]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H}{z},setsar=1[v{i}];"
    return (f"[t{i}]split=2[bgi{i}][fgi{i}];"
            f"[bgi{i}]scale={W // 4}:{H // 4}:force_original_aspect_ratio=increase,crop={W // 4}:{H // 4},boxblur=8:2,"
            f"scale={W}:{H},eq=brightness=-0.16:saturation=1.2[bg{i}];"
            f"[fgi{i}]scale={W}:{fg_h}:force_original_aspect_ratio=increase,crop={W}:{fg_h}[fg{i}];"
            f"[bg{i}][fg{i}]overlay=0:(H-h)/2{z},setsar=1[v{i}];")


def render_clip(src: Path, out_dir: Path, name: str, clip: dict, words: list, src_w: int, src_h: int,
                src_offset: float = 0.0, has_audio=True) -> dict:
    """Writes <name>.mp4 and <name>.jpg into out_dir. `words` and the clip's start/end share one clock; the file
    `src` begins at `src_offset` on that clock."""
    s = config.get()
    out_dir.mkdir(parents=True, exist_ok=True)
    start, end = float(clip["start"]), float(clip["end"])
    H = 1280 if int(clip.get("quality") or s["quality"]) <= 720 else 1920
    W = H * 9 // 16
    q = H / PH
    layout, fg_h, hook_y, cap_y = geometry(src_w, src_h, clip.get("layout") or s["layout"])
    fg_px = int(round((860 if layout == "smart" else fg_h) * q / 2)) * 2      # smart falls back to the zoom look
    edits = clip.get("edits", True)
    env = None
    if edits and has_audio:
        from . import media
        env = envelope(media.pcm(src, max(0.0, start - src_offset), end - start))
    keep = plan_cuts(start, end, words, env, jump_cuts=edits)
    shots = plan_shots(keep, words) if edits else [(start, end, 1.0)]
    track = face_track(src, max(0.0, start - src_offset), end - start, src_w, src_h) if layout == "smart" else []
    shots4 = split_by_focus(shots, track, start) if track else [(a, b, z, None) for a, b, z in shots]
    shots = [x[:3] for x in shots4]
    dur = sum(b - a for a, b, _ in shots)
    full = layout in ("smart", "crop")                     # full-screen picture: the title steps aside after the opening
    cap_words = remap(words, shots) if clip.get("captions", True) else []
    ass = build_ass(cap_words, dur, clip.get("hook", ""), clip.get("style") or s["style"],
                    clip.get("highlight_words", []), hook_y, cap_y, s.get("watermark", ""),
                    hook_dur=4.5 if full else None)
    faces = 0

    n = len(shots)
    fc = f"[0:v]fps={FPS},split={n}" + "".join(f"[s{i}]" for i in range(n)) + ";"
    if has_audio:
        fc += f"[0:a]aresample=48000,asplit={n}" + "".join(f"[r{i}]" for i in range(n)) + ";"
    for i, (a, b, z, focus) in enumerate(shots4):
        ra, rb = a - start, b - start                      # the input is opened at `start`, so shot times count from 0
        faces += focus is not None
        fc += f"[s{i}]trim=start={ra:.3f}:end={rb:.3f},setpts=PTS-STARTPTS[t{i}];"
        fc += _shot_chain(i, layout, fg_px, W, H, z, focus, src_w, src_h)
        if has_audio:
            fc += (f"[r{i}]atrim=start={ra:.3f}:end={rb:.3f},asetpts=PTS-STARTPTS,"
                   f"afade=t=in:d=0.015,afade=t=out:st={max(0.0, rb - ra - 0.015):.3f}:d=0.015[a{i}];")
    if has_audio:
        fc += "".join(f"[v{i}][a{i}]" for i in range(n)) + f"concat=n={n}:v=1:a=1[cv][ca];"
        fc += "[ca]loudnorm=I=-14:TP=-1.5:LRA=11[aout];"
    else:
        fc += "".join(f"[v{i}]" for i in range(n)) + f"concat=n={n}:v=1:a=0[cv];"
    bar_h = max(6, int(12 * q) // 2 * 2)
    fc += (f"color=c={BAR_COLOR}:s={W}x{bar_h}:r={FPS}[bar];"
           f"[cv][bar]overlay=x='-W+W*t/{dur:.3f}':y=H-h:shortest=1[pv];"
           f"[pv]ass=c.ass:fontsdir=f[vout]")

    work = Path(tempfile.mkdtemp(prefix="cs"))      # short relative names: ffmpeg's filter syntax dislikes long paths
    try:
        (work / "c.ass").write_text(ass, encoding="utf-8")
        (work / "f").mkdir()
        for f in config.FONTS.glob("*.ttf"):
            shutil.copyfile(f, work / "f" / f.name)
        cmd = ["ffmpeg", "-y", "-v", "error", "-ss", f"{max(0.0, start - src_offset):.3f}", "-t", f"{end - start + 0.1:.3f}",
               "-i", str(Path(src).resolve()), "-filter_complex", fc, "-map", "[vout]"]
        if has_audio:
            cmd += ["-map", "[aout]", "-c:a", "aac", "-b:a", "160k", "-ar", "48000"]
        cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "21", "-pix_fmt", "yuv420p", "-r", str(FPS),
                "-movflags", "+faststart", "-t", f"{dur:.3f}", "o.mp4"]
        if shutil.which("nice"):                    # editing gives way to a live stream's recording and transcript
            cmd = ["nice", "-n", "10"] + cmd
        p = subprocess.run(cmd, cwd=work, capture_output=True, text=True, errors="replace", timeout=1500)
        if p.returncode != 0:
            raise RuntimeError("The clip could not be made: " + (p.stderr or "")[-500:])
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", f"{min(1.2, dur / 2):.2f}", "-i", "o.mp4", "-frames:v", "1",
                        "-q:v", "3", "o.jpg"], cwd=work, capture_output=True, timeout=120)
        shutil.move(str(work / "o.mp4"), str(out_dir / f"{name}.mp4"))
        if (work / "o.jpg").exists():
            shutil.move(str(work / "o.jpg"), str(out_dir / f"{name}.jpg"))
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return {"file": f"{name}.mp4", "cover": f"{name}.jpg", "layout": layout, "duration": round(dur, 1),
            "cuts": len(keep) - 1, "shots": n, "face_shots": faces}
