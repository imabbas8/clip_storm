# ClipStorm ⚡

**Paste a link. Get viral shorts.**

Give ClipStorm any video or live-stream link (YouTube, Twitch, Kick and the hundreds of sites yt-dlp knows) or upload
a file. It finds the moments a stranger would stop scrolling for and turns each one into an edited 9:16 short:
dead air cut out, punch-in zooms, word-by-word captions, a hook title, a progress bar and levelled sound.

A **live stream is watched non-stop**. A panel of AI agents follows what is said, how loud it gets and how busy chat
is, and a new short appears every time something worth clipping happens, for as long as the stream runs.

## Start

You need Python 3.10+ and `ffmpeg` (`sudo apt install ffmpeg`, Mac: `brew install ffmpeg`).

```bash
./run.sh
```

The first run sets itself up. Then open **http://127.0.0.1:8787**.

1. **Settings → AI**: choose a service, paste your key, press **Connect and test**. A green "Success" means the key
   and the model really work (one small request is sent). It is saved at the same moment.
2. Paste a link on the front page and press **Make shorts**.

That is all. Without a key ClipStorm still runs, but it picks moments by loudness only and writes no hooks.

## What it does with a link

| The link is | What happens |
|---|---|
| A finished video | Downloaded, transcribed, the best moments ranked across the whole video, each one edited into a short |
| A stream that is live now | Recorded into a rolling buffer and clipped while it runs, until it ends or you press Stop |
| A channel that is not live yet | ClipStorm waits and starts by itself when the stream begins |

Several links can run at the same time.

## How a moment gets picked

- **Viral score (0-100).** Every candidate gets a score for how likely a stranger watches to the end. Only moments at
  or above your threshold become shorts (Settings → Live). A quiet stream gives no shorts, not weak ones.
- **A panel of agents.** Up to four agents judge the same stretch side by side, each with one thing in mind:
  **Hype** (energy, reactions), **Story** (opinions, facts, payoffs), **Hook** (the first three seconds) and **Funny**.
  When several pick the same moment on their own, its score goes up, and the short shows who spotted it.
- **Three triggers on a live stream.** A regular check every few minutes while people talk; an immediate check when
  loudness or Twitch chat suddenly jumps; and an immediate check when someone says "clip that".
- **Clip now.** A button on every live stream cuts the last 45 seconds on the spot.

## The edit

| Edit | What it is |
|---|---|
| Face tracking | The "Smart" picture follows the speaker's face and re-frames at every camera cut |
| Jump cuts | Pauses that are also quiet are cut out. A wordless but loud moment (gameplay, a crowd) stays |
| Punch-in zoom | The picture moves a little closer or wider at each sentence, so it never sits still |
| Captions | Every word as it is said, the current word in colour, the punch words highlighted. Three styles |
| Hook | A short title on top, written by the AI, true to what happens |
| Progress bar | Along the bottom, so people see how little is left |
| Sound | Levelled to the loudness the platforms expect |
| Watermark | Your channel name, small, on every short |
| Roman captions | Hindi, Urdu and other scripts can be written in English letters ("yaar kya baat hai") |

Turn any of it off in Settings → Shorts.

## Library, Channels and Edit

- **Library**: every short you have made, sorted by viral score or newest.
- **Channels**: add the streamers you follow. Each is checked every two minutes and recorded by itself when it goes live (up to 3 at once).
- **Edit**: change any short's hook, title, start and end, picture or caption style, and make it again.

## AI services

Claude (default), Gemini, OpenAI, Groq, OpenRouter, Grok, a local model through Ollama, or any OpenAI-compatible
address. **Find models** lists what your key can use. You pay the service directly.

Speech to text runs on your own machine by default (free and private). On a small computer, switch to Groq Whisper
in Settings → Speech: it is far faster and has a free tier.

## Posting to YouTube (optional)

Settings → YouTube walks through it: create one OAuth client in Google Cloud, paste its id and secret, press
**Save and connect**, then **Test**. Shorts can be posted with one click, or by themselves as they are made. New
posts are **private** unless you change that.

## Good to know

- **YouTube on a server.** YouTube often refuses data-centre addresses ("sign in to confirm you're not a bot"). In
  Settings → Downloads choose the browser on that machine where you are signed in to YouTube, or give a cookies.txt.
  Twitch, Kick and uploads do not need this.
- **Other people on the network.** ClipStorm listens only on this computer. To open it to others set
  `CLIPSTORM_HOST=0.0.0.0` and a password with `CLIPSTORM_PASSWORD=...`.
- **Only clip what you may.** Use your own content, or content you have permission to clip and post.
- Everything is kept in `data/` next to this file: settings, keys, videos and shorts.

| Setting (environment) | Default | For |
|---|---|---|
| `CLIPSTORM_PORT` | 8787 | The port |
| `CLIPSTORM_HOST` | 127.0.0.1 | Who can reach it |
| `CLIPSTORM_PASSWORD` | none | Ask for a password |
| `CLIPSTORM_DATA` | ./data | Where everything is kept |
| `CLIPSTORM_AI_PROVIDER`, `CLIPSTORM_AI_KEY`, `CLIPSTORM_AI_MODEL` | none | The AI, without using the page |

## For developers

| File | What it does |
|---|---|
| `clipstorm/server.py` | The web routes |
| `clipstorm/live.py` | Live recording, triggers and the watch loop |
| `clipstorm/scout.py` | The prompts, the agent panel, snapping to sentences, cleaning AI text |
| `clipstorm/render.py` | The edit: cuts, zooms, captions, hook, ffmpeg |
| `clipstorm/ai.py` | The only file that talks to an AI service |
| `clipstorm/asr.py` | Speech to text |
| `clipstorm/jobs.py`, `media.py`, `youtube.py`, `config.py` | Job store and queue, downloads, posting, settings |
| `web/index.html` | The whole page, no build step |

Tests need no key and no network: `.venv/bin/python -m unittest discover -s tests`

Credits and licences (PeakCut, OpenShorts, AutoClip, Streamliner-AI, AI YouTube Shorts Generator): [NOTICE.md](NOTICE.md).
