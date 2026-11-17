# Credits

ClipStorm is its own code. Its design was worked out by studying two MIT-licensed projects, and it owes them the ideas named here:

- **PeakCut** (https://github.com/haziqahmedqasmi/PeakCut): recording a live stream into a rolling buffer of short pieces, watching loudness and Twitch chat for sudden jumps, the "clip that" trigger, snapping clips to whole sentences, and word-by-word ASS captions.
- **OpenShorts** (https://github.com/mutonby/openshorts): ranking every candidate moment across the whole video instead of taking a fixed share per part, and treating everything read from a video as untrusted text.

- **AutoClip** (https://github.com/artbyjazi/autoclip, MIT): following the speaker's face for the vertical crop.
- **AI YouTube Shorts Generator** (https://github.com/Anil-matcha/AI-Youtube-Shorts-Generator, MIT): the ranked list of virality signals given to the AI.
- **Streamliner-AI** (https://github.com/oeilsimple/Streamliner-AI): weighting shouted "hype" words, and following a list of streamers that start recording by themselves.

The face detector `clipstorm/models/face_yunet.onnx` is YuNet from the OpenCV model zoo (MIT).

The caption fonts in `clipstorm/fonts` are Poppins and Anton, under the SIL Open Font License (see the `OFL-*.txt` files there).
