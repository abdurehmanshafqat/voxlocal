# Kokoro Studio — Local TTS & Voice Cloning

A small web app for playing with modern speech synthesis, entirely on your own machine. No API keys, no cloud, no audio leaving your computer.

Two things live here:

- **Studio** — text-to-speech powered by [Kokoro](https://github.com/thewh1teagle/kokoro-onnx) (an 82M-parameter model running on ONNX Runtime). Eight voices, adjustable speed, instant WAV preview with a waveform player, MP3 export.
- **Voice Clone** — zero-shot voice cloning with [F5-TTS](https://github.com/SWivid/F5-TTS). Give it a 5–12 second sample of a voice plus its transcript, and it speaks any text you type in that voice. You can upload a clip or record one directly in the browser — the recorder shows a live level meter, paces you through a phonetically rich training passage, and converts the take to WAV client-side before upload.

I built this while exploring TTS and voice cloning as a learning project. LLM integration is next (see roadmap).

## How it works

```
static/index.html  ── single-file frontend (vanilla JS, Web Audio API)
        │
        ▼  fetch
app.py (Flask, 127.0.0.1:5050)
        ├── /api/voices, /api/preview, /api/download   → Kokoro ONNX (CPU)
        └── /api/clone, /api/clone/download            → F5-TTS (CUDA, lazy-loaded)
```

- **Frontend**: one HTML file, no build step, no frameworks. Custom waveform player, MediaRecorder-based voice recorder with an in-browser PCM16 WAV encoder, dark/light theme from `prefers-color-scheme`.
- **Backend**: Flask serves the page and two small API surfaces. Kokoro loads at startup (fast); F5-TTS loads lazily on the first clone request and downloads its checkpoint on first use.
- **Audio out**: WAV for previews (via soundfile), MP3 for downloads (via lameenc).

## Setup

Requires Python 3.10+ and, for voice cloning, an NVIDIA GPU (a few GB of VRAM is enough — CPU fallback works too, just slower).

```bash
git clone https://github.com/abdurehmanshafqat/voxlocal.git
cd voxlocal
./setup.sh
```

This installs FFmpeg, creates a virtualenv, installs Python deps, downloads the Kokoro model files, and pre-generates the voice-sample clips. No model weights or dependencies are committed to the repo — everything is fetched at setup time.

Run it:

```bash
source venv/bin/activate
python app.py
# → http://127.0.0.1:5050
```

## Usage

**Studio**: type text, click a voice card (it plays a sample), set speed, hit Generate. Download MP3 if you like the result.

**Voice Clone**:
1. Pick *Record* and read the training passage aloud (5–12 s is the sweet spot; F5-TTS only conditions on ~12 s of reference audio, so recordings are capped there), or pick *Upload* and drop in an existing clip with its exact transcript.
2. Listen back to your recording before committing — cloning is the slow part.
3. Type what the cloned voice should say and hit *Generate Clone*.

The first clone request downloads the F5-TTS checkpoint and warms up the model (~15–30 s). After that, generation takes a few seconds per sentence.

## GPU notes

- Tested on an RTX 4050 Laptop GPU (6 GB VRAM) — F5-TTS inference fits comfortably.
- Kokoro runs on CPU via ONNX Runtime; it's an 82M model, so synthesis is near-instant anyway.
- F5-TTS should also fall back to CPU if no CUDA device is present, just much slower.

## Roadmap

- [ ] LLM integration — a conversational mode where a local LLM writes the words and Kokoro/F5 speaks them
- [ ] Streaming synthesis for long texts
- [ ] Save/manage cloned voice profiles instead of re-uploading references

## Acknowledgements

- [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) and [thewh1teagle/kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx)
- [SWivid/F5-TTS](https://github.com/SWivid/F5-TTS)
