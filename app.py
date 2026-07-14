import io
import os
import tempfile
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

import lameenc
import numpy as np
import soundfile as sf
from flask import Flask, Response, jsonify, request, send_from_directory
from kokoro_onnx import Kokoro

app = Flask(__name__, static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # cap voice-sample uploads
kokoro = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")

_f5tts = None


def get_f5tts():
    global _f5tts
    if _f5tts is None:
        from f5_tts.api import F5TTS
        _f5tts = F5TTS()
    return _f5tts


VOICES = [
    ("af_heart", "Heart", "US female"),
    ("af_bella", "Bella", "US female"),
    ("af_sarah", "Sarah", "US female"),
    ("am_adam", "Adam", "US male"),
    ("am_michael", "Michael", "US male"),
    ("bf_emma", "Emma", "UK female"),
    ("bm_george", "George", "UK male"),
    ("bm_fable", "Fable", "UK male"),
]
VOICE_IDS = {v[0] for v in VOICES}


def synthesize(text, voice, speed):
    samples, sample_rate = kokoro.create(text, voice=voice, speed=speed, lang="en-us")
    return samples, sample_rate


def to_wav_bytes(samples, sample_rate):
    buf = io.BytesIO()
    sf.write(buf, samples, sample_rate, format="WAV")
    buf.seek(0)
    return buf.read()


def to_mp3_bytes(samples, sample_rate):
    pcm16 = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
    encoder = lameenc.Encoder()
    encoder.set_bit_rate(192)
    encoder.set_in_sample_rate(sample_rate)
    encoder.set_channels(1)
    encoder.set_quality(2)
    mp3_data = encoder.encode(pcm16.tobytes())
    mp3_data += encoder.flush()
    return bytes(mp3_data)


def get_params():
    data = request.get_json(force=True)
    text = (data.get("text") or "").strip()
    voice = data.get("voice") or "af_heart"
    speed = float(data.get("speed") or 1.0)

    if not text:
        raise ValueError("Text is required")
    if len(text) > 2000:
        raise ValueError("Text is too long (max 2000 characters)")
    if voice not in VOICE_IDS:
        raise ValueError("Unknown voice")
    speed = max(0.5, min(2.0, speed))

    return text, voice, speed


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/voices")
def api_voices():
    return jsonify(
        [{"id": vid, "name": name, "desc": desc} for vid, name, desc in VOICES]
    )


@app.route("/api/preview", methods=["POST"])
def api_preview():
    try:
        text, voice, speed = get_params()
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    samples, sample_rate = synthesize(text, voice, speed)
    wav_bytes = to_wav_bytes(samples, sample_rate)
    return Response(wav_bytes, mimetype="audio/wav")


@app.route("/api/download", methods=["POST"])
def api_download():
    try:
        text, voice, speed = get_params()
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    samples, sample_rate = synthesize(text, voice, speed)
    mp3_bytes = to_mp3_bytes(samples, sample_rate)
    return Response(
        mp3_bytes,
        mimetype="audio/mpeg",
        headers={"Content-Disposition": f'attachment; filename="{voice}.mp3"'},
    )


def _clone_infer(audio_file, ref_text, gen_text, speed):
    suffix = os.path.splitext(audio_file.filename or "ref.wav")[1] or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        audio_file.save(tmp.name)
        tmp_path = tmp.name
    try:
        import librosa

        # F5-TTS silently clips reference audio to ~12 s; a longer clip then no
        # longer matches its transcript and the unspoken words bleed into the
        # generated speech. Reject early instead.
        try:
            ref_duration = librosa.get_duration(path=tmp_path)
        except Exception:
            ref_duration = None
        if ref_duration is not None and ref_duration > 15.0:
            raise ValueError(
                "reference audio too long "
                f"({ref_duration:.0f}s) — use a 5-12 second clip"
            )

        f5 = get_f5tts()
        # Always infer at speed 1.0: F5's speed parameter skews its duration
        # estimate, and at slow speeds the model fills the surplus time with
        # echoes of the reference audio. Stretch the output afterwards instead.
        wav, sr, _ = f5.infer(
            ref_file=tmp_path,
            ref_text=ref_text,
            gen_text=gen_text,
            speed=1.0,
            show_info=lambda *a, **kw: None,
        )
        if abs(speed - 1.0) > 1e-3:
            wav = librosa.effects.time_stretch(np.asarray(wav), rate=speed)
        return wav, sr
    finally:
        os.unlink(tmp_path)


@app.route("/api/clone", methods=["POST"])
def api_clone():
    audio_file = request.files.get("audio")
    ref_text = (request.form.get("ref_text") or "").strip()
    gen_text = (request.form.get("text") or "").strip()
    speed = float(request.form.get("speed") or 1.0)
    speed = max(0.5, min(2.0, speed))

    if not audio_file:
        return jsonify({"error": "voice sample required"}), 400
    if not ref_text:
        return jsonify({"error": "reference transcript required"}), 400
    if not gen_text:
        return jsonify({"error": "text to generate required"}), 400
    if len(gen_text) > 2000:
        return jsonify({"error": "text too long (max 2000 characters)"}), 400

    try:
        wav, sr = _clone_infer(audio_file, ref_text, gen_text, speed)
        return Response(to_wav_bytes(wav, sr), mimetype="audio/wav")
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/clone/download", methods=["POST"])
def api_clone_download():
    audio_file = request.files.get("audio")
    ref_text = (request.form.get("ref_text") or "").strip()
    gen_text = (request.form.get("text") or "").strip()
    speed = float(request.form.get("speed") or 1.0)
    speed = max(0.5, min(2.0, speed))

    if not audio_file:
        return jsonify({"error": "voice sample required"}), 400
    if not ref_text:
        return jsonify({"error": "reference transcript required"}), 400
    if not gen_text:
        return jsonify({"error": "text to generate required"}), 400

    try:
        wav, sr = _clone_infer(audio_file, ref_text, gen_text, speed)
        return Response(
            to_mp3_bytes(wav, sr),
            mimetype="audio/mpeg",
            headers={"Content-Disposition": 'attachment; filename="clone.mp3"'},
        )
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5050, debug=False)
