"""Pre-generate a short sample clip per voice into static/samples/."""
import os

import lameenc
import numpy as np
from kokoro_onnx import Kokoro

VOICES = [
    ("af_heart", "Heart"),
    ("af_bella", "Bella"),
    ("af_sarah", "Sarah"),
    ("am_adam", "Adam"),
    ("am_michael", "Michael"),
    ("bf_emma", "Emma"),
    ("bm_george", "George"),
    ("bm_fable", "Fable"),
]

SAMPLE_TEXT = "Hi, I'm {name}. This is how I sound."

os.makedirs("static/samples", exist_ok=True)
kokoro = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")

for voice_id, name in VOICES:
    out_path = f"static/samples/{voice_id}.mp3"
    if os.path.exists(out_path):
        print(f"skip {voice_id} (exists)")
        continue
    samples, sr = kokoro.create(
        SAMPLE_TEXT.format(name=name), voice=voice_id, speed=1.0, lang="en-us"
    )
    pcm16 = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
    enc = lameenc.Encoder()
    enc.set_bit_rate(128)
    enc.set_in_sample_rate(sr)
    enc.set_channels(1)
    enc.set_quality(2)
    data = enc.encode(pcm16.tobytes()) + enc.flush()
    with open(out_path, "wb") as f:
        f.write(bytes(data))
    print(f"wrote {out_path} ({len(samples) / sr:.2f}s)")

print("done")
