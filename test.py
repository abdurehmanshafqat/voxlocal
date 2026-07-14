import sys
from kokoro_onnx import Kokoro
import soundfile as sf

VOICES = {
    "1": ("af_heart", "US female - Heart"),
    "2": ("af_bella", "US female - Bella"),
    "3": ("af_sarah", "US female - Sarah"),
    "4": ("am_adam", "US male - Adam"),
    "5": ("am_michael", "US male - Michael"),
    "6": ("bf_emma", "UK female - Emma"),
    "7": ("bm_george", "UK male - George"),
    "8": ("bm_fable", "UK male - Fable"),
}

TEXT = "Hi Abdur, this is kokoro running locally on your device. You can use kokoro offline and integrate with apps of your choice."

def parse_choice(argv):
    for arg in argv:
        if arg.startswith("--") and arg[2:] in VOICES:
            return arg[2:]
    return "1"

choice = parse_choice(sys.argv[1:])
voice, label = VOICES[choice]

kokoro = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")

samples, sample_rate = kokoro.create(
    TEXT,
    voice=voice,
    speed=1.0,
    lang="en-us",
)

out_file = f"output_{choice}_{voice}.wav"
sf.write(out_file, samples, sample_rate)
print(f"[{choice}] {label} ({voice}) -> {out_file} ({len(samples) / sample_rate:.2f}s)")
