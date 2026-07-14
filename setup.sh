#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

echo "==> Installing ffmpeg (needed by PyTorch's audio loader for voice cloning)"
sudo apt-get update -qq && sudo apt-get install -y ffmpeg

echo "==> Creating virtualenv"
python3 -m venv venv
source venv/bin/activate

echo "==> Installing Python dependencies"
pip install --upgrade pip
pip install -r requirements.txt

echo "==> Downloading Kokoro model files"
wget -nc https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
wget -nc https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin

echo "==> Pre-generating voice sample clips"
python generate_samples.py

echo "==> Done. Run: source venv/bin/activate && python app.py"
echo "    Then open http://127.0.0.1:5050"
echo "    (F5-TTS checkpoint downloads automatically on first clone request)"
