#! /usr/bin/env -S uv run --script
#
# Get current mlx-audio commit
# git ls-remote https://github.com/Blaizzy/mlx-audio.git refs/heads/main | awk '{print $1}'
#
# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "mlx-audio@git+https://github.com/Blaizzy/mlx-audio.git@1f8dc29145e68f2b7a7bbd29d6e51a3cb3105503",
#     "transformers",
#     "typer",
#     "numpy",
#     "soundfile",
#     "scipy",
# ]
# ///

import sys

from mlx_audio.stt.utils import load


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: voxtral-transcribe.py <path-to-audio.wav>")
        raise SystemExit(1)

    audio_path = sys.argv[1]
    try:
        # model = load("mlx-community/Voxtral-Mini-4B-Realtime-2602-fp16")
        model = load("mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit")
    except Exception as e:
        print(f"Failed to load model: {e}")
        print("Tip: try transformers 4.x or a different mlx-community STT model.")
        raise

    # Streaming transcription
    for chunk in model.generate(audio_path, stream=True):
        print(chunk, end="", flush=True)


if __name__ == "__main__":
    main()
