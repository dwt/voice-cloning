#! /usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "transformers>=5.0.0rc1",
#     "mlx-audio>=0.3.0",
#     "typer",
#     "numpy",
#     "soundfile",
# ]
# ///
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Annotated

import numpy as np
import soundfile as sf
import typer
from mlx_audio.tts.utils import load_model


def eprint(*args, **kwargs) -> None:
    """Print to stderr (all user-facing output should go to stderr)."""
    print(*args, file=sys.stderr, **kwargs)


MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"


def get_unique_filename(base_path: Path) -> Path:
    """Return a unique filename, adding -2, -3, etc. if the file already exists."""
    if not base_path.exists():
        return base_path

    stem = base_path.stem
    suffix = base_path.suffix
    parent = base_path.parent

    counter = 2
    while True:
        new_path = parent / f"{stem}-{counter}{suffix}"
        if not new_path.exists():
            return new_path
        counter += 1


def resolve_text(text: str | None) -> str:
    if text is None:
        if sys.stdin.isatty():
            raise typer.BadParameter(
                "No text provided. Pass text as an argument or pipe it via stdin."
            )
        try:
            text = sys.stdin.read().strip()
        except KeyboardInterrupt:
            raise typer.Abort()

    text = (text or "").strip()
    if not text:
        raise typer.BadParameter("Text cannot be empty.")
    return text


def generate_audio(
    model,
    *,
    text: str,
    language: str,
    instruct: str | None,
    verbose: bool,
):
    if verbose:
        eprint(f"Generating audio for: {text[:50]}{'...' if len(text) > 50 else ''}")

    results = list(
        model.generate_voice_design(
            text=text,
            language=language,
            verbose=verbose,
            instruct=instruct or "",
        )
    )
    if not results:
        raise typer.ClickException("Audio generation failed - no results returned")
    return results[0].audio


def write_or_play(
    *,
    audio,
    sample_rate: int,
    output: str | None,
    play: bool,
    verbose: bool,
) -> None:
    if play:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_file:
            output_path = Path(temp_file.name)

            sf.write(str(output_path), np.array(audio), sample_rate)

            try:
                subprocess.run(["which", "afplay"], check=True, capture_output=True)
                subprocess.run(["afplay", str(output_path)], check=True)
                if verbose:
                    eprint("Audio played successfully")
            except subprocess.CalledProcessError:
                eprint("Failed to play audio - afplay not available or playback failed")
        return

    if output is None:
        raise typer.BadParameter(
            "No output filename provided. Use --play or pass --output."
        )
    output_path = Path(output)
    if output == "output.wav":
        output_path = get_unique_filename(output_path)

    sf.write(str(output_path), np.array(audio), sample_rate)
    if verbose:
        eprint(f"Audio saved to: {output_path}")


def main(
    text: Annotated[
        str | None,
        typer.Argument(help="Text to speak. If omitted, read from stdin."),
    ] = None,
    output: Annotated[
        str | None,
        typer.Option(
            "-o", "--output", help="Output filename (default: none; implies --play)"
        ),
    ] = None,
    language: Annotated[
        str,
        typer.Option("-l", "--language", help="Language for TTS (default: English)"),
    ] = "English",
    instruct: Annotated[
        str | None,
        typer.Option(
            "-i", "--instruct", help="Voice instruction (e.g., 'deep low voice')"
        ),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option("-v", "--verbose", help="Enable verbose output"),
    ] = False,
    play: Annotated[
        bool,
        typer.Option("-p", "--play", help="Auto-play generated audio (no file output)"),
    ] = False,
):
    """Generate audio using Qwen3-TTS and MLX Audio.

    Examples:
      q3_tts.py "say this text out loud"
      q3_tts.py -o saved.wav "hello world"
      q3_tts.py -l Chinese "你好世界"
      q3_tts.py -i "deep low voice" "hello"
      q3_tts.py -p "play this text immediately"
      echo "piped text" | q3_tts.py
    """
    # Default behavior: if neither --play nor --output is provided, play immediately.
    if not play and output is None:
        play = True

    if play and output is not None:
        raise typer.BadParameter("--play and --output cannot be used together")

    text = resolve_text(text)

    if verbose:
        eprint("Loading model...")
    model = load_model(MODEL_ID)

    audio = generate_audio(
        model,
        text=text,
        language=language,
        instruct=instruct,
        verbose=verbose,
    )
    write_or_play(
        audio=audio,
        sample_rate=model.sample_rate,
        output=output,
        play=play,
        verbose=verbose,
    )


if __name__ == "__main__":
    typer.run(main)
