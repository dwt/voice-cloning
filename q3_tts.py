#! /usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "transformers>=5.0.0rc1",
#     "mlx-audio>=0.3.0",
#     "typer",
#     "numpy",
#     "soundfile",
#     "scipy",
# ]
# ///
#
# Uses https://qwen3-tts.com to locally generate emotional shaped speech or clone a voice from 3-4 seconds of audio.
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Annotated

import numpy as np
import soundfile as sf
import typer
from mlx_audio.tts.utils import load_model
from scipy.signal import resample_poly

try:
    import mlx.core as mx
except (
    Exception
):  # mlx might not be importable in some environments until deps are installed
    mx = None


def eprint(*args, **kwargs) -> None:
    """Print to stderr (all user-facing output should go to stderr)."""
    print(*args, file=sys.stderr, **kwargs)


MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
CLONE_MODEL_ID = "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16"


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


def generate_audio_clone(
    model,
    *,
    text: str,
    ref_audio: str,
    ref_text: str,
    verbose: bool,
):
    if verbose:
        eprint(
            f"Generating cloned audio for: {text[:50]}{'...' if len(text) > 50 else ''}"
        )
        eprint(f"Using reference audio: {ref_audio}")

    # mlx-audio speech tokenizer path expects an MLX array (mx.array), not a numpy ndarray.
    if mx is None:
        raise typer.ClickException(
            "MLX is not available (failed to import mlx.core). Cannot run voice cloning."
        )

    try:
        ref_waveform, ref_sr = sf.read(ref_audio, always_2d=False)
    except FileNotFoundError:
        raise typer.BadParameter(f"Reference audio file not found: {ref_audio}")

    # Ensure float32 for consistent downstream behavior
    ref_waveform = np.asarray(ref_waveform, dtype=np.float32)

    # If multi-channel, downmix to mono (shape: [T])
    if ref_waveform.ndim > 1:
        ref_waveform = ref_waveform.mean(axis=1)

    target_sr = getattr(model, "sample_rate", None)
    if verbose:
        eprint(f"Reference audio sample rate: {ref_sr} Hz")
        if target_sr is not None:
            eprint(f"Model sample rate: {target_sr} Hz")

    # Resample reference audio to the model's expected sample rate (critical for cloning quality)
    if target_sr is not None and ref_sr != target_sr:
        if verbose:
            eprint(f"Resampling reference audio {ref_sr} Hz -> {target_sr} Hz")
        # Use rational polyphase resampling for good quality and speed
        g = np.gcd(int(ref_sr), int(target_sr))
        up = int(target_sr // g)
        down = int(ref_sr // g)
        ref_waveform = resample_poly(ref_waveform, up=up, down=down).astype(
            np.float32, copy=False
        )

    # Convert to MLX array
    ref_waveform_mx = mx.array(ref_waveform)

    results = list(
        model.generate(
            text=text,
            ref_audio=ref_waveform_mx,
            ref_text=ref_text,
        )
    )
    if not results:
        raise typer.Abort("Audio generation failed - no results returned")
    return results[0].audio


def resolve_ref_text(ref_text: str | None, ref_text_file: str | None) -> str:
    if ref_text is not None and ref_text_file is not None:
        raise typer.BadParameter(
            "--ref-text and --ref-text-file cannot be used together"
        )

    if ref_text is not None:
        ref_text = (ref_text or "").strip()
        if not ref_text:
            raise typer.BadParameter("Reference text cannot be empty.")
        return ref_text

    if ref_text_file is None:
        raise typer.BadParameter(
            "Missing reference text. Provide --ref-text or --ref-text-file."
        )

    try:
        ref_text = Path(ref_text_file).read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        raise typer.BadParameter(f"Reference text file not found: {ref_text_file}")
    if not ref_text:
        raise typer.BadParameter("Reference text file is empty.")
    return ref_text


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
    clone: Annotated[
        bool,
        typer.Option(
            "--clone",
            help="Enable voice cloning using a reference audio + reference text",
        ),
    ] = False,
    ref_audio: Annotated[
        str,
        typer.Option(
            "--ref-audio",
            help="Reference WAV/Audio file used for cloning (default: martin.wav)",
        ),
    ] = "martin.wav",
    ref_text: Annotated[
        str | None,
        typer.Option(
            "--ref-text",
            help="Reference transcript for the ref audio (use --ref-text-file alternatively)",
        ),
    ] = None,
    ref_text_file: Annotated[
        str,
        typer.Option(
            "--ref-text-file",
            help="Path to file containing reference transcript (default: martin.txt)",
        ),
    ] = "martin.txt",
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

      # Voice cloning (defaults to martin.wav + martin.txt in current directory)
      q3_tts.py --clone "Hallo, das ist ein Klon meiner Stimme."

      # Voice cloning with custom reference
      q3_tts.py --clone --ref-audio sample.wav --ref-text "This is what my voice sounds like." "Hello"
    """
    # Default behavior: if neither --play nor --output is provided, play immediately.
    if not play and output is None:
        play = True

    if play and output is not None:
        raise typer.BadParameter("--play and --output cannot be used together")

    text = resolve_text(text)

    if clone:
        if verbose:
            eprint(f"Loading clone model: {CLONE_MODEL_ID}")
            eprint(f"Clone prompt text: {text!r}")
        model = load_model(CLONE_MODEL_ID)

        ref_text_resolved = resolve_ref_text(ref_text, ref_text_file)
        if verbose:
            eprint(f"Reference transcript: {ref_text_resolved!r}")

        audio = generate_audio_clone(
            model,
            text=text,
            ref_audio=ref_audio,
            ref_text=ref_text_resolved,
            verbose=verbose,
        )
    else:
        if verbose:
            eprint(f"Loading model: {MODEL_ID}")
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
    app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)
    app.command()(main)
    app()

    # typer.run(main)
