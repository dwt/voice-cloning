# q3-tts

A command-line tool for text-to-speech synthesis using Qwen3-TTS and MLX Audio, with voice cloning support. Optimized for Apple Silicon.

## Features

- Text-to-speech generation with Qwen3-TTS-12Hz-1.7B
- Voice cloning from reference audio samples
- Voice design via natural language instructions
- Multi-language support
- Stdin pipe support for scripting
- Direct audio playback or file output

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.14+
- [uv](https://docs.astral.sh/uv/) (recommended)

## Installation

### Option A: Run Directly (no install)

The script uses [PEP 723](https://peps.python.org/pep-0723/) inline metadata, so uv handles dependencies automatically:

```bash
chmod +x q3_tts.py
./q3_tts.py "Hello, world!"
```

### Option B: Install as Package

For repeated use or if you prefer a cleaner command:

```bash
uv sync
uv run q3-tts "Hello, world!"
```

Both approaches are equivalent - choose whichever fits your workflow.

## Usage

### Basic Text-to-Speech

```bash
# Play immediately (default)
./q3_tts.py "Hello, this is a test."

# Save to file
./q3_tts.py -o output.wav "Hello, this is a test."

# Pipe from stdin
echo "Piped text" | ./q3_tts.py
```

### Voice Design

Control the voice characteristics with natural language:

```bash
./q3_tts.py -i "deep low voice" "This sounds dramatic."
./q3_tts.py -i "cheerful young female voice" "Great news everyone!"
```

### Voice Cloning

Clone a voice from a reference audio sample:

```bash
# Using default reference files (martin.wav + martin.txt)
./q3_tts.py --clone "This is my cloned voice speaking."

# Custom reference
./q3_tts.py --clone \
  --ref-audio sample.wav \
  --ref-text "Transcript of the sample audio." \
  "New text in the cloned voice."
```

### Multi-language Support

```bash
./q3_tts.py -l Chinese "你好世界"
./q3_tts.py -l German "Guten Tag"
```

## Options

| Option | Short | Description |
|--------|-------|-------------|
| `--output` | `-o` | Output WAV file path |
| `--play` | `-p` | Play audio immediately (default if no output specified) |
| `--language` | `-l` | Language for synthesis (default: English) |
| `--instruct` | `-i` | Voice design instruction |
| `--clone` | | Enable voice cloning mode |
| `--ref-audio` | | Reference audio file for cloning (default: martin.wav) |
| `--ref-text` | | Reference transcript text |
| `--ref-text-file` | | Path to reference transcript file (default: martin.txt) |
| `--verbose` | `-v` | Enable verbose output |

## Models

| Mode | Model |
|------|-------|
| Voice Design | `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` |
| Voice Cloning | `mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16` |

Models are downloaded automatically on first use via Hugging Face Hub.

## How It Works

1. **Voice Design Mode** (default): Uses the 1.7B parameter model to generate speech with controllable characteristics. Pass natural language instructions with `-i` to shape the voice.

2. **Voice Cloning Mode** (`--clone`): Uses the 0.6B base model with a reference audio sample to clone a specific voice. Requires a reference audio file and its transcript.

## Development

```bash
# Install with dev dependencies
uv sync

# Run linter
uv run ruff check .

# Run type checker
uv run mypy q3_tts.py
```

## License

MIT
