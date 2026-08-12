# Audio Clone Studio

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-111827)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-macOS%20%7C%20Linux%20%7C%20Windows-0F766E)](#installation)

Audio Clone Studio is a local-first voice cloning and speech production workspace built on top of the F5-TTS stack. It combines fast reference-based synthesis, project-aware asset management, trained voice checkpoints, reusable voice profiles, batch rendering, aligned speech editing, diagnostics, and a polished browser studio in one repository.

[Quick start](#quick-start) |
[Studio workflow](#studio-workflow) |
[Development](#development-commands) |
[F5-TTS upstream](https://github.com/SWivid/F5-TTS)

## Project status and attribution

This is my experimental product layer around the open-source F5-TTS codebase,
not an independently trained speech model. The package name, model code, and
core inference entry points remain those of F5-TTS. My work in this repository
focuses on the local studio, project and asset organization, voice profiles,
editing workflows, diagnostics, service access, and cross-platform operation.

F5-TTS is credited in the package metadata and remains the upstream reference
for the model architecture and original research. NVIDIA BigVGAN is included
as a pinned Git submodule for supported vocoder paths.

This repository is designed for practical production work:

- clone and render from curated reference clips
- manage reusable voice profiles and style prompts
- switch between shipped base models and local trained checkpoints
- queue preview and final renders safely on shared hardware
- perform transcript-aligned speech edits instead of destructive full rerenders
- review takes with diagnostics, comparison views, and export bundles
- serve the studio locally or mount it behind a small FastAPI service

## Highlights

- **Voice Studio**: project-based browser workspace for references, styles, profiles, takes, and exports
- **Trained Voice Workflow**: route renders through a local finetuned checkpoint with profile-based reference selection
- **Voice Profiles**: group multiple clean references so the app can choose the best clip per render
- **Editing Tools**: alignment-first replace, insert, and delete actions with localized rerendering
- **Batch and Queueing**: render long scripts one job at a time on constrained machines
- **Diagnostics**: transcript drift checks, silence checks, reference scoring, and take review helpers
- **Apple Silicon Support**: Apple-friendly transcription extras and an optional MLX backend for shipped base-model inference
- **API Mode**: optional FastAPI server with mounted studio UI and local REST endpoints

## What’s Included

### User-Facing Apps

- `f5-tts_voice-studio`
  The main browser studio for cloning, rendering, editing, and export.

- `f5-tts_infer-gradio`
  The broader inference demo with multi-style and multi-speaker utilities.

- `f5-tts_studio-server`
  FastAPI server that exposes the studio UI under `/app` and API routes under `/api/v1`.

### Production Workflows

- short-form and long-form voice cloning
- trained checkpoint auditioning
- project-scoped pronunciation dictionaries
- style prompt reuse
- batch render pipelines
- take comparison and export packaging
- transcript-aligned speech editing

## Installation

### 1. Create an environment

```bash
python -m venv .venv
source .venv/bin/activate
```

If you prefer Conda:

```bash
conda create -n audio-clone python=3.11
conda activate audio-clone
```

### 2. Install FFmpeg

FFmpeg is required for reference preparation, audio decoding, and several export paths.

macOS:

```bash
brew install ffmpeg
```

Ubuntu / Debian:

```bash
sudo apt-get update
sudo apt-get install ffmpeg
```

### 3. Install PyTorch for your machine

Apple Silicon:

```bash
pip install torch torchaudio
```

NVIDIA:

```bash
pip install torch==2.8.0+cu128 torchaudio==2.8.0+cu128 --extra-index-url https://download.pytorch.org/whl/cu128
```

### 4. Install the project

Studio install:

```bash
pip install -e ".[studio]"
```

Apple-Silicon studio install:

```bash
pip install -e ".[studio,apple_audio]"
```

Training extras:

```bash
pip install -e ".[train]"
```

Development extras:

```bash
pip install -e ".[dev]"
```

## Quick Start

### Launch the main studio

```bash
f5-tts_voice-studio
```

### Launch the API server

```bash
f5-tts_studio-server
```

### Launch the broader inference demo

```bash
f5-tts_infer-gradio
```

### Command-line generation

```bash
f5-tts_infer-cli \
  --model F5TTS_v1_Base \
  --ref_audio "ref_audio.wav" \
  --ref_text "Reference transcript." \
  --gen_text "Text to render."
```

## Studio Workflow

### 1. Create a project

Start with a project in the studio and keep related references, styles, renders, and exports together.

### 2. Save clean references

Add short, clean clips with confirmed transcripts. The studio can analyze and score them for later reuse.

### 3. Build a voice profile

Group several strong references from the same speaker. The studio can then pick the most suitable clip automatically for each render.

### 4. Add style prompts

Save alternate delivery examples for pacing, tone, or mood so they can be reused across scripts.

### 5. Render

Use:

- **Quick Preview** for faster iteration
- **Final Render** for higher-quality delivery
- **Voices** for direct text-to-speech with a saved profile or single reference
- **Trained Voice** for checkpoint-based rendering

### 6. Review and export

Compare takes, run diagnostics, and export WAV, MP3, transcripts, or packaged bundles.

## Apple Silicon Notes

On Apple Silicon, the studio supports:

- `mlx-whisper` for local transcription
- optional `Apple MLX F5-TTS` base-model inference from the Runtime panel
- automatic fallback to PyTorch when a local finetuned checkpoint is selected

Recommended install:

```bash
pip install -e ".[studio,apple_audio]"
```

## Local Data Locations

The studio stores project data outside the repository.

macOS:

- support data: `~/Library/Application Support/F5-TTS-Studio`
- cache: `~/Library/Caches/F5-TTS-Studio`

This keeps references, profiles, takes, exports, and the local library out of the working tree.

## Sharing

For temporary remote access, the repository includes helper scripts for local tunnel workflows:

```bash
./scripts/voice_studio_quick_tunnel.sh
./scripts/voice_studio_zrok.sh
```

The studio can also enforce optional access controls through environment variables:

```bash
export F5_TTS_STUDIO_USERNAME=demo
export F5_TTS_STUDIO_PASSWORD=change-me
export F5_TTS_STUDIO_TOKEN=optional-api-token
export F5_TTS_MAX_UPLOAD_MB=64
```

## Repository Layout

```text
src/f5_tts/
  api.py                 Public Python API
  infer/                 Inference entrypoints and demos
  studio/                Voice Studio app, runtime, storage, API, editing
  model/                 Model and trainer code
  train/                 Finetuning entrypoints
  runtime/               Deployment integrations

scripts/                 Local launch and sharing helpers
testsuite/               Test coverage for studio and service flows
ckpts/                   Optional local checkpoints
```

## Development Commands

Syntax check:

```bash
python -m py_compile src/f5_tts/api.py
```

Run targeted tests:

```bash
python -m unittest -v testsuite.test_studio_service
```

Run the full test suite:

```bash
python -m unittest discover -s testsuite
```

## Training

Finetuning entrypoints are available when training extras are installed:

```bash
f5-tts_finetune-cli
f5-tts_finetune-gradio
```

For a deeper walkthrough of inference usage, examples, and editing tools, see:

- [Inference Guide](src/f5_tts/infer/README.md)
- [Runtime Deployment Notes](src/f5_tts/runtime/triton_trtllm/README.md)

## Responsible use

Use voices only with the speaker's informed permission. Generated audio should
be disclosed wherever a listener could reasonably mistake it for an authentic
recording. Local processing reduces data exposure; it does not remove the
legal or ethical responsibility attached to a cloned voice.

## License

MIT. See [`LICENSE`](LICENSE) and retain upstream notices when redistributing
the model code or derivative builds.
