# 🎧 Air Traffic Control (ATC) ASR Transcription Pipeline

![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![React](https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB)
![SQLite](https://img.shields.io/badge/SQLite-07405E?style=for-the-badge&logo=sqlite&logoColor=white)
![CUDA / GPU Support](https://img.shields.io/badge/CUDA%20%2F%20GPU-Supported-76B900?style=for-the-badge&logo=nvidia&logoColor=white)

An end-to-end, high-performance Automatic Speech Recognition (ASR) system tailored specifically for **Air Traffic Control (ATC)** audio communications. This project integrates acoustic preprocessing (bandpass filtering and spectral noise reduction), robust Voice Activity Detection (VAD), domain-specific hotword biasing, and a fast inference wrapper (`crispasr.exe`) powered by fine-tuned speech models. It provides both a **FastAPI** backend server with SQLite persistence and a modern **React/Vite** web frontend.

---

## 📑 Table of Contents
- [✨ Project Overview & Workflow](#-project-overview--workflow)
- [✨ Key Features](#-key-features)
- [🛠️ Phase 1: Prerequisites & External Dependencies](#-phase-1-prerequisites--external-dependencies)
  - [1. Download the Crisp ASR BINARIES](#1-download-the-crisp-asr-binaries)
  - [2. Install ffmpeg](#2-install-ffmpeg)
  - [3. Downloaded trained model & Convert to GGUF](#3-downloaded-trained-model--convert-to-gguf)
  - [4. VAD model](#4-vad-model)
- [🚀 Phase 2: Installation & Running the Application](#-phase-2-installation--running-the-application)
  - [1. How to run Backend](#how-to-run-backend)
  - [2. How to run Frontend](#how-to-run-frontend)
- [🕹️ Usage & API Guide](#-usage--api-guide)
- [📂 Project Structure & File Descriptions](#-project-structure--file-descriptions)
- [❓ Troubleshooting & Support](#-troubleshooting--support)

---

## ✨ Project Overview & Workflow

Aviation audio is typically plagued by narrow radio bandwidth, static, and rapid multi-speaker speech. This pipeline solves these challenges by chaining three distinct stages:

```
[Raw ATC Audio Input]
         │
         ▼
[Stage 1: Preprocessing & Noise Removal]
  ├── Butterworth Bandpass Filter (100Hz - 8000Hz)
  └── Spectral Gating Noise Reduction (noisereduce coefficient: 0.75)
         │
         ▼
[Stage 2: Voice Activity Detection & Chunking]
  ├── Automatic duration-based chunking (e.g., 30s chunks, 5s overlap)
  └── VAD speech timestamp isolation
         │
         ▼
[Stage 3: Fast ASR Inference Engine (crispasr.exe)]
  ├── Beam Search (size 5) + Forced English Language (-l en)
  └── Domain Biasing (Hotwords: ils, dme, localizer, runway, heading, etc.)
         │
         ▼
[Output: Structured JSON + SQLite Persistence + Web UI Visualization]
```

---

## ✨ Key Features
- **Global & Chunk-Based Noise Removal**: Automatically filters frequencies outside human speech bounds (100Hz–8000Hz) using Butterworth bandpass filtering and suppresses non-stationary background acoustic noise (`noisereduce`).
- **Domain-Specific Hotword Biasing**: Prioritizes ATC callsigns, waypoints, and aviation terminology (e.g., `ils`, `dme`, `localizer`, `runway`, `heading`, `knots`, `altitude`, `approach`).
- **Overlapping Chunk Processing**: Supports automatic or manual time-chunked audio processing (e.g., 30s chunks with 5s overlaps) for long-duration recordings with parallel multi-worker execution.
- **SQLite Transcription Storage**: Persistent record management for all processed transcriptions, speech durations, device modes (`cuda`/`cpu`), and intermediate audio chunk paths (`transcriptions.db`).
- **REST API & Web UI**: Full-featured API providing file upload and query endpoints paired with a responsive web dashboard.

---

## 🛠️ Phase 1: Prerequisites & External Dependencies

> [!IMPORTANT]
> **Complete these setup steps *before* starting the backend server** to ensure `crispasr.exe`, `ffmpeg`, and the GGUF models are properly configured on your system.

### 1. Download the Crisp ASR BINARIES

#### For Windows: 

##### CPU 
https://github.com/CrispStrobe/CrispASR/releases/download/v0.8.10/crispasr-windows-x86_64-cpu-legacy.zip

##### GPU 
https://github.com/CrispStrobe/CrispASR/releases/download/v0.8.10/crispasr-windows-x86_64-cuda.zip


#### For Linux

##### CPU
https://github.com/CrispStrobe/CrispASR/releases/download/v0.8.10/crispasr-linux-x86_64.tar.gz

##### GPU cuda 13
https://github.com/CrispStrobe/CrispASR/releases/download/v0.8.10/crispasr-linux-x86_64-cuda13.tar.gz

##### GPU cuda 12
https://github.com/CrispStrobe/CrispASR/releases/download/v0.8.10/crispasr-linux-x86_64-cuda.tar.gz

copy them into bin/cuda or bin/cpu based on your system

---

### 2. Install ffmpeg 

You can download an installer for your OS from the [ffmpeg Website](https://ffmpeg.org/download.html).  

Or use a package manager:

- **On Ubuntu or Debian**:
    ```bash
    sudo apt update && sudo apt install ffmpeg
    ```

- **On Arch Linux**:
    ```bash
    sudo pacman -S ffmpeg
    ```

- **On MacOS using Homebrew** ([https://brew.sh/](https://brew.sh/)):
    ```bash
    brew install ffmpeg
    ```

- **On Windows using Winget** [official documentation](https://learn.microsoft.com/en-us/windows/package-manager/winget/) :
    ```bash
    winget install Gyan.FFmpeg
    ```
    
- **On Windows using Chocolatey** ([https://chocolatey.org/](https://chocolatey.org/)):
    ```bash
    choco install ffmpeg
    ```

- **On Windows using Scoop** ([https://scoop.sh/](https://scoop.sh/)):
    ```bash
    scoop install ffmpeg
    ```   

---

### 3. Download Trained Model & Convert to GGUF

#### Download Trained Model 
Download the trained NeMo checkpoint model from [Hugging Face](https://huggingface.co/qenneth/parakeet-tdt-0.6b-v3-finetuned-for-ATC/blob/main/parakeet-tdt-0.6b-v3-finetuned-for-ATC.nemo).

#### Convert to GGUF Format
Using the conversion [script](https://github.com/CrispStrobe/CrispASR/blob/main/models/convert-parakeet-to-gguf.py):

```bash
python convert-parakeet-to-gguf.py --nemo <file.nemo> --output models/ggufs/speech-model.gguf 
```

---

### 4. Download & Cache Silero VAD Model

CrispASR utilizes the Silero Voice Activity Detection (VAD) model (`ggml-silero-v6.2.0.bin`) to accurately isolate active speech timestamps before inference.

#### Download the Silero VAD Model
- **Direct Download Link**: [ggml-silero-v6.2.0.bin](https://huggingface.co/ggml-org/whisper-vad/resolve/main/ggml-silero-v6.2.0.bin)

#### Where CrispASR Caches the VAD Model by OS:
- **On Windows**: Place the downloaded `ggml-silero-v6.2.0.bin` inside:
  ```cmd
  C:\Users\<username>\.cache\crispasr\
  ```
  *(Alternatively, run `crispasr.exe --vad` once locally so it auto-downloads into this directory.)*

- **On Linux / inside Docker (`/root` or `$HOME`)**: Place `ggml-silero-v6.2.0.bin` inside:
  ```bash
  $HOME/.cache/crispasr/
  # Or explicitly inside container runtime:
  /root/.cache/crispasr/
  ```
  *(Note: Our Dockerfiles automatically download and cache `ggml-silero-v6.2.0.bin` inside `/root/.cache/crispasr/` and `/app/cache/` during image build, ensuring zero-configuration offline execution.)*

---

## 🚀 Phase 2: Installation & Running the Application

Once you have downloaded the binaries, models, and dependencies above, follow these exact instructions to launch the application:

## How to run Backend
1. Create a virtual environment
```bash
python -m venv venv
```
2. Activate the virtual environment
```cmd
venv\Scripts\activate
```
or In Linux
```bash
source venv/bin/activate
```
3. Install dependencies
```bash
pip install -r requirements_transcribe.txt
```
4. Run the application

```bash
uvicorn app:app --reload
```

## How to run Frontend
```bash
cd frontend
npm install
npm run dev
```

## How to run via Docker
Two standalone Dockerfiles are provided inside the `docker/` directory targeting **Python 3.12**. The speech model (`speech-model.gguf`), Silero VAD (`ggml-silero-v6.2.0.bin`), and SQLite database (`transcriptions.db`) are **completely self-contained within the built container**.

### 1. Build and Run for CPU
```bash
# Build image from project root
docker build -f docker/Dockerfile.cpu -t atc-asr:cpu .

# Run container (API on port 8000)
docker run -d -p 8000:8000 --name atc-asr-cpu atc-asr:cpu
```

### 2. Build and Run for GPU (CUDA 12)
```bash
# Build image from project root
docker build -f docker/Dockerfile.gpu -t atc-asr:gpu .

# Run container with NVIDIA GPU acceleration
docker run -d --gpus all -p 8000:8000 --name atc-asr-gpu atc-asr:gpu
```

*(Optional: Add `-v "%cd%/output:/app/output"` to `docker run` if you wish to export generated audio chunks directly to your local Windows directory.)*

---

## 🕹️ Usage & API Guide

### Interactive API Documentation (Swagger UI)
Once the backend is running (`uvicorn app:app --reload`), visit:
- **Swagger Docs:** `http://127.0.0.1:8000/docs`
- **ReDoc Docs:** `http://127.0.0.1:8000/redoc`

### Example `cURL` Transcription Request
To transcribe an audio file directly from your terminal using `curl`:

```bash
curl -X POST "http://127.0.0.1:8000/transcribe" ^
     -F "file=@test-files/segment_03_30-40min.wav" ^
     -F "apply_noise_removal=true" ^
     -F "noise_reduction=0.75" ^
     -F "language=en" ^
     -F "save_intermediate=true"
```

### CLI Direct Execution
You can also run the pipeline directly via terminal on any audio file using `transcribe_atc.py` or `run_transcription.py`:

```bash
python transcribe_atc.py test-files\audio_0.wav --model models\ggufs\speech-model.gguf --exe_path bin\cuda\crispasr.exe
```

---

## 📂 Project Structure & File Descriptions

### Core Backend Files
| File Name | Description |
| :--- | :--- |
| **`app.py`** | **FastAPI Application Entry Point**: Defines the web server and REST API routes (`/`, `/transcribe`, `/transcriptions`, `/transcriptions/{id}`). Handles multipart audio uploads, triggers the acoustic pipeline, mounts static output folders (`/output`), and records metrics to SQLite. |
| **`transcribe_atc.py`** | **Core Audio Transcription Pipeline**: Implements the `AudioTranscriptionPipeline` class. Manages audio loading via `librosa`, bandpass filtering (`scipy.signal.butter`), spectral gating noise reduction (`noisereduce`), intermediate overlapping chunk saving, and executes the `crispasr.exe` inference wrapper with beam search (`-bs 5`) and hotwords. |
| **`run_transcription.py`** | **Single-File & Parallel Chunking Engine**: A versatile CLI script and module for running transcription on single audio files. Features `determine_chunking_config` for duration-based auto-scaling, parallel chunk processing (`AudioChunker`, `ParallelAudioProcessor`), VAD-driven paragraph breaks, and timestamp-adjusted result merging. |
| **`database.py`** | **Database Persistence Layer**: Handles connection initialization (`init_db`) and CRUD operations for the local SQLite database (`transcriptions.db`). Stores transcription records, JSON segments, VAD/denoising flags, execution time metrics, and relative file paths for intermediate audio chunks. |

### Configuration & Frontend Structure
| File / Directory Name | Description |
| :--- | :--- |
| **`frontend/`** | **Web Frontend Application**: Contains the React + Vite single-page application (`index.html`, `vite.config.js`, `package.json`, `src/`) for user interaction, audio uploading, and visualizing transcriptions. |
| **`requirements_transcribe.txt`** | **Core Backend Dependencies**: Lists required Python packages (`numpy`, `scipy`, `librosa`, `soundfile`, `noisereduce`, `fastapi`, `uvicorn`, `python-multipart`, `gguf`) for running the transcription backend and audio processing filters. |
| **`commands.txt`** | **Quick Reference Commands**: A cheat sheet containing sample `crispasr.exe` terminal commands and `curl` HTTP request snippets for interacting with the `/transcribe` endpoint. |

---

## ❓ Troubleshooting & Support

- **`crispasr.exe not found`**: Ensure you have placed the executable inside `bin/cuda/crispasr.exe` (or `bin/cpu/crispasr.exe` if running on CPU) as instructed in [Phase 1](#1-download-the-crisp-asr-binaries).
- **Audio Loading / Resampling Errors**: Verify that `ffmpeg` is installed and accessible in your system `PATH` (`ffmpeg -version`).
- **GPU Inference Fails**: If CUDA inference crashes, switch your target device to `cpu` when calling `/transcribe` (`-F "device=cpu"`) or run using `bin/cpu/crispasr.exe`.
- **Inspecting CLI Flags**: Run `bin/cuda/crispasr.exe --help` directly in your terminal to view all available low-level arguments supported by `crispasr.exe`.