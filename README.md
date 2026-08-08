# Hey Leo — Custom Wake Word Model

A custom "hey leo" wake word detection model trained with [livekit-wakeword](https://github.com/livekit/livekit-wakeword), an open-source toolkit for training on-device wake word models from synthetic (TTS-generated) speech — no manual audio recording required.

The training pipeline synthesizes positive/negative speech samples, augments them with noise and reverb, extracts audio features, trains a 3-phase adaptive classifier, and exports a production-ready ONNX model.

## Requirements

- **Python 3.11 or higher** (required)
- System packages: `espeak-ng`, `ffmpeg`, `sox`, `libsndfile`, `portaudio`

**macOS**
```bash
brew install espeak-ng ffmpeg sox libsndfile portaudio
```

**Ubuntu/Debian**
```bash
sudo apt install espeak-ng ffmpeg sox libsndfile1 portaudio19-dev
```

> **If you have multiple Python versions installed**, don't rely on the plain `python3` command — it might point to an older version. Check first:
> ```bash
> python3 --version
> ```
> If that isn't 3.11+, use the versioned command explicitly (e.g. `python3.11`) for every step below. On macOS, install it if needed with `brew install python@3.11`.

## Setup

1. **Clone the repo**
```bash
git clone https://github.com/<your-username>/hey-leo-wakeword.git
cd hey-leo-wakeword
```

2. **Create and activate a virtual environment**

Use `python3.11` explicitly (or whichever 3.11+ version you have) to make sure the venv is built on the right interpreter:
```bash
python3.11 -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows
```
Your terminal prompt should now show `(venv)` at the start of the line, confirming it's active.

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **When you're done working**, deactivate the virtual environment:
```bash
deactivate
```
This returns your terminal to the system Python. Run `source venv/bin/activate` again next time you want to work on this project.

## Training

1. **Download required model artifacts** (TTS voice model, embedding model, and the ACAV100M negative-audio dataset — this step downloads several GB, so it can take a while depending on your connection):
```bash
livekit-wakeword setup --config hey_leo.yaml
```

2. **Run the full training pipeline** (generate → augment → extract → train → export → evaluate):
```bash
python3 train.py
```

This single script runs every stage in sequence. It will take a while to complete — expect a real training run, not a quick script.

## Output

On success, you'll get:
- `output/hey_leo/hey_leo.pt` — trained PyTorch checkpoint
- `output/hey_leo/hey_leo.onnx` — exported ONNX model, ready for on-device inference
- Printed evaluation metrics:
```
AUT=0.00xx  FPPH=0.xx  Recall=xx.x%
```
  - **AUT** — area under the false-positive-rate curve (lower is better)
  - **FPPH** — false positives per hour (lower is better)
  - **Recall** — how often the model correctly detects "hey leo" (higher is better)

## Testing the model

Model testing uses the `openwakeword` package directly (the exported `.onnx` model is fully compatible with it — `openwakeword` and `sounddevice` are already listed in `requirements.txt`).

1. **Download openWakeWord's required resource models** (mel-spectrogram + embedding models — one-time step):
```bash
python3 -c "import openwakeword; openwakeword.utils.download_models()"
```

2. **Run the live mic test**:
```bash
python3 test_model.py
```

It prints `Waiting for wake word...`, listens continuously, and prints a confidence score plus `Wake word detected!` whenever "Hey Leo" is heard, then goes back to waiting.

> **Sanity check first**: if detection never fires, run `python3 mic_check.py` to confirm your microphone is actually capturing audio (peak/rms values should rise noticeably when you talk). This isolates mic/permission issues from model issues.

If you hit a `KeyError` on the prediction dictionary, your model's actual key may not match `WAKE_MODEL_KEY` in `test_model.py` — print `list(prediction.keys())` once to confirm the real key name and update the constant accordingly.

The exported ONNX model is compatible with `openWakeWord`-based integrations (e.g. Home Assistant) with no changes.

## Files

| File | Purpose |
|---|---|
| `hey_leo.yaml` | Training config (data generation, model architecture, training hyperparameters) |
| `train.py` | Runs the full pipeline: generate, augment, extract, train, export, evaluate |
| `requirements.txt` | Python dependencies — install with `pip install -r requirements.txt` |
| `test_model.py` | Live microphone test — listens for "Hey Leo" and prints detection + confidence |
