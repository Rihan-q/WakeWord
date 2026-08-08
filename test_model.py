import numpy as np
import sounddevice as sd
from openwakeword.model import Model

# --- Wake word config ---
# (run live_ab_test.py or evaluate_wakeword.py to compare).
WAKE_MODEL_PATH = "/Users/rihan/Desktop/wake word/Hey_Leo.onnx"   # or Chage the path according to your's
WAKE_MODEL_KEY = "Hey_Leo"         # filename without .onnx, must match dict key openWakeWord returns

WAKE_THRESHOLD = 0.5                # tune based on evaluate_wakeword.py results
SAMPLE_RATE = 16000                 # openWakeWord requires 16kHz mono
CHUNK_SIZE = 1280                   # 80ms chunks, openWakeWord's expected frame size

oww_model = Model(
    wakeword_models=[WAKE_MODEL_PATH],
    inference_framework="onnx",
)


def listen_for_wake_word():
    """Blocks until 'Hey Leo' is detected"""
    print(f"Listening for wake word ({WAKE_MODEL_PATH})...")

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16",
        blocksize=CHUNK_SIZE,
    ) as stream:
        while True:
            audio_chunk, _ = stream.read(CHUNK_SIZE)
            audio_chunk = audio_chunk.flatten()

            prediction = oww_model.predict(audio_chunk)
            score = prediction[WAKE_MODEL_KEY]

            if score > WAKE_THRESHOLD:
                print(f"Wake word detected! (confidence: {score:.2f})")
                oww_model.reset()  # clear internal buffers before next listen
                return

if __name__ == "__main__":
    while True:
        listen_for_wake_word()