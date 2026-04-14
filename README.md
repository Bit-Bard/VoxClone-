# VoxClone 🎤

**Turn any text into speech that sounds like a specific person.**

VoxClone is an AI voice synthesis system that demonstrates how modern voice cloning actually works — from extracting a speaker's vocal identity to generating natural-sounding audio. It ships with both a local TTS engine and a high-quality API integration, wrapped in a clean Streamlit interface.

![VoxClone Interface](Voxclone_interface.png)

---

## How it works

Voice cloning isn't magic — it's a three-stage pipeline where each component has one clear job.

```
┌──────────────────┐
│   Voice Sample   │──────────────────────────────────────────────┐
│    (.wav file)   │                                              │
└──────────────────┘                                              ▼
                                                     ┌────────────────────┐
┌──────────────────┐                                 │  Speaker Encoder   │
│    Text Input    │──────────────────────────┐      │                    │
│  "Hello, world"  │                          │      │  Extracts a unique │
└──────────────────┘                          │      │  voice fingerprint │
                                              │      │  → speaker embed.  │
                                              │      └────────────────────┘
                                              │               │
                                              ▼               ▼
                                       ┌─────────────────────────┐
                                       │       Synthesizer        │
                                       │                         │
                                       │  Fuses text + identity  │
                                       │  → mel spectrogram      │
                                       └─────────────────────────┘
                                                    │
                                                    ▼
                                          ┌──────────────────┐
                                          │     Vocoder      │
                                          │                  │
                                          │  Spectrogram     │
                                          │  → audio signal  │
                                          └──────────────────┘
                                                    │
                                                    ▼
                                          🔊 Generated Speech
```

### Stage 1 — Speaker Encoder
Listens to a voice sample and extracts its unique characteristics — pitch, cadence, timbre — into a compact numerical vector called a **speaker embedding**. Think of it as a voice fingerprint: a concise description of *how* someone sounds.

### Stage 2 — Synthesizer
Takes the text you want to say *and* the speaker embedding, then produces a **mel spectrogram** — a visual representation of what the speech should look like in terms of frequency and time. This is where voice identity gets fused with words.

### Stage 3 — Vocoder
Converts the mel spectrogram into an actual **audio waveform** — the file you can press play on. No voice identity lives here; it's purely a translator between AI representation and real sound.

---

## Features

| | Feature | Description |
|---|---|---|
| `local` | **Basic TTS** | Lightweight synthesis using SpeechT5. Fast to run, no API key needed. |
| `api` | **ElevenLabs** | Production-quality voice generation. Noticeably more realistic output. |
| `ui` | **Streamlit App** | Upload audio, type text, pick a mode, download the result. |

---

## Getting started

```bash
# 1. Clone the repo
git clone https://github.com/your-username/VoxClone.git
cd VoxClone

# 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the app
python -m streamlit run app.py
```

---

## Usage

1. Upload a `.wav` file of the voice you want to clone
2. Type the text you want spoken
3. Choose a mode — **Basic TTS** for local speed, **ElevenLabs** for quality
4. Hit **Generate Voice**
5. Download the output audio

---

## Libraries

| Library | Role |
|---|---|
| `torch` + `torchaudio` | Deep learning inference and audio tensor handling |
| `librosa` | Audio loading, normalization, and feature extraction |
| `soundfile` | Saving the generated `.wav` output |
| `TTS` | Pretrained text-to-speech models (Tacotron2, SpeechT5) |
| `spacy` | Optional NLP preprocessing for cleaner text handling |
| `tqdm` | Progress tracking during processing steps |

---

## On training

This project uses **pretrained models** rather than training from scratch. Full training demands clean datasets, transcripts, GPU time, and patience — often hours or days. Using pretrained weights means you can focus on understanding the pipeline and building the application instead.

For reference, a minimal training setup looks like this:

```python
from TTS.api import TTS

tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC")
tts.train(
    dataset_path="your_dataset",
    epochs=10,
    batch_size=16
)
```

For production-quality output, the ElevenLabs integration is the recommended path.

---

## What I Learned
1. Voice cloning is not just TTS → it requires speaker identity modeling using embeddings
2. Embedding quality matters more than model complexity → mismatch leads to poor cloning
3. Pretrained models are useful but limited → true cloning needs fine-tuning or aligned systems
4. SpeechT5 requires speaker embeddings → it cannot generate speech independently
5. Pipeline understanding is critical: Encoder → Synthesizer → Vocoder
6. Audio preprocessing (trim, normalize) directly impacts output quality
7. Model loading optimization (caching) improves performance significantly
8. Real-world AI systems often rely on APIs (like ElevenLabs) for production-level results
9. Free-tier APIs have limitations, so system design must adapt accordingly
10. Debugging environment issues is part of ML engineering, not just modeling
11. Choosing the right tool is more important than forcing a solution
12. Building a working product > perfect model accuracy in real-world projects

## Key takeaways

- Voice cloning quality depends heavily on the alignment between the speaker embedding and the synthesis model
- Pretrained models are fast and practical, but have a ceiling on how closely they can match an unseen voice
- API-based solutions close the quality gap significantly for real-world applications

---

*Made with 🩷 by **Bit-Bard***
### Connect with me 
Gmail : dhruvdevaliya@gmail.com
Linkedln : https://www.linkedin.com/in/dhruv-devaliya/
