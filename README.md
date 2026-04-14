# VoxClone-

# 🎙 VoxClone — AI Voice Cloning System

## 🚀 What is VoxClone?

**VoxClone** is an AI-based voice generation system that allows users to **convert text into speech using a specific voice**. It demonstrates both:

* **Basic Text-to-Speech (TTS)** using pretrained models
* **High-quality voice generation** using external APIs (ElevenLabs)

👉 The goal of this project is to help understand **how modern voice cloning systems work internally** while also providing a **working application**.

---

## 🧠 Model Architecture (Core Concept)

A typical voice cloning system is built using **3 main components**:

---

### 1️⃣ Speaker Encoder

👉 **What it does:**

* Takes a voice sample (audio)
* Extracts unique features of that voice
* Converts it into a numerical vector called **speaker embedding**

👉 **Simple idea:**

> “This is how this person sounds”

---

### 2️⃣ Synthesizer

👉 **What it does:**

* Takes:

  * **Text**
  * **Speaker embedding**
* Generates an intermediate representation called a **mel spectrogram**

👉 **Simple idea:**

> “What should this person sound like while speaking this text?”

---

### 3️⃣ Vocoder

👉 **What it does:**

* Converts the **mel spectrogram → actual audio waveform**

👉 **Simple idea:**

> “Turn AI representation into real sound”

---

## 🔄 Full Pipeline

```text
Voice Sample → Speaker Encoder → Speaker Embedding
Text + Embedding → Synthesizer → Mel Spectrogram
Mel Spectrogram → Vocoder → Generated Speech
```

---

### 🔍 Example

Suppose:

* Voice sample = **Dhruv’s voice**
* Text = *"Hello, I am learning AI"*

👉 Process:

1. Speaker Encoder → captures Dhruv’s voice features
2. Synthesizer → combines text + voice style
3. Vocoder → produces final audio

👉 Output:

> AI-generated speech sounding like Dhruv (approximation)

---

## 📦 Libraries Used

| Library        | Purpose                                          |
| -------------- | ------------------------------------------------ |
| **torch**      | Deep learning framework for model inference      |
| **torchaudio** | Audio processing with PyTorch                    |
| **librosa**    | Audio loading, normalization, feature extraction |
| **soundfile**  | Saving generated audio files                     |
| **spacy**      | Text processing (optional NLP enhancements)      |
| **TTS**        | Pretrained Text-to-Speech models                 |
| **tqdm**       | Progress tracking during processing              |

---

## ⚠️ Training Note

I **did NOT train the model from scratch** because:

```text
Training requires large datasets + GPU + long time (hours/days)
```

👉 Instead, I used **pretrained models** for:

* Faster development
* Practical implementation

---

## 🧪 (Optional) Training Code (Basic Idea)

Below is a simplified example of how training would look:

```python
from TTS.api import TTS

# Load model
tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC")

# Train (pseudo example)
tts.train(
    dataset_path="your_dataset",
    epochs=10,
    batch_size=16
)
```

👉 Note:

* Real training requires **clean dataset + transcripts**
* GPU strongly recommended

---

## 🛠 Features Implemented

### 🔹 1. Basic TTS (Local)

* Uses SpeechT5
* Fast but **no real voice cloning**

---

### 🔹 2. ElevenLabs Integration

* High-quality voice generation
* Realistic output
* API-based approach

---

### 🔹 3. Streamlit App

* Upload audio
* Enter text
* Generate speech
* Download output

---

## ⚙️ How to Run the Project

### 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/VoxClone.git
cd VoxClone
```

---

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

---

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4️⃣ Run Application

```bash
python -m streamlit run app.py
```

---

## 🧑‍💻 How to Use

1. Upload a `.wav` audio file
2. Enter text
3. Choose mode:

   * Basic TTS
   * ElevenLabs
4. Click **Generate Voice**
5. Download output

---

## 📌 Important Learnings

* Voice cloning requires:

  * aligned embeddings
  * trained models
* Pretrained models = fast but limited
* APIs = production-level results

---

## ❤️ Made with love by Bit-Bard
