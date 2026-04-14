import streamlit as st
import soundfile as sf
import torch
import requests
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VoxClone AI",
    page_icon="🎙️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ─── CUSTOM CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');

/* ── ROOT ── */
:root {
    --bg:        #07080d;
    --surface:   #0d0f1a;
    --border:    #1e2235;
    --accent:    #6c63ff;
    --accent2:   #ff6b9d;
    --accent3:   #00e5c3;
    --text:      #e8eaf6;
    --muted:     #5a5f7a;
    --glow:      0 0 40px rgba(108,99,255,.35);
}

/* ── GLOBAL ── */
html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    font-family: 'Syne', sans-serif;
    color: var(--text);
}
[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed; inset: 0; z-index: 0;
    background:
        radial-gradient(ellipse 80% 60% at 10% 10%,  rgba(108,99,255,.12) 0%, transparent 60%),
        radial-gradient(ellipse 60% 50% at 90% 90%,  rgba(255,107,157,.10) 0%, transparent 60%),
        radial-gradient(ellipse 50% 40% at 50% 50%,  rgba(0,229,195,.06) 0%, transparent 70%);
    pointer-events: none;
}
[data-testid="stAppViewContainer"] > * { position: relative; z-index: 1; }
.block-container { max-width: 760px; padding: 2.5rem 2rem 4rem; }

/* ── HEADER ── */
.vox-header {
    text-align: center;
    padding: 2.5rem 0 1rem;
    position: relative;
}
.vox-logo {
    font-size: 3.6rem;
    font-weight: 800;
    letter-spacing: -2px;
    background: linear-gradient(135deg, #6c63ff 0%, #ff6b9d 50%, #00e5c3 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1;
    margin-bottom: .4rem;
    text-shadow: none;
}
.vox-badge {
    display: inline-block;
    font-family: 'DM Mono', monospace;
    font-size: .7rem;
    font-weight: 500;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--accent3);
    background: rgba(0,229,195,.08);
    border: 1px solid rgba(0,229,195,.25);
    border-radius: 100px;
    padding: .3rem 1rem;
    margin-bottom: 1.2rem;
}
.vox-tagline {
    color: var(--muted);
    font-size: .95rem;
    font-weight: 400;
    margin: 0;
    letter-spacing: .5px;
}

/* ── DIVIDER ── */
.fancy-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--border) 30%, var(--accent) 50%, var(--border) 70%, transparent);
    margin: 1.8rem 0;
    border: none;
}

/* ── CARD ── */
.vox-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 1.8rem 2rem;
    margin-bottom: 1.4rem;
    transition: border-color .3s;
    position: relative;
    overflow: hidden;
}
.vox-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(108,99,255,.6), transparent);
}
.vox-card:hover { border-color: rgba(108,99,255,.45); }

.card-label {
    font-family: 'DM Mono', monospace;
    font-size: .68rem;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: .9rem;
    display: flex;
    align-items: center;
    gap: .5rem;
}
.card-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
}

/* ── MODE PILLS ── */
.mode-row { display: flex; gap: .8rem; margin-bottom: .5rem; }
.mode-pill {
    flex: 1;
    text-align: center;
    padding: .85rem 1rem;
    border-radius: 14px;
    cursor: pointer;
    font-size: .88rem;
    font-weight: 600;
    letter-spacing: .3px;
    border: 1.5px solid var(--border);
    background: transparent;
    color: var(--muted);
    transition: all .25s;
}
.mode-pill.active-basic  { border-color: var(--accent);  color: var(--accent);  background: rgba(108,99,255,.1); box-shadow: var(--glow); }
.mode-pill.active-eleven { border-color: var(--accent2); color: var(--accent2); background: rgba(255,107,157,.09); box-shadow: 0 0 40px rgba(255,107,157,.25); }

/* ── STREAMLIT OVERRIDES ── */
/* file uploader */
[data-testid="stFileUploader"] {
    background: rgba(108,99,255,.05) !important;
    border: 1.5px dashed rgba(108,99,255,.3) !important;
    border-radius: 16px !important;
    padding: 1rem !important;
    transition: border-color .3s !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(108,99,255,.6) !important;
}
[data-testid="stFileUploader"] label { color: var(--muted) !important; }

/* text area */
[data-testid="stTextArea"] textarea {
    background: rgba(255,255,255,.03) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 14px !important;
    color: var(--text) !important;
    font-family: 'DM Mono', monospace !important;
    font-size: .88rem !important;
    padding: 1rem !important;
    resize: vertical !important;
    transition: border-color .3s !important;
    min-height: 120px !important;
}
[data-testid="stTextArea"] textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(108,99,255,.15) !important;
    outline: none !important;
}
[data-testid="stTextArea"] label { display: none !important; }

/* radio */
[data-testid="stRadio"] { display: none !important; }

/* main button */
[data-testid="stButton"] > button {
    width: 100% !important;
    background: linear-gradient(135deg, #6c63ff, #a855f7) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 16px !important;
    padding: 1rem 2rem !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 1.05rem !important;
    font-weight: 700 !important;
    letter-spacing: .5px !important;
    cursor: pointer !important;
    box-shadow: 0 4px 30px rgba(108,99,255,.4) !important;
    transition: all .3s !important;
    text-transform: uppercase !important;
}
[data-testid="stButton"] > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 40px rgba(108,99,255,.6) !important;
}
[data-testid="stButton"] > button:active {
    transform: translateY(0) !important;
}

/* download button */
[data-testid="stDownloadButton"] > button {
    width: 100% !important;
    background: transparent !important;
    border: 1.5px solid var(--accent3) !important;
    color: var(--accent3) !important;
    border-radius: 14px !important;
    padding: .8rem 1.5rem !important;
    font-family: 'Syne', sans-serif !important;
    font-size: .9rem !important;
    font-weight: 600 !important;
    transition: all .25s !important;
    margin-top: .5rem !important;
}
[data-testid="stDownloadButton"] > button:hover {
    background: rgba(0,229,195,.08) !important;
    box-shadow: 0 0 24px rgba(0,229,195,.25) !important;
}

/* audio player */
audio {
    width: 100% !important;
    border-radius: 12px !important;
    margin-top: .5rem !important;
    accent-color: var(--accent) !important;
}

/* spinner */
[data-testid="stSpinner"] { color: var(--accent) !important; }

/* success / warning / error */
[data-testid="stAlert"] {
    border-radius: 14px !important;
    border: 1px solid !important;
}

/* ── STATS ROW ── */
.stats-row {
    display: flex;
    gap: 1rem;
    margin-top: .5rem;
}
.stat-chip {
    flex: 1;
    text-align: center;
    background: rgba(255,255,255,.03);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: .8rem .5rem;
}
.stat-num  { font-size: 1.4rem; font-weight: 800; background: linear-gradient(135deg,#6c63ff,#ff6b9d); -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text; }
.stat-lbl  { font-size: .65rem; font-family: 'DM Mono',monospace; letter-spacing: 2px; text-transform: uppercase; color: var(--muted); margin-top: .2rem; }

/* ── WAVEFORM ANIMATION ── */
.waveform {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 4px;
    height: 36px;
    margin: .8rem 0;
}
.wave-bar {
    width: 4px;
    border-radius: 4px;
    background: linear-gradient(180deg, var(--accent), var(--accent2));
    animation: wavePulse 1.2s ease-in-out infinite;
}
.wave-bar:nth-child(1)  { height: 14px; animation-delay: 0s;    }
.wave-bar:nth-child(2)  { height: 24px; animation-delay: .1s;   }
.wave-bar:nth-child(3)  { height: 32px; animation-delay: .2s;   }
.wave-bar:nth-child(4)  { height: 20px; animation-delay: .3s;   }
.wave-bar:nth-child(5)  { height: 28px; animation-delay: .15s;  }
.wave-bar:nth-child(6)  { height: 16px; animation-delay: .25s;  }
.wave-bar:nth-child(7)  { height: 32px; animation-delay: .05s;  }
.wave-bar:nth-child(8)  { height: 22px; animation-delay: .35s;  }
.wave-bar:nth-child(9)  { height: 18px; animation-delay: .1s;   }
.wave-bar:nth-child(10) { height: 26px; animation-delay: .2s;   }
.wave-bar:nth-child(11) { height: 14px; animation-delay: .3s;   }
.wave-bar:nth-child(12) { height: 30px; animation-delay: .0s;   }
@keyframes wavePulse {
    0%, 100% { transform: scaleY(.5); opacity: .5; }
    50%       { transform: scaleY(1); opacity: 1;   }
}

/* ── OUTPUT CARD ── */
.output-header {
    font-size: .7rem;
    font-family: 'DM Mono', monospace;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--accent3);
    margin-bottom: .6rem;
}
.output-glow {
    border-color: rgba(0,229,195,.35) !important;
}
.output-glow::before {
    background: linear-gradient(90deg, transparent, rgba(0,229,195,.5), transparent);
}

/* ── FOOTER ── */
.vox-footer {
    text-align: center;
    padding: 2rem 0 .5rem;
    font-size: .72rem;
    font-family: 'DM Mono', monospace;
    color: var(--muted);
    letter-spacing: 1.5px;
}
.vox-footer span { color: var(--accent2); }

/* hide streamlit chrome */
#MainMenu, footer, header, [data-testid="stToolbar"] { display: none !important; }
</style>
""", unsafe_allow_html=True)

# ─── SESSION STATE ─────────────────────────────────────────────────────────────
if "mode" not in st.session_state:
    st.session_state.mode = "basic"
if "generated" not in st.session_state:
    st.session_state.generated = False

# ─── HEADER ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="vox-header">
    <div class="vox-logo">VoxClone AI</div>
    <div class="vox-badge">✦ Neural Voice Synthesis ✦</div>
    <p class="vox-tagline">Transform text into lifelike speech — powered by deep learning</p>
</div>
""", unsafe_allow_html=True)

# Stats strip
st.markdown("""
<div class="stats-row">
    <div class="stat-chip"><div class="stat-num">2</div><div class="stat-lbl">Voice Engines</div></div>
    <div class="stat-chip"><div class="stat-num">16k</div><div class="stat-lbl">Sample Rate</div></div>
    <div class="stat-chip"><div class="stat-num">∞</div><div class="stat-lbl">Characters</div></div>
    <div class="stat-chip"><div class="stat-num">AI</div><div class="stat-lbl">Powered</div></div>
</div>
<hr class="fancy-divider">
""", unsafe_allow_html=True)

# ─── LOAD MODELS ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    model     = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
    vocoder   = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
    return processor, model, vocoder

processor, model, vocoder = load_models()

# ─── VOICE UPLOAD ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="vox-card">
    <div class="card-label">⬆ Voice Sample</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload voice (.wav)",
    type=["wav"],
    label_visibility="collapsed"
)
st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file:
    with open("input.wav", "wb") as f:
        f.write(uploaded_file.read())
    st.markdown("""
    <div class="vox-card">
        <div class="card-label">🎧 Voice Preview</div>
    """, unsafe_allow_html=True)
    st.audio("input.wav")
    st.markdown('</div>', unsafe_allow_html=True)

# ─── TEXT INPUT ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="vox-card">
    <div class="card-label">✏ Script</div>
""", unsafe_allow_html=True)

text = st.text_area(
    "Text",
    placeholder="Type or paste the text you want to synthesise…",
    label_visibility="collapsed",
    height=130
)

if text:
    char_count = len(text)
    word_count = len(text.split())
    st.markdown(f"""
    <div style="text-align:right; font-family:'DM Mono',monospace; font-size:.7rem; color:var(--muted); margin-top:.4rem;">
        {char_count} chars · {word_count} words
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ─── ENGINE SELECTION ─────────────────────────────────────────────────────────
st.markdown("""
<div class="vox-card">
    <div class="card-label">⚡ Engine</div>
    <div class="mode-row">
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    if st.button("⚡  Basic TTS  —  Fast", use_container_width=True, key="btn_basic"):
        st.session_state.mode = "basic"
with col2:
    if st.button("✨  ElevenLabs  —  Premium", use_container_width=True, key="btn_eleven"):
        st.session_state.mode = "elevenlabs"

mode_label = "⚡ Basic TTS (SpeechT5)" if st.session_state.mode == "basic" else "✨ ElevenLabs (High Fidelity)"
accent_color = "var(--accent)" if st.session_state.mode == "basic" else "var(--accent2)"

st.markdown(f"""
    </div>
    <div style="text-align:center; font-family:'DM Mono',monospace; font-size:.75rem; color:{accent_color}; margin-top:.5rem; letter-spacing:1.5px;">
        ACTIVE ENGINE → {mode_label}
    </div>
</div>
""", unsafe_allow_html=True)

# Hidden radio keeps logic clean
mode_radio = st.radio(
    "mode",
    ["Basic TTS (Fast)", "ElevenLabs (High Quality Voice)"],
    index=0 if st.session_state.mode == "basic" else 1,
    label_visibility="collapsed"
)

# ─── GENERATE ─────────────────────────────────────────────────────────────────

from dotenv import load_dotenv
import os

load_dotenv()

API_KEY = os.getenv("ELEVENLAB_API_KEY")


st.markdown("<div style='margin-top:.5rem;'></div>", unsafe_allow_html=True)
generate_clicked = st.button("🎙  Generate Voice", use_container_width=True, key="gen_btn")

if generate_clicked:
    if not text.strip():
        st.warning("⚠  Please enter some text to synthesise.")
    else:
        with st.spinner("Synthesising voice..."):

            st.markdown("""
            <div class="waveform">
                <div class="wave-bar"></div><div class="wave-bar"></div>
                <div class="wave-bar"></div><div class="wave-bar"></div>
                <div class="wave-bar"></div><div class="wave-bar"></div>
                <div class="wave-bar"></div><div class="wave-bar"></div>
                <div class="wave-bar"></div><div class="wave-bar"></div>
                <div class="wave-bar"></div><div class="wave-bar"></div>
            </div>
            """, unsafe_allow_html=True)

            # ── Basic TTS ──
            if st.session_state.mode == "basic":
                inputs = processor(text=text, return_tensors="pt")
                speaker_embeddings = torch.randn(1, 512)
                speech = model.generate_speech(
                    inputs["input_ids"],
                    speaker_embeddings,
                    vocoder=vocoder
                )
                sf.write("output.wav", speech.numpy(), samplerate=16000)

            # ── ElevenLabs ──
            else:
                voice_id = "EXAVITQu4vr4xnSDxMaL"
                response = requests.post(
                    f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
                    headers={
                        "xi-api-key": API_KEY,
                        "Content-Type": "application/json"
                    },
                    json={"text": text}
                )
                if response.status_code != 200:
                    st.error(f"ElevenLabs API Error ({response.status_code}): {response.text}")
                    st.stop()
                with open("output.wav", "wb") as f:
                    f.write(response.content)

        st.session_state.generated = True

# ─── OUTPUT SECTION ───────────────────────────────────────────────────────────
if st.session_state.generated:
    try:
        st.markdown("""
        <div class="vox-card output-glow" style="margin-top:1.2rem;">
            <div class="card-label" style="color:var(--accent3);">✅ Output Audio</div>
        """, unsafe_allow_html=True)

        st.audio("output.wav")

        st.markdown('</div>', unsafe_allow_html=True)

        with open("output.wav", "rb") as f:
            st.download_button(
                label="⬇  Download Audio File",
                data=f,
                file_name="voxclone_output.wav",
                mime="audio/wav",
                use_container_width=True
            )
    except FileNotFoundError:
        pass

# ─── FOOTER ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="vox-footer">
    Made with <span>♥</span> · VoxClone AI · Neural Voice Synthesis Platform
</div>
""", unsafe_allow_html=True)