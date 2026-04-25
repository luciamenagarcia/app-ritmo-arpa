import os
import tempfile
import base64
import numpy as np
import streamlit as st
import librosa
import streamlit.components.v1 as components
from audiorecorder import audiorecorder

# -----------------------
# CONFIG
# -----------------------

st.set_page_config(page_title="Juego de Ritmo", layout="centered")

# -----------------------
# RUTAS BASE
# -----------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# -----------------------
# NIVELES
# -----------------------

levels = [
    {"nivel": 1, "nombre": "Negras", "img": os.path.join(BASE_DIR, "data", "nivel1.png"), "audio": os.path.join(BASE_DIR, "data", "nivel1.wav"), "tdi_threshold": 0.087, "color1": "#EAF4FF", "color2": "#F7FBFF", "accent": "#2D8CFF"},
    {"nivel": 2, "nombre": "Corcheas", "img": os.path.join(BASE_DIR, "data", "nivel2.png"), "audio": os.path.join(BASE_DIR, "data", "nivel2.wav"), "tdi_threshold": 0.095, "color1": "#EAFBF1", "color2": "#F8FFFB", "accent": "#2EB872"},
    {"nivel": 3, "nombre": "Puntillo", "img": os.path.join(BASE_DIR, "data", "nivel3.png"), "audio": os.path.join(BASE_DIR, "data", "nivel3.wav"), "tdi_threshold": 0.110, "color1": "#FFF3E6", "color2": "#FFFBF6", "accent": "#FF9F43"},
    {"nivel": 4, "nombre": "Semicorcheas", "img": os.path.join(BASE_DIR, "data", "nivel4.png"), "audio": os.path.join(BASE_DIR, "data", "nivel4.wav"), "tdi_threshold": 0.130, "color1": "#F2ECFF", "color2": "#FBF9FF", "accent": "#7B61FF"},
    {"nivel": 5, "nombre": "Tresillos", "img": os.path.join(BASE_DIR, "data", "nivel5.png"), "audio": os.path.join(BASE_DIR, "data", "nivel5.wav"), "tdi_threshold": 0.150, "color1": "#FFEAF3", "color2": "#FFF8FB", "accent": "#FF5C8A"},
]

# -----------------------
# ESTADO
# -----------------------

if "nivel_actual" not in st.session_state:
    st.session_state.nivel_actual = 0

nivel = levels[st.session_state.nivel_actual]

# -----------------------
# CSS
# -----------------------

st.markdown(f"""
<style>
.stApp {{
    background: linear-gradient(180deg, {nivel["color1"]} 0%, {nivel["color2"]} 100%);
    --accent-color: {nivel["accent"]};
}}

.main-title {{
    text-align: center;
    font-size: 44px;
    font-weight: 800;
}}

.subtitle {{
    text-align: center;
    color: #666;
    margin-bottom: 20px;
}}

.level-title {{
    text-align: center;
    font-size: 28px;
    font-weight: 800;
}}

.section-divider {{
    height: 1px;
    background-color: rgba(0,0,0,0.1);
    margin: 30px 0;
}}

/* Botones */
div.stButton > button {{
    border-radius: 10px;
    border: 1px solid var(--accent-color);
    border-bottom: 3px solid var(--accent-color);
    background: white;
    font-weight: 600;
}}

button[kind="secondary"] {{
    border-radius: 10px !important;
    border: 1px solid var(--accent-color) !important;
    border-bottom: 3px solid var(--accent-color) !important;
    background: white !important;
    font-weight: 600 !important;
}}

/* QUITAR FRANJA BLANCA */
.record-wrapper {{
    background: transparent !important;
    padding: 0 !important;
    margin: 0 !important;
}}

.record-wrapper iframe {{
    background: transparent !important;
}}
</style>
""", unsafe_allow_html=True)

# -----------------------
# HEADER
# -----------------------

st.markdown("<div class='main-title'>Juego de Ritmo</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Escucha, toca y comprueba si tu ritmo está bien</div>", unsafe_allow_html=True)

st.progress((st.session_state.nivel_actual + 1) / len(levels))

# -----------------------
# AUDIO PLAYER
# -----------------------

def audio_player(path, accent):
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    components.html(f"""
    <audio id="a" controls style="width:100%">
        <source src="data:audio/wav;base64,{b64}">
    </audio>
    <button onclick="var a=document.getElementById('a');a.currentTime=0;a.play();"
    style="margin-top:10px;border:1px solid {accent};border-bottom:3px solid {accent};padding:8px 16px;border-radius:10px;background:white;font-weight:600;">
    Volver a escuchar
    </button>
    """, height=120)

# -----------------------
# FUNCIONES
# -----------------------

def reset_grabacion():
    for k in list(st.session_state.keys()):
        if k.startswith("audio_recorder"):
            del st.session_state[k]

def compute_tdi_metrics(ref_path, user_path):
    y_ref, sr = librosa.load(ref_path, sr=None)
    y_usr, _ = librosa.load(user_path, sr=sr)

    _, wp = librosa.sequence.dtw(X=y_usr.reshape(1,-1), Y=y_ref.reshape(1,-1))
    d = wp[:,1]-wp[:,0]
    return {"tdi_norm": float(np.mean(np.abs(d)))}

# -----------------------
# UI
# -----------------------

st.markdown(f"<div class='level-title'>Nivel {nivel['nivel']} - {nivel['nombre']}</div>", unsafe_allow_html=True)
st.image(nivel["img"], use_container_width=True)

st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

st.write("Escucha el ejemplo")
audio_player(nivel["audio"], nivel["accent"])

st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

st.write("Ahora te toca a ti")

st.markdown("<div class='record-wrapper'>", unsafe_allow_html=True)
audio = audiorecorder("Grabar", "Parar", key=f"audio_recorder_{st.session_state.nivel_actual}")
st.markdown("</div>", unsafe_allow_html=True)

if len(audio) > 0:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        path = tmp.name
        audio.export(path, format="wav")

    st.success("¡Grabación completada!")
    st.audio(path)

    if st.button("Evaluar ritmo"):
        m = compute_tdi_metrics(nivel["audio"], path)

        if m["tdi_norm"] <= nivel["tdi_threshold"]:
            st.success("¡Muy bien!")
            if st.button("Siguiente nivel"):
                st.session_state.nivel_actual += 1
                reset_grabacion()
                st.rerun()
        else:
            st.warning("Inténtalo otra vez")

st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

if col1.button("Anterior"):
    if st.session_state.nivel_actual > 0:
        st.session_state.nivel_actual -= 1
        reset_grabacion()
        st.rerun()

if col2.button("Siguiente"):
    if st.session_state.nivel_actual < len(levels)-1:
        st.session_state.nivel_actual += 1
        reset_grabacion()
        st.rerun()
