import os
import tempfile
import numpy as np
import streamlit as st
import librosa
from audiorecorder import audiorecorder

# -----------------------
# CONFIG
# -----------------------

st.set_page_config(page_title="Juego de Ritmo", layout="centered")

st.markdown("""
<style>
.main {background-color: #FFFDF7;}
h1 {text-align: center; color: #FF6B6B;}
.level-title {font-size: 28px; text-align: center; margin-bottom: 10px; font-weight: 700;}
.small-text {text-align: center; color: #666666; font-size: 16px;}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>🎵 Juego de Ritmo 🎵</h1>", unsafe_allow_html=True)
st.markdown("<div class='small-text'>Escucha, toca y comprueba si el ritmo está bien 👏</div>", unsafe_allow_html=True)

# -----------------------
# NIVELES
# -----------------------

levels = [
    {"nivel": 1, "nombre": "🥁 Negras", "img": "data/nivel1.png", "audio": "data/nivel1.wav", "tdi_threshold": 8},
    {"nivel": 2, "nombre": "🎶 Blancas", "img": "data/nivel2.png", "audio": "data/nivel2.wav", "tdi_threshold": 10},
    {"nivel": 3, "nombre": "⚡ Corcheas", "img": "data/nivel3.png", "audio": "data/nivel3.wav", "tdi_threshold": 12},
    {"nivel": 4, "nombre": "🔥 Semicorcheas", "img": "data/nivel4.png", "audio": "data/nivel4.wav", "tdi_threshold": 15},
    {"nivel": 5, "nombre": "🎯 Tresillos", "img": "data/nivel5.png", "audio": "data/nivel5.wav", "tdi_threshold": 18},
]

# -----------------------
# ESTADO
# -----------------------

if "nivel_actual" not in st.session_state:
    st.session_state.nivel_actual = 0

nivel = levels[st.session_state.nivel_actual]

st.progress((st.session_state.nivel_actual + 1) / len(levels))

# -----------------------
# FUNCIONES
# -----------------------

def safe_zscore(x):
    x = np.asarray(x)
    x[~np.isfinite(x)] = 0
    mu = np.mean(x)
    sd = np.std(x)
    if sd == 0:
        return np.zeros_like(x)
    return (x - mu) / sd


def trim_audio(y, sr):
    rms = librosa.feature.rms(y=y)[0]
    idx = np.where(rms > np.max(rms) * 0.05)[0]
    if len(idx) == 0:
        return y
    start = idx[0] * 512
    end = idx[-1] * 512
    return y[start:end]


def rhythm_flux(y, sr):
    S = np.abs(librosa.stft(y))
    flux = np.sum(np.maximum(np.diff(S, axis=1), 0), axis=0)
    flux = np.log(flux + 1e-6)
    return safe_zscore(flux)


def compute_tdi(ref_path, user_path):
    y_ref, sr = librosa.load(ref_path)
    y_usr, sr = librosa.load(user_path)

    y_ref = trim_audio(y_ref, sr)
    y_usr = trim_audio(y_usr, sr)

    s_ref = rhythm_flux(y_ref, sr)
    s_usr = rhythm_flux(y_usr, sr)

    D, wp = librosa.sequence.dtw(s_usr.reshape(1, -1), s_ref.reshape(1, -1))
    wp = wp[::-1]

    i = wp[:, 0]
    j = wp[:, 1]

    delay = np.abs(i - j)
    tdi = np.mean(delay) / len(s_ref) * 100

    return tdi

# -----------------------
# INTERFAZ
# -----------------------

st.markdown(f"<div class='level-title'>Nivel {nivel['nivel']} - {nivel['nombre']}</div>", unsafe_allow_html=True)

st.image(nivel["img"], use_container_width=True)

# 🔥 BLOQUE MEJORADO
st.markdown("### 🎧 Escucha cómo suena")
st.info("Escucha primero el ejemplo antes de grabarte 🎵")

st.audio(nivel["audio"])

if st.button("🔊 Volver a escuchar"):
    st.audio(nivel["audio"])

# -----------------------
# GRABACIÓN
# -----------------------

st.write("🎙️ ¡Ahora te toca a ti!")

audio = audiorecorder("▶️ Grabar", "⏹️ Parar")

if len(audio) > 0:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        user_audio_path = tmp.name
        audio.export(user_audio_path, format="wav")

    st.success("🎧 ¡Grabación completada!")

    st.write("🔁 Escucha tu grabación:")
    st.audio(user_audio_path)

    if st.button("✅ Evaluar mi ritmo"):
        tdi = compute_tdi(nivel["audio"], user_audio_path)

        st.write(f"**TDI:** {tdi:.2f}")

        if tdi < nivel["tdi_threshold"]:
            st.balloons()
            st.success("🎉 ¡Muy bien! Pasa al siguiente nivel")

            if st.button("🚀 Siguiente nivel"):
                if st.session_state.nivel_actual < len(levels) - 1:
                    st.session_state.nivel_actual += 1
                    st.rerun()
                else:
                    st.success("🏆 ¡Has completado todos los niveles!")
        else:
            st.warning("😅 Inténtalo otra vez")

# -----------------------
# NAVEGACIÓN
# -----------------------

col1, col2 = st.columns(2)

with col1:
    if st.button("⬅️ Atrás"):
        if st.session_state.nivel_actual > 0:
            st.session_state.nivel_actual -= 1
            st.rerun()

with col2:
    if st.button("➡️ Adelante"):
        if st.session_state.nivel_actual < len(levels) - 1:
            st.session_state.nivel_actual += 1
            st.rerun()
