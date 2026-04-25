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

st.markdown("""
<style>
.main {background-color: #FFFDF7;}
h1 {text-align: center; color: #FF6B6B;}
.level-title {
    font-size: 30px;
    text-align: center;
    margin-bottom: 10px;
    font-weight: 800;
    color: #333333;
}
.small-text {text-align: center; color: #666666; font-size: 16px;}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>🎵 Juego de Ritmo 🎵</h1>", unsafe_allow_html=True)
st.markdown(
    "<div class='small-text'>Escucha, toca y comprueba si tu ritmo está bien 👏</div>",
    unsafe_allow_html=True
)

# -----------------------
# RUTAS BASE
# -----------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# -----------------------
# NIVELES
# -----------------------

levels = [
    {"nivel": 1, "nombre": "Negras", "img": os.path.join(BASE_DIR, "data", "nivel1.png"), "audio": os.path.join(BASE_DIR, "data", "nivel1.wav"), "tdi_threshold": 0.087},
    {"nivel": 2, "nombre": "Corcheas", "img": os.path.join(BASE_DIR, "data", "nivel2.png"), "audio": os.path.join(BASE_DIR, "data", "nivel2.wav"), "tdi_threshold": 0.095},
    {"nivel": 3, "nombre": "Puntillo", "img": os.path.join(BASE_DIR, "data", "nivel3.png"), "audio": os.path.join(BASE_DIR, "data", "nivel3.wav"), "tdi_threshold": 0.110},
    {"nivel": 4, "nombre": "Semicorcheas", "img": os.path.join(BASE_DIR, "data", "nivel4.png"), "audio": os.path.join(BASE_DIR, "data", "nivel4.wav"), "tdi_threshold": 0.130},
    {"nivel": 5, "nombre": "Tresillos", "img": os.path.join(BASE_DIR, "data", "nivel5.png"), "audio": os.path.join(BASE_DIR, "data", "nivel5.wav"), "tdi_threshold": 0.150},
]

# -----------------------
# ESTADO
# -----------------------

if "nivel_actual" not in st.session_state:
    st.session_state.nivel_actual = 0

nivel = levels[st.session_state.nivel_actual]

st.progress((st.session_state.nivel_actual + 1) / len(levels))

# -----------------------
# AUDIO PLAYER
# -----------------------

def audio_player(path):
    with open(path, "rb") as f:
        audio_bytes = f.read()

    audio_base64 = base64.b64encode(audio_bytes).decode()

    components.html(
        f"""
        <div style="width: 100%;">
            <audio id="audio_ejemplo" controls style="width: 100%;">
                <source src="data:audio/wav;base64,{audio_base64}" type="audio/wav">
            </audio>

            <button 
                onclick="
                    var audio = document.getElementById('audio_ejemplo');
                    audio.currentTime = 0;
                    audio.play();
                "
                style="
                    margin-top: 12px;
                    padding: 8px 16px;
                    border-radius: 10px;
                    border: 1px solid #d0d0d0;
                    border-bottom: 3px solid #bdbdbd;
                    background-color: white;
                    cursor: pointer;
                    font-size: 15px;
                    box-shadow: 0 1px 2px rgba(0,0,0,0.05);
                "
            >
                🔊 Volver a escuchar
            </button>
        </div>
        """,
        height=120
    )

# -----------------------
# FUNCIONES AUDIO
# -----------------------

def reset_grabacion():
    for key in list(st.session_state.keys()):
        if key.startswith("audio_recorder"):
            del st.session_state[key]


def safe_zscore(x):
    x = np.asarray(x, dtype=float)
    x[~np.isfinite(x)] = 0.0
    mu = np.mean(x)
    sd = np.std(x)
    if sd == 0:
        return np.zeros_like(x)
    return (x - mu) / sd


def trim_audio(y, sr, hop_length=512, threshold_ratio=0.05):
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    if len(rms) == 0 or np.max(rms) == 0:
        return y

    idx = np.where(rms > np.max(rms) * threshold_ratio)[0]
    if len(idx) == 0:
        return y

    start = int(idx[0] * hop_length)
    end = int(min(len(y), idx[-1] * hop_length + hop_length))

    if start >= end:
        return y

    return y[start:end]


def rhythm_flux(y, sr, n_fft=2048, hop_length=512):
    S = np.abs(librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length))
    if S.shape[1] < 2:
        return np.zeros(1)

    flux = np.sum(np.maximum(np.diff(S, axis=1), 0), axis=0)
    flux = np.log(flux + 1e-6)
    flux = safe_zscore(flux)

    return flux if len(flux) > 0 else np.zeros(1)


def compute_tdi_metrics(ref_path, user_path):
    y_ref, sr_ref = librosa.load(ref_path, sr=None)
    y_usr, _ = librosa.load(user_path, sr=sr_ref)

    y_ref = trim_audio(y_ref, sr_ref)
    y_usr = trim_audio(y_usr, sr_ref)

    s_ref = rhythm_flux(y_ref, sr_ref)
    s_usr = rhythm_flux(y_usr, sr_ref)

    if len(s_ref) < 2 or len(s_usr) < 2:
        return {"tdi_norm": np.nan}

    _, wp = librosa.sequence.dtw(
        X=s_usr.reshape(1, -1),
        Y=s_ref.reshape(1, -1)
    )

    wp = wp[::-1]
    d = wp[:, 1] - wp[:, 0]
    tdi_norm = float(np.sum(np.abs(d)) / (len(wp) * len(s_ref)))

    return {"tdi_norm": tdi_norm}

# -----------------------
# INTERFAZ
# -----------------------

st.markdown(
    f"<div class='level-title'>Nivel {nivel['nivel']} - {nivel['nombre']}</div>",
    unsafe_allow_html=True
)

st.image(nivel["img"], use_container_width=True)

st.markdown("### 🎧 Escucha cómo suena")
st.info("Escucha primero el ejemplo antes de grabarte 🎵")

audio_player(nivel["audio"])

# -----------------------
# GRABACIÓN
# -----------------------

st.write("🎙️ ¡Ahora te toca a ti!")

audio = audiorecorder(
    "▶️ Grabar",
    "⏹️ Parar",
    key=f"audio_recorder_nivel_{st.session_state.nivel_actual}"
)

if len(audio) > 0:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        user_audio_path = tmp.name
        audio.export(user_audio_path, format="wav")

    st.success("🎧 ¡Grabación completada!")
    st.write("🔁 Escucha tu grabación:")
    st.audio(user_audio_path)

    if st.button("✅ Evaluar mi ritmo"):
        metrics = compute_tdi_metrics(nivel["audio"], user_audio_path)

        if not np.isfinite(metrics["tdi_norm"]):
            st.error("No se ha podido analizar correctamente la grabación.")
        else:
            if metrics["tdi_norm"] <= nivel["tdi_threshold"]:
                st.balloons()
                st.success("🎉 ¡Muy bien! Tu ritmo está dentro del rango admisible.")

                if st.button("🚀 Siguiente nivel"):
                    if st.session_state.nivel_actual < len(levels) - 1:
                        st.session_state.nivel_actual += 1
                        reset_grabacion()
                        st.rerun()
                    else:
                        st.success("🏆 ¡Has completado todos los niveles!")
            else:
                st.warning("😅 ¡Casi lo tienes! El ritmo no está del todo bien todavía. Inténtalo otra vez 👏")

# -----------------------
# NAVEGACIÓN
# -----------------------

col1, col2 = st.columns(2)

with col1:
    if st.button("⬅️ Atrás"):
        if st.session_state.nivel_actual > 0:
            st.session_state.nivel_actual -= 1
            reset_grabacion()
            st.rerun()

with col2:
    if st.button("➡️ Adelante"):
        if st.session_state.nivel_actual < len(levels) - 1:
            st.session_state.nivel_actual += 1
            reset_grabacion()
            st.rerun()
