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
.metric-box {
    padding: 12px;
    border-radius: 12px;
    background-color: #F8F8F8;
    text-align: center;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>🎵 Juego de Ritmo 🎵</h1>", unsafe_allow_html=True)
st.markdown(
    "<div class='small-text'>Escucha, toca y comprueba si tu ritmo está dentro de la variabilidad temporal admisible 👏</div>",
    unsafe_allow_html=True
)

# -----------------------
# RUTAS BASE
# -----------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# -----------------------
# NIVELES
# -----------------------
# Umbral base del TFG: tau = 0.087
# Se relaja ligeramente según la complejidad rítmica del nivel

levels = [
    {
        "nivel": 1,
        "nombre": "🥁 Negras",
        "img": os.path.join(BASE_DIR, "data", "nivel1.png"),
        "audio": os.path.join(BASE_DIR, "data", "nivel1.wav"),
        "tdi_threshold": 0.087
    },
    {
        "nivel": 2,
        "nombre": "🎶 Corcheas",
        "img": os.path.join(BASE_DIR, "data", "nivel2.png"),
        "audio": os.path.join(BASE_DIR, "data", "nivel2.wav"),
        "tdi_threshold": 0.095
    },
    {
        "nivel": 3,
        "nombre": "⚡ Puntillo",
        "img": os.path.join(BASE_DIR, "data", "nivel3.png"),
        "audio": os.path.join(BASE_DIR, "data", "nivel3.wav"),
        "tdi_threshold": 0.110
    },
    {
        "nivel": 4,
        "nombre": "🔥 Semicorcheas",
        "img": os.path.join(BASE_DIR, "data", "nivel4.png"),
        "audio": os.path.join(BASE_DIR, "data", "nivel4.wav"),
        "tdi_threshold": 0.130
    },
    {
        "nivel": 5,
        "nombre": "🎯 Tresillos",
        "img": os.path.join(BASE_DIR, "data", "nivel5.png"),
        "audio": os.path.join(BASE_DIR, "data", "nivel5.wav"),
        "tdi_threshold": 0.150
    },
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
    # Representación rítmica ligera basada en flujo espectral
    S = np.abs(librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length))
    if S.shape[1] < 2:
        return np.zeros(1)

    flux = np.sum(np.maximum(np.diff(S, axis=1), 0), axis=0)
    flux = np.log(flux + 1e-6)
    flux = safe_zscore(flux)

    if len(flux) == 0:
        return np.zeros(1)

    return flux


def compute_tdi_metrics(ref_path, user_path):
    # Cargamos ambos audios con la misma frecuencia de muestreo
    y_ref, sr_ref = librosa.load(ref_path, sr=None)
    y_usr, _ = librosa.load(user_path, sr=sr_ref)

    # Recorte básico para eliminar silencios exteriores
    y_ref = trim_audio(y_ref, sr_ref)
    y_usr = trim_audio(y_usr, sr_ref)

    # Representación rítmica
    s_ref = rhythm_flux(y_ref, sr_ref)
    s_usr = rhythm_flux(y_usr, sr_ref)

    # Comprobaciones de seguridad
    if len(s_ref) < 2 or len(s_usr) < 2:
        return {
            "tdi": np.nan,
            "tdi_norm": np.nan,
            "mean_delay": np.nan,
            "max_delay": np.nan
        }

    # DTW
    _, wp = librosa.sequence.dtw(
        X=s_usr.reshape(1, -1),
        Y=s_ref.reshape(1, -1)
    )
    wp = wp[::-1]

    # Índices del camino
    i = wp[:, 0]
    j = wp[:, 1]

    # Retardo local: d_k = j_k - i_k
    d = j - i
    abs_d = np.abs(d)

    # Métricas
    mean_delay = float(np.mean(abs_d))
    max_delay = float(np.max(abs_d))

    # TDI como deformación acumulada
    tdi = float(np.sum(abs_d))

    # TDI normalizado
    # Se normaliza por la longitud del camino y de la referencia
    tdi_norm = float(tdi / (len(wp) * len(s_ref)))

    return {
        "tdi": tdi,
        "tdi_norm": tdi_norm,
        "mean_delay": mean_delay,
        "max_delay": max_delay
    }


def format_metric(value, decimals=4):
    if value is None or not np.isfinite(value):
        return "No disponible"
    return f"{value:.{decimals}f}"


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
        metrics = compute_tdi_metrics(nivel["audio"], user_audio_path)

        if not np.isfinite(metrics["tdi_norm"]):
            st.error("No se ha podido analizar correctamente la grabación. Intenta grabar de nuevo.")
        else:
            st.markdown("### 📊 Resultado del análisis")

            colm1, colm2 = st.columns(2)
            with colm1:
                st.markdown(
                    f"<div class='metric-box'><b>TDI</b><br>{format_metric(metrics['tdi'], 2)}</div>",
                    unsafe_allow_html=True
                )
                st.markdown(
                    f"<div class='metric-box'><b>Retardo medio</b><br>{format_metric(metrics['mean_delay'], 2)}</div>",
                    unsafe_allow_html=True
                )
            with colm2:
                st.markdown(
                    f"<div class='metric-box'><b>TDI normalizado</b><br>{format_metric(metrics['tdi_norm'], 4)}</div>",
                    unsafe_allow_html=True
                )
                st.markdown(
                    f"<div class='metric-box'><b>Retardo máximo</b><br>{format_metric(metrics['max_delay'], 2)}</div>",
                    unsafe_allow_html=True
                )

            st.write(f"**Umbral del nivel:** {nivel['tdi_threshold']:.3f}")

            if metrics["tdi_norm"] <= nivel["tdi_threshold"]:
                st.balloons()
                st.success("🎉 ¡Muy bien! Tu ritmo está dentro del rango admisible de este nivel.")

                if st.button("🚀 Siguiente nivel"):
                    if st.session_state.nivel_actual < len(levels) - 1:
                        st.session_state.nivel_actual += 1
                        st.rerun()
                    else:
                        st.success("🏆 ¡Has completado todos los niveles!")
            else:
                st.warning("😅 Hay una desviación rítmica mayor que la admisible en este nivel. Inténtalo otra vez.")

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
