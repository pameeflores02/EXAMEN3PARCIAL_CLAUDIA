import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

st.set_page_config(
    page_title="Clasificador de Imágenes - IA en la Nube",
    page_icon="🧠",
    layout="centered",
)

NOMBRE_AUTOR = "CLAUDIA AGUILAR"

CLASES = ['avión', 'auto', 'pájaro', 'gato', 'ciervo',
          'perro', 'rana', 'caballo', 'barco', 'camión']

ICONOS = {
    'avión': '✈️', 'auto': '🚗', 'pájaro': '🐦', 'gato': '🐱', 'ciervo': '🦌',
    'perro': '🐶', 'rana': '🐸', 'caballo': '🐴', 'barco': '🚢', 'camión': '🚚'
}


st.markdown("""
<style>
    /* Fondo general */
    .stApp {
        background: linear-gradient(180deg, #0f1220 0%, #1a1f38 100%);
    }

    /* Encabezado con gradiente */
    .header-box {
        background: linear-gradient(135deg, #7b2ff7 0%, #4361ee 50%, #4cc9f0 100%);
        padding: 2.2rem 1.5rem;
        border-radius: 18px;
        text-align: center;
        margin-bottom: 1.8rem;
        box-shadow: 0 8px 24px rgba(67, 97, 238, 0.35);
    }
    .header-box h1 {
        color: white;
        font-size: 2.1rem;
        margin: 0;
        font-weight: 800;
    }
    .header-box p {
        color: rgba(255,255,255,0.9);
        margin-top: 0.4rem;
        font-size: 1rem;
    }

    /* Tarjetas generales */
    .card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 16px;
        padding: 1.3rem 1.5rem;
        margin-bottom: 1.2rem;
    }

    /* Tarjeta de resultado */
    .result-card {
        background: linear-gradient(135deg, rgba(67,97,238,0.25), rgba(76,201,240,0.15));
        border: 1px solid rgba(76,201,240,0.4);
        border-radius: 18px;
        padding: 1.6rem;
        text-align: center;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .result-card .icon {
        font-size: 3.2rem;
    }
    .result-card .clase {
        font-size: 1.8rem;
        font-weight: 800;
        color: #ffffff;
        margin: 0.3rem 0;
    }
    .result-card .confianza {
        font-size: 1.05rem;
        color: #a8f0ff;
        font-weight: 600;
    }

    /* Etiquetas de clases (chips) */
    .chip {
        display: inline-block;
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.15);
        border-radius: 999px;
        padding: 0.35rem 0.9rem;
        margin: 0.2rem;
        font-size: 0.85rem;
        color: #e0e0e0;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: rgba(255,255,255,0.4);
        font-size: 0.85rem;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid rgba(255,255,255,0.08);
    }

    /* Textos generales más legibles sobre fondo oscuro */
    p, label, .stMarkdown, .stRadio label {
        color: #e6e6e6 !important;
    }
</style>
""", unsafe_allow_html=True)


st.markdown(f"""
<div class="header-box">
    <h1>🧠 Clasificador de Objetos con IA</h1>
    <p>Proyecto de Computación en la Nube · por CLAUDIA AGUILAR </p>
</div>
""", unsafe_allow_html=True)


st.markdown("""
<div class="card">
    <b>📋 ¿Cómo funciona?</b><br><br>
    Sube una imagen o toma una foto con tu cámara. El modelo la analizará
    y te dirá a cuál de estas 10 categorías pertenece:
</div>
""", unsafe_allow_html=True)

chips_html = " ".join([f'<span class="chip">{ICONOS[c]} {c.capitalize()}</span>' for c in CLASES])
st.markdown(f'<div style="margin-bottom:1.5rem;">{chips_html}</div>', unsafe_allow_html=True)

# -----------------------------
# CARGAR EL MODELO
# -----------------------------
@st.cache_resource
def cargar_modelo():
    return tf.keras.models.load_model("modelo_cifar10.keras")

modelo = cargar_modelo()

# -----------------------------
# ENTRADA DE IMAGEN
# -----------------------------
st.markdown("### 📸 Elige tu imagen")
opcion = st.radio(
    "Elige cómo quieres dar la imagen:",
    ["Subir imagen", "Tomar foto"],
    horizontal=True,
    label_visibility="collapsed",
)

imagen_subida = None
if opcion == "Subir imagen":
    imagen_subida = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])
else:
    imagen_subida = st.camera_input("Toma una foto")

# -----------------------------
# PROCESAR Y PREDECIR
# -----------------------------
if imagen_subida is not None:
    imagen = Image.open(imagen_subida).convert("RGB")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(imagen, caption="Imagen cargada", use_container_width=True)

    imagen_procesada = imagen.resize((32, 32))
    arreglo = np.array(imagen_procesada).astype("float32") / 255.0
    arreglo = np.expand_dims(arreglo, axis=0)

    with st.spinner("🔎 Analizando imagen..."):
        prediccion = modelo.predict(arreglo, verbose=0)

    indice_predicho = int(np.argmax(prediccion[0]))
    clase_predicha = CLASES[indice_predicho]
    confianza = float(np.max(prediccion[0]))

    with col2:
        st.markdown(f"""
        <div class="result-card">
            <div class="icon">{ICONOS[clase_predicha]}</div>
            <div class="clase">{clase_predicha.capitalize()}</div>
            <div class="confianza">Confianza: {confianza:.1%}</div>
        </div>
        """, unsafe_allow_html=True)
        st.progress(float(confianza))

    with st.expander("📊 Ver todas las probabilidades"):
        for clase, prob in sorted(zip(CLASES, prediccion[0]), key=lambda x: -x[1]):
            st.write(f"{ICONOS[clase]} **{clase.capitalize()}** — {prob:.2%}")
            st.progress(float(prob))

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("""
<div class="footer">
    Modelo CNN entrenado en Google Colab con el dataset CIFAR-10 🚀
</div>
""", unsafe_allow_html=True)
