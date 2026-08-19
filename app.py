
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

st.set_page_config(page_title="Clasificador de Imágenes - IA en la Nube", page_icon="🧠")

NOMBRE_AUTOR = "CLAUDIA AGUILAR" 

st.title("🧠Clasificador de Objetos con IA")
st.caption(f"Proyecto de Computación en la Nube — por Claudia Aguilar")

st.write(
    "Sube una imagen o toma una foto con tu cámara. "
    "El modelo identificará si es un avión, auto, pájaro, gato, ciervo, "
    "perro, rana, caballo, barco o camión."
)

CLASES = ['avión', 'auto', 'pájaro', 'gato', 'ciervo',
          'perro', 'rana', 'caballo', 'barco', 'camión']


@st.cache_resource
def cargar_modelo():
    modelo = tf.keras.models.load_model("modelo_cifar10.keras")
    return modelo


modelo = cargar_modelo()

opcion = st.radio("Elige cómo quieres dar la imagen:", ["Subir imagen", "Tomar foto"])

imagen_subida = None
if opcion == "Subir imagen":
    imagen_subida = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])
else:
    imagen_subida = st.camera_input("Toma una foto")

if imagen_subida is not None:
    imagen = Image.open(imagen_subida).convert("RGB")
    st.image(imagen, caption="Imagen cargada", use_container_width=True)

 
    imagen_procesada = imagen.resize((32, 32))
    arreglo = np.array(imagen_procesada).astype("float32") / 255.0
    arreglo = np.expand_dims(arreglo, axis=0)  # (1, 32, 32, 3)

    with st.spinner("Analizando imagen..."):
        prediccion = modelo.predict(arreglo)

    indice_predicho = int(np.argmax(prediccion[0]))
    clase_predicha = CLASES[indice_predicho]
    confianza = float(np.max(prediccion[0]))

    st.success(f"### Predicción: **{clase_predicha.capitalize()}**")
    st.write(f"Confianza: **{confianza:.2%}**")

    with st.expander("Ver todas las probabilidades"):
        for clase, prob in sorted(zip(CLASES, prediccion[0]), key=lambda x: -x[1]):
            st.write(f"{clase.capitalize()}: {prob:.2%}")
            st.progress(float(prob))

st.markdown("---")
st.caption("Modelo CNN entrenado en Google Colab con el dataset CIFAR-10.")
