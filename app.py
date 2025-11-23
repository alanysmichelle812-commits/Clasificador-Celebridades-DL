import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io

# --- 1. CONFIGURACIÓN Y MODELO ---
# Asegúrate de que este nombre sea EXACTAMENTE el mismo que usaste en tu Notebook
MODEL_PATH = 'clasificador_celebridades_v1.h5' 
IMG_SIZE = (224, 224)

# Las clases deben estar en el mismo orden que en tu Notebook
CLASS_NAMES = ['Angelina Jolie', 'Brad Pitt', 'Denzel Washington', 'Hugh Jackman', 
               'Jennifer Lawrence', 'Johnny Depp', 'Kate Winslet', 'Leonardo DiCaprio', 
               'Megan Fox', 'Natalie Portman', 'Nicole Kidman', 'Robert Downey Jr', 
               'Sandra Bullock', 'Scarlett Johansson', 'Tom Cruise', 'Tom Hanks', 'Will Smith']

# Cargar el modelo (Usa caché para cargar el modelo una sola vez, mejorando el rendimiento)
@st.cache_resource
def load_model_cached():
    """Carga el modelo solo una vez al iniciar la app."""
    return load_model(MODEL_PATH)

try:
    model = load_model_cached()
    MODEL_LOADED = True
except Exception as e:
    st.error(f"Error al cargar el modelo: {e}. Asegúrate de que el archivo '{MODEL_PATH}' esté en la misma carpeta.")
    MODEL_LOADED = False

# --- 2. FUNCIÓN DE PREDICCIÓN ---
def predict_image(image_file):
    # Cargar y redimensionar la imagen
    img = Image.open(image_file).convert('RGB')
    img = img.resize(IMG_SIZE)
    
    # Convertir a array de numpy y preprocesar
    img_array = np.array(img, dtype=np.float32) / 255.0 # Normalización (Ajusta si usas otra)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predecir
    predictions = model.predict(img_array)
    
    # Convertir logits a probabilidades (asumiendo que tu modelo usa from_logits=True)
    score = tf.nn.softmax(predictions[0])
    
    # Obtener el resultado
    predicted_index = np.argmax(score)
    confidence = 100 * np.max(score)
    predicted_class = CLASS_NAMES[predicted_index]
    
    return predicted_class, confidence, score

# --- 3. INTERFAZ DE STREAMLIT ---
if MODEL_LOADED:
    st.title("🌟 Clasificador de Celebridades DL 🧠")
    st.markdown("Carga una imagen para clasificarla entre las 17 celebridades entrenadas.")

    uploaded_file = st.file_uploader("Sube una imagen (JPG o PNG)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Mostrar la imagen
        st.image(uploaded_file, caption="Imagen Subida.", use_column_width=True)
        st.write("")
        
        # Botón de predicción
        if st.button("Clasificar Celebridad"):
            with st.spinner('Analizando y Clasificando...'):
                try:
                    predicted_class, confidence, all_scores = predict_image(uploaded_file)
                    
                    st.success(f"## 🎉 Predicción: {predicted_class}")
                    st.subheader(f"Confianza: {confidence:.2f}%")
                    
                    # Mostrar las 5 principales predicciones
                    st.markdown("---")
                    st.markdown("**Top 5 Predicciones:**")
                    
                    # Obtener los 5 índices con mayor score
                    top_5_indices = np.argsort(all_scores)[-5:][::-1]
                    top_5_scores = all_scores[top_5_indices]
                    
                    for i in range(5):
                        st.write(f"- **{CLASS_NAMES[top_5_indices[i]]}**: {100 * top_5_scores[i]:.2f}%")

                except Exception as e:
                    st.error(f"Ocurrió un error durante la predicción: {e}")