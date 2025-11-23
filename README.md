# 🌟 Clasificador de Celebridades con Deep Learning 🧠

### Estado del Despliegue
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Space-blue)](https://huggingface.co/spaces/Alanysss/clasificador-celebridades-final)

---

## 🚀 Descripción General del Proyecto

Este repositorio contiene el código fuente y el modelo entrenado (`clasificador_celebridades_v1.h5`) para un clasificador de imágenes de celebridades desarrollado con Deep Learning. El proyecto utiliza una arquitectura de Transferencia de Aprendizaje con TensorFlow/Keras para intentar identificar 17 figuras públicas distintas a partir de imágenes.

El modelo ha sido encapsulado en una aplicación web interactiva utilizando Gradio y desplegado en Hugging Face Spaces, permitiendo la clasificación en tiempo real con solo subir una foto.

## ✨ Características Principales

* **Clasificación Multiclase:** Identifica una de las 17 celebridades entrenadas.
* **Transferencia de Aprendizaje:** Utiliza una red preentrenada como base para la extracción de características.
* **Interfaz Interactiva (Gradio):** Despliegue público y funcional (ver el enlace de la insignia arriba).

## 🛠️ Estructura del Repositorio

| Archivo/Carpeta | Propósito |
| :--- | :--- |
| `app.py` | Contiene la lógica de la aplicación web **Gradio** (interfaz y función de predicción). |
| `clasificador_celebridades_v1.h5` | **El modelo de red neuronal entrenado** (pesos y arquitectura). |
| `requirements.txt` | Lista de dependencias necesarias (`tensorflow`, `gradio`, `numpy`, `Pillow`). |
| `README.md` | Este archivo, con la documentación del proyecto. |


---

**Desarrollado por:** Alanys Ortega
