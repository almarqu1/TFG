# DistilMatch: Sistema Interpretable de Adecuación de Talento con LLMs

**Autor:** Álvaro Martínez Quilis  
**TFG - Grado en Ingeniería Informática - ETSIINF (UPV)**  
**Fecha:** Septiembre 2025

[![Python](https://img.shields.io/badge/Python-3.11-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![Hugging Face Transformers](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-PEFT-yellow)](https://huggingface.co/docs/peft/index)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit)](https://streamlit.io/)
[![DVC](https://img.shields.io/badge/DVC-Data%20Version%20Control-blue?style=flat&logo=dvc)](https://dvc.org/)

---

## 1. Visión General del Proyecto

**DistilMatch** es un sistema avanzado de apoyo a la decisión para procesos de selección de personal, desarrollado como Trabajo de Fin de Grado. El proyecto aborda las limitaciones de los *Applicant Tracking Systems* (ATS) tradicionales, que se basan en la coincidencia de palabras clave, proponiendo una solución fundamentada en la **comprensión semántica profunda** del lenguaje natural.

El sistema utiliza un Gran Modelo de Lenguaje (LLM) especializado para evaluar la idoneidad entre un currículum y una oferta de empleo, proporcionando no solo una puntuación numérica, sino también una **justificación razonada** de su decisión, alineándose con los principios de la IA Explicable (XAI).

Este repositorio contiene todo el código, los datos, el modelo entrenado y una demo interactiva para explorar y reproducir la investigación.

## 2. Características Clave

La arquitectura de DistilMatch se basa en tecnologías de vanguardia para lograr un equilibrio entre rendimiento y eficiencia:

*   **🧠 Destilación de Conocimiento (Knowledge Distillation):** Un modelo *Teacher* de frontera (**Google Gemini 2.5 Pro**) transfiere su capacidad de razonamiento a un modelo *Student* mucho más ligero y eficiente (**Qwen3-4B-Instruct**).
*   **⚡ Ajuste Fino Eficiente (PEFT):** Se utiliza **LoRA (Low-Rank Adaptation)** junto con **cuantización de 4-bits** para entrenar el modelo *Student* en hardware de consumo (una única GPU con <16GB de VRAM), haciendo la investigación accesible y reproducible.
*   **🔍 Explicabilidad Inherente (Generative XAI):** A diferencia de métodos post-hoc, el modelo genera explicaciones textuales (fortalezas, debilidades, veredicto) como parte de su tarea principal, ofreciendo una transparencia total en su toma de decisiones.
*   **💡 El Descubrimiento Central: "Explanation-Tuning":** La investigación demostró que entrenar al modelo para replicar el **razonamiento completo** del *Teacher* (no solo su puntuación final) es significativamente más efectivo, mejorando todas las métricas de evaluación.
*   **🧪 Demo Interactiva:** Una aplicación desarrollada en **Streamlit** permite a cualquier usuario interactuar con el modelo final, probar sus capacidades con diferentes perfiles y ofertas, y visualizar su rendimiento en tiempo real.

## 3. Demo Interactiva en Streamlit

Para una exploración práctica del sistema, puedes ejecutar la demo localmente. La aplicación permite buscar en los datasets, seleccionar una oferta y una lista de candidatos, y obtener un ranking de compatibilidad detallado y justificado.

Para lanzar la aplicación:
```bash
# Asegúrate de tener el entorno virtual activado y las dependencias instaladas
streamlit run app.py
```

## 4. Estructura del Repositorio

El proyecto sigue una estructura MLOps para garantizar la modularidad y reproducibilidad:
```
TFG_DistilMatch/
├── config/              # Ficheros de configuración (parámetros, prompts)
├── data/                # Datos del proyecto (gestionados por DVC)
├── models/              # Adaptadores LoRA de cada exp. (gestionados por DVC)
├── notebooks/           # Notebooks para exploración y análisis cualitativo
├── outputs/             # Informes de evaluación, ejemplos de explicabilidad
├── scripts/             # Scripts ejecutables (generación de datos, evaluación)
├── src/                 # Código fuente principal (lógica de entrenamiento, utils)
├── app.py               # Código de la aplicación de Streamlit
├── .dvc/                # Metadatos de DVC
├── dvc.yaml             # Definición del pipeline de MLOps
└── requirements.txt     # Dependencias de Python
```

## 5. Instalación y Uso

### 5.1. Prerrequisitos
*   Python 3.10 o superior
*   Git y Git LFS
*   DVC (`pip install "dvc[gdrive]"` para usar Google Drive como remoto)

### 5.2. Pasos de Instalación
1.  **Clona el repositorio:**
    ```bash
    git clone https://github.com/almarqu1/TFG
    cd TFG_DistilMatch
    ```

2.  **Crea y activa un entorno virtual:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # macOS/Linux
    # .\.venv\Scripts\activate  # Windows
    ```

3.  **Instala las dependencias:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Descarga los datos y modelos con DVC:**
    ```bash
    dvc pull
    ```
    > NOTA: Por el momento, se ha optado por no emplear DVC. Los adaptadores se encuentran en la carpeta "distilmatch-v4-adapters" del repositorio.

5.  **Descarga los datasets de Kaggle:**
    *https://www.kaggle.com/datasets/saugataroyarghya/resume-dataset (Dataset CVs)
    *https://www.kaggle.com/datasets/arshkon/linkedin-job-postings
    > NOTA: asegúrate de que tienes los archivos descargados en data/00_raw (necesitarás crear la carpeta) o de que tu config.yaml refleja la ubicación de los datasets.

## 6. Resultados Clave

La estrategia de **Explanation-Tuning** (modelo `v4`) demostró una mejora sustancial sobre la capacidad *zero-shot* del modelo base, validando el éxito de la destilación de conocimiento.

| Métrica de Evaluación | Baseline (Student Zero-Shot) | Modelo Final (v4) | Mejora Relativa |
| :--- | :---: | :---: | :---: |
| **Error Absoluto Medio (MAE) ↓** | 27.16 | **19.98** | -26.4% |
| **Correlación de Spearman ($\rho$) ↑** | 0.388 | **0.489** | +26.2% |
| **Exactitud Categórica ↑** | 31.8% | **56.1%** | +76.4% |

La mejora en la **Correlación de Spearman** es el resultado más importante, ya que indica que el modelo final es significativamente mejor para **ordenar correctamente a los candidatos** de más a menos idóneo, que es el objetivo principal de un sistema de preselección.

## 7. Limitaciones y Trabajo Futuro
*   **Limitaciones:** El `Golden Set` fue anotado por un único experto. Los datasets están en inglés y centrados en el mercado de EEUU. Existe un riesgo potencial de heredar sesgos del modelo *Teacher*.
*   **Trabajo Futuro:**
    1.  **Auditoría y Mitigación de Sesgos (Fairness).**
    2.  **Implementación de un ciclo de Aprendizaje Activo (*Human-in-the-loop*).**
    3.  **Especialización por Dominios** mediante diferentes adaptadores LoRA.
    4.  **Expansión a arquitecturas multimodales** (ej. análisis de repositorios de GitHub).
    5.  **Expansión de Golden Set y Silver Set con más datos.**