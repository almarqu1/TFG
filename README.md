# DistilMatch: Sistema de Scoring Interpretable para Matching de Talento

**Autor:** [Tu Nombre Completo]  
**TFG - [Nombre de tu Titulación] - [Nombre de tu Universidad]**  
**Fecha:** [Mes y Año de Finalización]

[![Python](https://img.shields.io/badge/Python-3.11-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![Hugging Face Transformers](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Transformers-yellow)](https://huggingface.co/transformers/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![DVC](https://img.shields.io/badge/DVC-Data%20Version%20Control-blue?style=flat&logo=dvc)](https://dvc.org/)
[![SHAP](https://img.shields.io/badge/SHAP-Explainable%20AI-green)](https://shap.readthedocs.io/en/latest/)

---

## 1. Introducción

**DistilMatch** es un proyecto de fin de grado que presenta un framework para la asignación automática y el scoring de compatibilidad entre candidatos (CVs) y ofertas de trabajo en el mercado europeo. El sistema está diseñado para ser:

*   **Cuantitativo:** Proporciona un score numérico de compatibilidad (0-100).
*   **Interpretable:** Ofrece explicaciones claras sobre por qué se asigna un determinado score, destacando las palabras y frases clave.
*   **Adaptativo:** Utiliza un bucle de **Active Learning** para mejorar continuamente con un mínimo esfuerzo de etiquetado humano.
*   **Eficiente:** Emplea **Knowledge Distillation** para transferir el conocimiento de un modelo grande (GPT-4) a un modelo de producción ligero y rápido (BERT).

Este repositorio contiene todo el código fuente, los datos versionados, los modelos entrenados y la documentación necesaria para reproducir los experimentos y resultados de la memoria del TFG.

## 2. Metodología

El core de DistilMatch se basa en un pipeline de **Teacher-Student** con un humano en el bucle (*human-in-the-loop*):

1.  **Etiquetado Humano (Gold Standard):** Un conjunto de datos de ~360 pares (CV, oferta) es anotado por expertos humanos usando una escala ordinal de 5 categorías para establecer la "verdad absoluta".
2.  **Modelo Teacher (GPT-4):** Un Large Language Model (LLM) actúa como un "anotador experto" para generar *soft labels* (scores numéricos) a gran escala sobre el resto del dataset.
3.  **Modelo Student (BERT):** Un modelo `bert-base-multilingual-cased` es entrenado para replicar los scores del Teacher y anclarse a los datos Gold Standard, resultando en un modelo rápido y preciso.
4.  **Active Learning:** El modelo Student identifica los pares más ambiguos para que sean etiquetados por un humano, optimizando el ciclo de mejora continua.
5.  **Explicabilidad (XAI):** Se utiliza la librería `shap` para generar explicaciones a nivel de token, proporcionando transparencia en las decisiones del modelo.

## 3. Estructura del Repositorio

```
TFG_DistilMatch/
├── data/              # Datos del proyecto (gestionados por DVC)
├── models/            # Modelos entrenados (gestionados por DVC)
├── notebooks/         # Jupyter Notebooks para exploración
├── outputs/           # Reportes de métricas y explicaciones
├── report/            # Documento de la memoria y presentación
├── src/               # Código fuente principal del proyecto
├── .dvcignore         # Archivos ignorados por DVC
├── .gitignore         # Archivos ignorados por Git
├── dvc.yaml           # Definición del pipeline de MLOps
├── Dockerfile         # Receta para construir la imagen de la aplicación
├── README.md          # Este archivo
└── requirements.txt   # Dependencias de Python
```

## 4. Instalación y Configuración

Para poner en marcha este proyecto en tu máquina local, sigue estos pasos.

### 4.1. Prerrequisitos

*   Python 3.10 o superior
*   Git
*   DVC (`pip install dvc`)
*   (Opcional, para DVC) Configurar un almacenamiento remoto (S3, GCS, etc.). Sigue la [guía de DVC](https://dvc.org/doc/command-reference/remote/add).

### 4.2. Pasos de Instalación

1.  **Clona el repositorio:**
    ```bash
    git clone [URL-DE-TU-REPOSITORIO]
    cd TFG_DistilMatch
    ```

2.  **Crea y activa un entorno virtual:**
    ```bash
    python -m venv .venv
    # En Windows:
    .\.venv\Scripts\activate
    # En macOS/Linux:
    source .venv/bin/activate
    ```

3.  **Instala las dependencias:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Descarga los datos y modelos versionados con DVC:**
    ```bash
    dvc pull
    ```
    Este comando descargará los datasets de `data/` y los modelos de `models/` desde el almacenamiento remoto configurado.

## 5. Uso y Reproducción de Experimentos

El pipeline completo está definido en el archivo `dvc.yaml`. Puedes reproducir todo el experimento con un solo comando.

### 5.1. Reproducir el Pipeline Completo

Este comando ejecutará todas las etapas definidas en `dvc.yaml`: pre-procesamiento de datos, entrenamiento del modelo, y evaluación. DVC se encargará de ejecutar solo las partes que hayan cambiado desde la última ejecución.

```bash
dvc repro
```

### 5.2. Ejecutar Etapas Individualmente

Puedes ejecutar scripts individuales si lo necesitas. Asegúrate de que tu entorno virtual esté activado.

*   **Entrenar el modelo:**
    ```bash
    python src/train.py
    ```

*   **Evaluar el modelo:**
    ```bash
    python src/evaluate.py
    ```

*   **Obtener un score para un par (CV, Oferta):**
    ```bash
    python src/score.py --cv_path "ruta/a/un/cv.txt" --oferta_path "ruta/a/una/oferta.txt"
    ```

## 6. Resultados Clave

El modelo DistilMatch ha sido evaluado rigurosamente contra un baseline de similitud por coseno, demostrando una mejora significativa en todas las métricas clave.

| Métrica | Baseline (Coseno) | DistilMatch (v1) |
| :--- | :--- | :--- |
| **Pearson r** | 0.45 | **0.82** |
| **MAE** | 25.8 | **12.3** |
| **Accuracy Cat.** | 41% | **75%** |
| **MAP@10** | 0.38 | **0.65** |

*(Nota: Estos son valores de ejemplo. Reemplázalos con tus resultados finales.)*

## 7. Limitaciones y Trabajo Futuro

*   **Limitaciones:** El dataset está fechado en 2004, lo que puede no reflejar el mercado laboral actual. El análisis de fairness se basa en inferencias demográficas que pueden contener errores.
*   **Trabajo Futuro:** Explorar arquitecturas de modelos *student* más avanzadas (ej. DeBERTa), integrar un feedback de usuario más granular en el bucle de Active Learning y expandir el dataset con datos más recientes y de más sectores.