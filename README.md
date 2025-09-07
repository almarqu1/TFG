DistilMatch: Sistema Interpretable de Adecuación de Talento con LLMs
Autor: Álvaro Martínez Quilis
TFG - Grado en Ingeniería Informática - ETSINF (UPV)
Fecha: Septiembre 2025

Python
Hugging Face Transformers
PyTorch
Streamlit
DVC

1. Visión General del Proyecto
DistilMatch es un sistema avanzado de apoyo a la decisión para procesos de selección de personal, desarrollado como Trabajo de Fin de Grado. El proyecto aborda las limitaciones de los Applicant Tracking Systems (ATS) tradicionales, que se basan en la coincidencia de palabras clave, proponiendo una solución fundamentada en la comprensión semántica profunda del lenguaje natural.

El sistema utiliza un Gran Modelo de Lenguaje (LLM) especializado para evaluar la idoneidad entre un currículum y una oferta de empleo, proporcionando no solo una puntuación numérica, sino también una justificación razonada de su decisión, alineándose con los principios de la IA Explicable (XAI).

Este repositorio contiene todo el código, los datos, los modelos entrenados y una demo interactiva para explorar y reproducir la investigación.

2. Características Clave
La arquitectura de DistilMatch se basa en tecnologías de vanguardia para lograr un equilibrio entre rendimiento y eficiencia:

🧠 Destilación de Conocimiento (Knowledge Distillation): Un modelo Teacher de frontera (Google Gemini 2.5 Pro) transfiere su capacidad de razonamiento a un modelo Student mucho más ligero y eficiente (Qwen3-4B-Instruct).
⚡ Ajuste Fino Eficiente (PEFT): Se utiliza LoRA (Low-Rank Adaptation) junto con cuantización de 4-bits para entrenar el modelo Student en hardware de consumo (una única GPU con <16GB de VRAM), haciendo la investigación accesible y reproducible.
🔍 Explicabilidad Inherente (Generative XAI): A diferencia de métodos post-hoc como SHAP, el modelo genera explicaciones textuales (fortalezas, debilidades, potencial) como parte de su tarea principal, ofreciendo una transparencia total en su toma de decisiones.
💡 El Descubrimiento Central: "Explanation-Tuning": La investigación demostró que entrenar al modelo para replicar el razonamiento completo del Teacher (no solo su puntuación final) es significativamente más efectivo, mejorando todas las métricas de evaluación.
🧪 Demo Interactiva: Una aplicación desarrollada en Streamlit permite a cualquier usuario interactuar con el modelo final, probar sus capacidades con diferentes perfiles y ofertas, y visualizar su rendimiento en tiempo real.
3. Demo Interactiva en Streamlit
Para una exploración práctica del sistema, puedes ejecutar la demo localmente. La aplicación permite buscar en los datasets, seleccionar una oferta y una lista de candidatos, y obtener un ranking de compatibilidad detallado y justificado.

Demo de DistilMatch en Streamlit

Para lanzar la aplicación:

bash
# Asegúrate de tener el entorno virtual activado
streamlit run app.py
4. Estructura del Repositorio
El proyecto sigue una estructura MLOps para garantizar la modularidad y reproducibilidad:

text
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
5. Instalación y Uso
5.1. Prerrequisitos
Python 3.10 o superior
Git y Git LFS
DVC (pip install dvc[s3])
5.2. Pasos de Instalación
Clona el repositorio:

bash
git clone https://github.com/tu_usuario/tu_repositorio.git
cd TFG_DistilMatch
Crea y activa un entorno virtual:

bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .\.venv\Scripts\activate  # Windows
Instala las dependencias:

bash
pip install -r requirements.txt
Descarga los datos y modelos con DVC:

bash
dvc pull
Este comando descargará los datasets procesados y los adaptadores del modelo final.

5.3. Reproducir Experimentos
El pipeline está definido en dvc.yaml. Para reproducir el experimento de evaluación del modelo final:

bash
# Este comando evaluará el checkpoint del modelo v4 contra el Golden Set
dvc repro evaluate_v4
Para ejecutar el entrenamiento de un experimento específico (requiere una GPU configurada):

bash
# Ejemplo para lanzar el entrenamiento del experimento v4
python scripts/run_finetuning.py --experiment_id v4
6. Resultados Clave
La estrategia de Explanation-Tuning (modelo v4) demostró una mejora sustancial sobre la capacidad zero-shot del modelo base, validando el éxito de la destilación de conocimiento.

Métrica de Evaluación	Baseline (Student Zero-Shot)	Modelo Final (v4 - Explanation-Tuning)	Mejora
Error Absoluto Medio (MAE) ↓	27.16	19.98	-26.4%
Correlación de Spearman (
ρ
ρ) ↑	0.388	0.489	+26.2%
Exactitud Categórica ↑	31.8%	56.1%	+76.4%
La mejora en la Correlación de Spearman es el resultado más importante, ya que indica que el modelo final es significativamente mejor para ordenar correctamente a los candidatos de más a menos idóneo, que es el objetivo principal de un sistema de preselección.

7. Limitaciones y Trabajo Futuro
Limitaciones: El Golden Set fue anotado por un único experto. Los datasets están en inglés y centrados en el mercado de EEUU. Existe un riesgo potencial de heredar sesgos del modelo Teacher.
Trabajo Futuro:
Auditoría y Mitigación de Sesgos (Fairness).
Implementación de un ciclo de Aprendizaje Activo (Human-in-the-loop).
Especialización por Dominios mediante diferentes adaptadores LoRA.
Expansión a arquitecturas multimodales (ej. análisis de repositorios de GitHub).