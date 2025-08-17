# TFG_DistilMatch/scripts/evaluate_zero_shot.py

import pandas as pd
import yaml # <--- Importamos
from pathlib import Path

# --- Carga de Configuración Centralizada ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

# --- Extracción de Parámetros desde el config ---
TEST_DATASET_PATH = PROJECT_ROOT / config['data_paths']['gold_standard']['test_jsonl']
OUTPUT_DIR = PROJECT_ROOT / config['output_paths']['reports']
STUDENT_MODEL_NAME = config['student_model']['base_model_name'] # ¡Leemos el nombre del modelo también!
TEACHER_MODEL_NAME = config['teacher_model']['model_name']

# Asegúrate de que el directorio de salida exista
OUTPUT_DIR.mkdir(parents=True, exist_ok=True) 

def load_model(model_name: str):
    # ... (lógica futura)
    pass

def load_dataset(dataset_path: Path) -> pd.DataFrame:
    # ... (lógica futura)
    pass

def run_evaluation(model, tokenizer, dataset: pd.DataFrame):
    # ... (lógica futura)
    pass

def main():
    print("--- Script de Evaluación Zero-Shot (configurado por YAML) ---")
    
    # Decidir qué modelo evaluar cambiando solo el YAML o con un argumento
    print(f"\n--- Evaluando Baseline del Student: {STUDENT_MODEL_NAME} ---")
    # model_student, tokenizer_student = load_model(STUDENT_MODEL_NAME)
    # test_dataset = load_dataset(TEST_DATASET_PATH)
    # if not test_dataset.empty:
    #     run_evaluation(model_student, tokenizer_student, test_dataset)

    print(f"\n--- Evaluando Candidato a Teacher: {TEACHER_MODEL_NAME} ---")
    # model_teacher, tokenizer_teacher = load_model(TEACHER_MODEL_NAME)
    # test_dataset = load_dataset(TEST_DATASET_PATH)
    # if not test_dataset.empty:
    #     run_evaluation(model_teacher, tokenizer_teacher, test_dataset)
    
if __name__ == "__main__":
    main()