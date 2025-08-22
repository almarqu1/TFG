# TFG_DistilMatch/scripts/evaluate_zero_shot.py

"""
Script de Evaluación "Zero-Shot" para los modelos Student y Teacher.

Propósito:
Este script evalúa el rendimiento de un modelo de lenguaje (ya sea un modelo local
como Qwen o un modelo de API como Gemini) en la tarea de scoring de compatibilidad
CV-oferta, sin ningún fine-tuning previo.

Ejecución:
- Para evaluar el Student (Qwen): `python scripts/evaluate_zero_shot.py student`
- Para evaluar el Teacher (Gemini): `python scripts/evaluate_zero_shot.py teacher`

Resultados Generados:
1.  Un archivo CSV en `outputs/reports/` (ej. `student_baseline_results.csv`) que contiene
    un registro detallado de cada predicción, incluyendo el prompt, el score real,
    el score predicho y la respuesta completa del modelo. Este archivo es crucial
    para el análisis de errores.
2.  Un resumen de las métricas de rendimiento (MAE y Correlación de Spearman)
    impreso directamente en la consola al finalizar la ejecución.
"""

import pandas as pd
import yaml
import torch
import re
import argparse 
import google.generativeai as genai 
import dotenv 
import os
import time 
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig
from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# Carga de Configuración Centralizada 
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

# Extracción de Parámetros desde el config 
TEST_DATASET_PATH = PROJECT_ROOT / config['data_paths']['gold_standard']['test_jsonl']
OUTPUT_DIR = PROJECT_ROOT / config['output_paths']['reports']
STUDENT_BASELINE_RESULTS_PATH = PROJECT_ROOT / config['output_paths']['eval_results']['student_baseline']
TEACHER_RESULTS_PATH = PROJECT_ROOT / config['output_paths']['eval_results']['teacher_candidate']
STUDENT_MODEL_NAME = config['student_model']['base_model_name']
TEACHER_MODEL_NAME = config['teacher_model']['model_name']

OUTPUT_DIR.mkdir(parents=True, exist_ok=True) 

def parse_score(text_response: str) -> float | None:
    if not text_response:
        return None
    match = re.search(r"Score:\s*([0-9]+\.?[0-9]*)", text_response, re.IGNORECASE)
    if match:
        try:
            return float(match.group(1))
        except (ValueError, IndexError):
            return None
    return None

def load_model_and_tokenizer(model_name: str):
    print(f"Cargando modelo local: {model_name}...")
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, device_map="auto")
    model.generation_config = GenerationConfig(max_new_tokens=150, do_sample=False, pad_token_id=tokenizer.pad_token_id)
    print("Modelo y tokenizador cargados y configurados para evaluación determinista.")
    return model, tokenizer

def load_dataset(dataset_path: Path) -> pd.DataFrame:
    try:
        print(f"Cargando dataset desde: {dataset_path}")
        return pd.read_json(dataset_path, lines=True)
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo de dataset en {dataset_path}")
        return pd.DataFrame()

def run_teacher_evaluation(dataset: pd.DataFrame) -> pd.DataFrame:
    print(f"Evaluando con el modelo Teacher de API: {TEACHER_MODEL_NAME}")
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("La variable de entorno GOOGLE_API_KEY no está configurada.")
    genai.configure(api_key=api_key)
    
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }
    
    model = genai.GenerativeModel(TEACHER_MODEL_NAME, safety_settings=safety_settings)
    predictions = []

    for index, row in tqdm(dataset.iterrows(), total=dataset.shape[0], desc=f"Evaluando {TEACHER_MODEL_NAME}"):
        prompt = row['prompt']
        true_score = row['response']
        
        try:
            response = model.generate_content(prompt)
            if response.parts:
                predicted_score = parse_score(response.text)
                full_response = response.text
            else:
                finish_reason = response.candidates[0].finish_reason if response.candidates else "N/A"
                print(f"\nRespuesta vacía en la fila {index}. Razón de finalización: {finish_reason}")
                predicted_score = None
                full_response = f"EMPTY_RESPONSE | Finish Reason: {finish_reason}"
        except Exception as e:
            print(f"\nError de API no manejado en la fila {index}: {e}")
            predicted_score = None
            full_response = f"API_ERROR: {e}"
            with open("failed_prompts.txt", "a", encoding="utf-8") as f:
                f.write(f"--- ERROR: {e} ---\n{prompt}\n\n")

        predictions.append({'prompt': prompt, 'true_score': true_score, 'predicted_score': predicted_score, 'full_response': full_response})
        
        # Añadimos delay para respetar el Rate Limit de la Free Tier (5 RPM) 
        time.sleep(12)

    return pd.DataFrame(predictions)
    
def run_student_evaluation(model, tokenizer, dataset: pd.DataFrame) -> pd.DataFrame:
    predictions = []
    for index, row in tqdm(dataset.iterrows(), total=dataset.shape[0], desc=f"Evaluando {STUDENT_MODEL_NAME}"):
        prompt = row['prompt']
        true_score = row['response']
        messages = [{"role": "user", "content": prompt}]
        text_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text_input], return_tensors="pt").to(model.device)
        generated_ids = model.generate(**model_inputs)
        response_text = tokenizer.batch_decode(generated_ids[:, model_inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
        predicted_score = parse_score(response_text)
        predictions.append({'prompt': prompt, 'true_score': true_score, 'predicted_score': predicted_score, 'full_response': response_text})
    return pd.DataFrame(predictions)

def calculate_and_print_metrics(results_df: pd.DataFrame, model_type: str):
    df = results_df.copy()
    df['true_score'] = df['true_score'].apply(lambda x: parse_score(str(x)) if isinstance(x, str) else float(x))
    df.dropna(subset=['predicted_score', 'true_score'], inplace=True)
    if df.empty:
        print("No se pudieron calcular las métricas: no hay predicciones o valores reales válidos después de la limpieza.")
        return
    y_true = df['true_score'].astype(float)
    y_pred = df['predicted_score'].astype(float)
    mae = mean_absolute_error(y_true, y_pred)
    spearman_corr, _ = spearmanr(y_true, y_pred)
    
    print(f"\n--- Resultados de la Evaluación para: {model_type.upper()} ---")
    print(f"Error Absoluto Medio (MAE): {mae:.4f}")
    print(f"Correlación de Spearman (ρ): {spearman_corr:.4f}")
    print("-----------------------------------------")

def main():
    dotenv.load_dotenv() 
    parser = argparse.ArgumentParser(description="Ejecutar evaluación Zero-Shot para un modelo.")
    parser.add_argument("model_type", choices=['student', 'teacher'], help="El tipo de modelo a evaluar ('student' o 'teacher').")
    args = parser.parse_args()

    print(f"--- Iniciando evaluación para el modelo: {args.model_type.upper()} ---")
    
    test_dataset = load_dataset(TEST_DATASET_PATH)
    if test_dataset.empty:
        return

    if args.model_type == 'student':
        model, tokenizer = load_model_and_tokenizer(STUDENT_MODEL_NAME)
        results_df = run_student_evaluation(model, tokenizer, test_dataset)
        output_path = STUDENT_BASELINE_RESULTS_PATH
    elif args.model_type == 'teacher':
        results_df = run_teacher_evaluation(test_dataset)
        output_path = TEACHER_RESULTS_PATH
    
    # El resultado final del script son dos artefactos:
    # 1. El archivo CSV guardado en la ruta 'output_path'.
    print(f"\nGuardando resultados detallados en: {output_path}")
    results_df.to_csv(output_path, index=False)
    
    # 2. El resumen de métricas impreso en la consola.
    calculate_and_print_metrics(results_df, model_type=args.model_type)
    
    print("\nEvaluación completada con éxito.")

if __name__ == "__main__":
    main()