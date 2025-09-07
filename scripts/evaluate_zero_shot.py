# TFG_DistilMatch/scripts/evaluate_zero_shot.py

"""
Script de Evaluación "Zero-Shot" para Modelos de Lenguaje.

Propósito:
Este script establece el rendimiento base (baseline) de diferentes LLMs en la tarea de
scoring de compatibilidad CV-oferta sin ningún tipo de fine-tuning. Es un paso
fundamental para:
1.  Seleccionar objetivamente el mejor modelo "Teacher" (el que mejor se alinea
    con el juicio humano de fábrica).
2.  Establecer la métrica de rendimiento que el modelo "Student" deberá superar
    después del proceso de Knowledge Distillation.

Uso:
- Para evaluar el Student (Qwen): `python scripts/evaluate_zero_shot.py student`
- Para evaluar el Teacher (Gemini): `python scripts/evaluate_zero_shot.py teacher`
"""

import pandas as pd
import torch
import argparse
import google.generativeai as genai
import dotenv
import os
import time
import logging
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr, pearsonr
from abc import ABC, abstractmethod
from typing import Dict, Any
from src.utils import load_config, parse_score_from_string
from src.constants import SCORE_TO_CATEGORY, CATEGORY_TO_SCORE


# --- 1. CONFIGURACIÓN Y UTILIDADES (Candidatos a src/utils.py) ---

# Configuración del logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def map_score_to_category(score: float) -> str:
    """Encuentra la categoría de la rúbrica más cercana a un score dado."""
    if score is None:
        return None
    # Encuentra la clave (score de la rúbrica) que tiene la mínima diferencia absoluta con el score dado
    closest_score = min(CATEGORY_TO_SCORE.values(), key=lambda k: abs(k - score))
    return SCORE_TO_CATEGORY[closest_score]

def calculate_and_log_metrics(results_df: pd.DataFrame, model_name: str):
    """Calcula y muestra las métricas de evaluación clave, incluyendo la accuracy categórica."""
    df = results_df.copy()
    # Limpiamos el df para asegurar que solo tenemos pares válidos para el cálculo.
    df['true_score'] = df['true_score'].apply(lambda x: parse_score_from_string(str(x)))
    df.dropna(subset=['predicted_score', 'true_score'], inplace=True)
    
    if len(df) < 2:
        logging.warning("No hay suficientes predicciones válidas para calcular las métricas.")
        return

    y_true = df['true_score'].astype(float)
    y_pred = df['predicted_score'].astype(float)

    # Métricas de Regresión y Correlación
    mae = mean_absolute_error(y_true, y_pred)
    spearman_corr, _ = spearmanr(y_true, y_pred)
    pearson_corr, _ = pearsonr(y_true, y_pred)

    # Cálculo de la Accuracy Categórica
    df['true_category'] = df['true_score'].map(SCORE_TO_CATEGORY)
    df['predicted_category'] = df['predicted_score'].apply(map_score_to_category)
    
    # Calculamos la accuracy solo donde la categoría predicha no es nula
    valid_categories_df = df.dropna(subset=['predicted_category'])
    if not valid_categories_df.empty:
        categorical_accuracy = (valid_categories_df['true_category'] == valid_categories_df['predicted_category']).mean() * 100
    else:
        categorical_accuracy = 0.0

    logging.info(f"\n--- Métricas de Evaluación para: {model_name.upper()} ---")
    logging.info(f"Pares Válidos Evaluados: {len(df)}")
    logging.info(f"Error Absoluto Medio (MAE): {mae:.4f}")
    logging.info(f"Correlación de Pearson (r): {pearson_corr:.4f} (Mide relación lineal)")
    logging.info(f"Correlación de Spearman (ρ): {spearman_corr:.4f} (Mide relación monotónica, más robusta)")
    logging.info(f"Accuracy Categórica: {categorical_accuracy:.2f}% (Alineamiento con decisión de negocio)")
    logging.info("--------------------------------------------------")


# --- 2. ABSTRACCIÓN DE MODELOS ---

class Evaluator(ABC):
    """Clase base abstracta para cualquier evaluador de modelos."""
    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    def predict(self, prompt: str) -> Dict[str, Any]:
        """Debe devolver un diccionario con 'predicted_score' y 'full_response'."""
        pass

    def evaluate(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Ciclo de evaluación genérico que itera sobre un dataset."""
        predictions = []
        pbar = tqdm(dataset.iterrows(), total=len(dataset), desc=f"Evaluando {self.model_name}")
        for _, row in pbar:
            prompt = row['prompt']
            true_score_str = row['response']
            
            result = self.predict(prompt)
            
            predictions.append({
                'prompt': prompt,
                'true_score': true_score_str,
                'predicted_score': result['predicted_score'],
                'full_response': result['full_response']
            })
        return pd.DataFrame(predictions)

class StudentEvaluator(Evaluator):
    """Evaluador para modelos locales de Hugging Face como Qwen."""
    def __init__(self, model_name: str):
        super().__init__(model_name)
        logging.info(f"Cargando modelo local: {model_name}...")
        # Cuantización en 4-bit  para reducir el consumo de VRAM
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto" # Distribuye el modelo en las GPUs disponibles
        )
        logging.info("Modelo Student cargado y configurado.")

    def predict(self, prompt: str) -> Dict[str, Any]:
        messages = [{"role": "user", "content": prompt}]
        text_input = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer([text_input], return_tensors="pt").to(self.model.device)
        
        with torch.no_grad(): # Desactiva el cálculo de gradientes para acelerar la inferencia
            generated_ids = self.model.generate(**model_inputs, max_new_tokens=20, do_sample=False)
        
        # Decodificamos solo los tokens generados, no el prompt de entrada
        response_text = self.tokenizer.batch_decode(generated_ids[:, model_inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
        
        return {
            'predicted_score': parse_score_from_string(response_text),
            'full_response': response_text
        }

class TeacherEvaluator(Evaluator):
    """Evaluador para modelos de API como Gemini."""
    def __init__(self, model_name: str, delay_sec: float = 12.0):
        super().__init__(model_name)
        logging.info(f"Configurando cliente de API para: {model_name}")
        dotenv.load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("La variable de entorno GEMINI_API_KEY no está configurada.")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.delay_sec = delay_sec

    def predict(self, prompt: str) -> Dict[str, Any]:
        try:
            response = self.model.generate_content(prompt)
            response_text = response.text
            predicted_score = parse_score_from_string(response_text)
        except Exception as e:
            logging.error(f"Error de API: {e}")
            response_text = f"API_ERROR: {e}"
            predicted_score = None
        
        # Respetar los límites de la API
        time.sleep(self.delay_sec)
        
        return {
            'predicted_score': predicted_score,
            'full_response': response_text
        }

# --- 3. PIPELINE PRINCIPAL ---

def main():
    parser = argparse.ArgumentParser(description="Ejecutar evaluación Zero-Shot para un modelo.")
    parser.add_argument("model_type", choices=['student', 'teacher'], help="El tipo de modelo a evaluar.")
    args = parser.parse_args()

    try:
        cfg = load_config()
        project_root = Path(__file__).resolve().parent.parent
        
    except (FileNotFoundError, KeyError) as e:
        logging.error(f"Error de configuración: {e}")
        return

    logging.info(f"--- Iniciando evaluación para el modelo: {args.model_type.upper()} ---")
    
    # Cargamos el dataset de test curado
    test_dataset_path = project_root / cfg['data_paths']['gold_standard']['enriched_full_jsonl']
    try:
        test_dataset = pd.read_json(test_dataset_path, lines=True)
        logging.info(f"Dataset de test cargado desde '{test_dataset_path}' ({len(test_dataset)} filas).")
    except FileNotFoundError:
        logging.error(f"No se encontró el archivo de dataset en {test_dataset_path}")
        return

    # Selección polimórfica del evaluador
    if args.model_type == 'student':
        evaluator = StudentEvaluator(cfg['student_model']['base_model_name'])
        output_path = project_root / cfg['output_paths']['eval_results']['student_baseline']
    else: # teacher
        evaluator = TeacherEvaluator(
            cfg['teacher_model']['model_name'],
            delay_sec=cfg['silver_set_generation'].get('delay_between_requests_sec', 12.0)
        )
        output_path = project_root / cfg['output_paths']['eval_results']['teacher_candidate']
    
    # Ejecución de la evaluación
    results_df = evaluator.evaluate(test_dataset)
    
    # Guardado de resultados y reporte de métricas
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    logging.info(f"Resultados detallados guardados en: {output_path}")
    
    calculate_and_log_metrics(results_df, model_name=evaluator.model_name)
    
    logging.info("Evaluación completada con éxito.")

if __name__ == "__main__":
    main()