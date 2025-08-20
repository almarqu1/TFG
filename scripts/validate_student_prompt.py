# TFG_DistilMatch/scripts/validate_student_prompt.py

import pandas as pd
import yaml
import re
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from typing import List

# --- 1. CONFIGURACIÓN Y CARGA DE PARÁMETROS ---
# Este script realiza una prueba cualitativa rápida para validar que el modelo Student
# comprende y sigue el formato del prompt antes de lanzar experimentos a gran escala.

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
PROMPTS_CONFIG_PATH = PROJECT_ROOT / "config" / "experiment_prompts.yaml"

with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

# Parámetros de la prueba
MODEL_NAME = config['student_model']['base_model_name']
PROMPTS_TO_VALIDATE = ["S-03_student_silver_bullet"]
VALIDATION_SET_PATH = PROJECT_ROOT / config['data_paths']['test_sets']['prompt_validation_set']
PROCESSED_OFFERS_PATH = PROJECT_ROOT / config['data_paths']['processed']['offers']
PROCESSED_CVS_PATH = PROJECT_ROOT / config['data_paths']['processed']['cvs']
JOB_ID_COL, CANDIDATE_ID_COL = 'job_id', 'candidate_id'

# --- 2. FUNCIONES AUXILIARES ---

def load_prompt_template(prompt_id: str) -> str:
    """Carga una plantilla de prompt desde el archivo de configuración de prompts."""
    with open(PROMPTS_CONFIG_PATH, 'r', encoding='utf-8') as f:
        prompts_config = yaml.safe_load(f)
    prompt_info = prompts_config.get(prompt_id)
    if not prompt_info:
        raise ValueError(f"Prompt ID '{prompt_id}' no encontrado en {PROMPTS_CONFIG_PATH}.")
    with open(PROJECT_ROOT / prompt_info['path'], 'r', encoding='utf-8') as f:
        return f.read()

def format_entity_as_text(record: pd.Series, entity_columns: List[str]) -> str:
    """Convierte las columnas relevantes de una entidad (CV/oferta) en un bloque de texto formateado."""
    text_parts = []
    for col in entity_columns:
        if col in record and pd.notna(record[col]) and str(record[col]).strip() != '':
            col_name = col.replace('_', ' ').title()
            text_parts.append(f"--- {col_name} ---\n{record[col]}\n")
    return "\n".join(text_parts)

def parse_score(text: str) -> (float, bool):
    """
    Extrae un score numérico del texto y valida si sigue el formato 'Score: [número]'.
    Devuelve una tupla: (score_extraido, formato_correcto).
    """
    match = re.search(r"Score:\s*(\d+\.?\d*)", text, re.IGNORECASE)
    if match:
        return float(match.group(1)), True
    return None, False

# --- 3. PIPELINE PRINCIPAL DE VALIDACIÓN ---

def main():
    """
    Orquesta el pipeline de validación:
    1. Carga los datos de prueba y las plantillas de prompt.
    2. Carga el modelo LLM Student con cuantización para eficiencia.
    3. Para cada prompt a validar, itera sobre los datos de prueba.
    4. Formatea el input usando el 'chat template' oficial del modelo.
    5. Genera una respuesta y la analiza para verificar el formato y la coherencia.
    """
    print("--- Iniciando Validación Cualitativa de Prompts para el Modelo Student ---")

    # Paso 1: Cargar y preparar datos de validación
    try:
        print("Cargando y preparando datos...")
        df_val_ids = pd.read_csv(VALIDATION_SET_PATH, dtype=str)
        df_offers = pd.read_csv(PROCESSED_OFFERS_PATH, dtype=str)
        df_cvs = pd.read_csv(PROCESSED_CVS_PATH, dtype=str)
        
        cv_columns = [col for col in df_cvs.columns if col != CANDIDATE_ID_COL]
        offer_columns = [col for col in df_offers.columns if col != JOB_ID_COL]

        df_full_val = pd.merge(df_val_ids, df_offers, on=JOB_ID_COL, how='left')
        df_full_val = pd.merge(df_full_val, df_cvs, on=CANDIDATE_ID_COL, how='left')
        df_full_val.dropna(subset=['formatted_work_history', 'description'], inplace=True)
        print(f"Se probarán {len(df_full_val)} pares de validación.")
    except FileNotFoundError as e:
        print(f"ERROR: No se encontró el archivo '{e.filename}'. Revisa las rutas en config.yaml.")
        return

    # Paso 2: Cargar el modelo LLM Student
    print(f"\nCargando el modelo '{MODEL_NAME}' en 4-bit. Esto puede tardar unos minutos...")
    quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quantization_config,
        device_map="auto"
    )
    print("Modelo cargado con éxito.")

    # Paso 3: Iterar y validar prompts
    for prompt_id in PROMPTS_TO_VALIDATE:
        print(f"\n\n--- VALIDANDO PROMPT: {prompt_id} ---")
        prompt_template = load_prompt_template(prompt_id)

        for _, row in df_full_val.iterrows():
            cv_text = format_entity_as_text(row, cv_columns)
            offer_text = format_entity_as_text(row, offer_columns)
            
            # Construir el contenido del mensaje del usuario
            user_content = prompt_template.format(text_of_cv=cv_text, text_of_job_posting=offer_text)
            
            # Formatear el input usando el Chat Template
            messages = [{"role": "user", "content": user_content}]
            
            # El tokenizer convierte el chat en una sola cadena de texto con tokens especiales.
            templated_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            # Ahora, tokenizamos esa cadena de texto formateada.
            model_inputs = tokenizer([templated_text], return_tensors="pt").to(model.device)
            
            # Generar la respuesta del modelo
            generated_ids = model.generate(**model_inputs, max_new_tokens=25)
            
            # Decodificar solo la parte nueva de la respuesta
            output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
            model_response = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
            
            # Analizar y mostrar el resultado
            score, is_ok = parse_score(model_response)
            verdict = "OK" if is_ok else "FALLO DE FORMATO"
            
            print(f"\n  - Par: {row[JOB_ID_COL]} / {row[CANDIDATE_ID_COL]}")
            print(f"  - Respuesta del Modelo: '{model_response}'")
            print(f"  - Veredicto: {verdict} (Score: {score})")

    print("\n--- Validación completada ---")

if __name__ == "__main__":
    main()