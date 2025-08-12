"""
Script para la Ejecución de Experimentos de Prompts (Validación del Teacher).

Este script ejecuta sistemáticamente cada prompt definido en 'config/experiment_prompts.yaml'
contra un conjunto de datos de validación ("Golden Set") que contiene etiquetas verdaderas.

El objetivo es generar un archivo de resultados crudos que compare las predicciones de cada
prompt con la verdad terreno, permitiendo una evaluación cuantitativa para seleccionar
el mejor prompt para el modelo Teacher.

Uso (desde la raíz del proyecto):
python scripts/run_prompt_experiments.py
"""
import google.generativeai as genai
import pandas as pd
import yaml
import json
import os
import time
import dotenv
from pathlib import Path
from tqdm import tqdm

# --- CONFIGURACIÓN Y CONSTANTES ---
project_root = Path(__file__).parent.parent
CONFIG_FILE = project_root / 'config' / 'config.yaml'
PROMPTS_CONFIG_FILE = project_root / 'config' / 'experiment_prompts.yaml'
PROCESSED_DATA_DIR = project_root / 'data' / '01_processed'
OUTPUT_DIR = project_root / 'outputs' / 'reports'
OUTPUT_FILE = OUTPUT_DIR / 'prompt_experiment_raw_results.csv'

MODEL_NAME = "gemini-2.5-flash-lite"

# --- FUNCIONES DE AYUDA ---
def setup_api_client():
    """Configura y valida la clave de API de Gemini."""
    try:
        dotenv.load_dotenv()  # Cargar variables de entorno desde el archivo .env
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("La variable de entorno GEMINI_API_KEY no está configurada.")
        genai.configure(api_key=api_key)
    except Exception as e:
        print(f"Error de configuración de API: {e}")
        exit()

def load_yaml_config(path):
    """Carga un archivo de configuración YAML."""
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def load_and_format_prompt(template_path, cv_text, offer_text):
    """Carga y formatea un prompt. Requiere que las llaves del JSON estén escapadas (ej. {{...}})."""
    with open(template_path, 'r', encoding='utf-8') as f:
        prompt_template = f.read()
    # Usamos .replace() para robustez, aunque con los archivos de prompt corregidos, .format() también funcionaría.
    prompt_with_offer = prompt_template.replace('{text_of_job_posting}', offer_text)
    return prompt_with_offer.replace('{text_of_cv}', cv_text)

# --- FUNCIÓN PRINCIPAL ---
def main():
    setup_api_client()

    # 1. Cargar configuraciones y datos
    print("Cargando configuraciones y datos...")
    app_config = load_yaml_config(CONFIG_FILE)
    prompts_config = load_yaml_config(PROMPTS_CONFIG_FILE)
    
    
    golden_set_path = project_root / app_config['data_paths']['test_sets']['prompt_validation_set']
    golden_set_df = pd.read_csv(golden_set_path, dtype={'job_id': str, 'candidate_id': str})

    offers_df = pd.read_csv(PROCESSED_DATA_DIR / 'offers_processed.csv', index_col='job_id', dtype={'job_id': str})
    cvs_df = pd.read_csv(PROCESSED_DATA_DIR / 'cvs_processed.csv', index_col='candidate_id', dtype={'candidate_id': str})

    offer_texts = offers_df['description'].to_dict()
    cv_texts = cvs_df['responsibilities'].to_dict()
    
    print(f"Se evaluarán {len(prompts_config)} prompts sobre {len(golden_set_df)} pares del Golden Set.")
    
    # 2. Configurar modelo y preparar almacenamiento de resultados
    generation_config = genai.GenerationConfig(temperature=0.1)
    model = genai.GenerativeModel(MODEL_NAME, generation_config=generation_config)
    all_results = []

    # 3. Bucle principal de experimentación
    for prompt_name, prompt_config in tqdm(prompts_config.items(), desc="Total Prompts"):
        prompt_template_path = project_root / prompt_config['path']
        
        for index, pair_row in tqdm(golden_set_df.iterrows(), total=len(golden_set_df), desc=f"Prompt: {prompt_name}", leave=False):
            job_id, candidate_id = pair_row['job_id'], pair_row['candidate_id']
            pair_id = f"{job_id}-{candidate_id}" # ID único para el par
            
            # Datos de "Verdad Terreno"
            true_category = pair_row['category']
            true_score = pair_row['score']

            # Construir el prompt
            offer_text, cv_text = offer_texts.get(job_id), cv_texts.get(candidate_id)
            prompt = load_and_format_prompt(prompt_template_path, cv_text, offer_text)

            # Inicializar variables de resultado
            predicted_category, predicted_score, raw_response, error_msg = None, None, None, None
            format_is_correct = False

            try:
                # 4. Llamar a la API y parsear la respuesta
                response = model.generate_content(prompt)
                raw_response = response.text
                json_str = raw_response.strip().removeprefix("```json").removesuffix("```").strip()
                parsed_json = json.loads(json_str)
                
                predicted_category = parsed_json.get('category')
                predicted_score = parsed_json.get('score')
                format_is_correct = True # Si llegamos aquí, el formato fue correcto

            except json.JSONDecodeError:
                error_msg = "JSONDecodeError"
            except Exception as e:
                error_msg = str(e)

            # 5. Almacenar el resultado
            all_results.append({
                'prompt_name': prompt_name,
                'pair_id': pair_id,
                'true_category': true_category,
                'true_score': true_score,
                'predicted_category': predicted_category,
                'predicted_score': predicted_score,
                'format_is_correct': format_is_correct,
                'error_message': error_msg,
                'raw_response': raw_response,
            })
            
            time.sleep(1.1) # Respetar el rate limit
    
    # 6. Guardar todos los resultados en un único CSV
    print("\nGuardando resultados del experimento...")
    results_df = pd.DataFrame(all_results)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(OUTPUT_FILE, index=False)
    
    print(f"✅ ¡Experimento completado! Resultados guardados en: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()