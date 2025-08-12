"""
Script para la Generación de "Soft Labels" usando la API de Gemini.

Este script lee un archivo de configuración de prompts (YAML), permite al usuario 
seleccionar un prompt específico por su nombre, y genera etiquetas de compatibilidad para pares de CV-Oferta.

Uso desde la línea de comandos (desde la raíz del proyecto):
python scripts/generate_soft_labels.py --prompt_name P-01_zero_shot
python scripts/generate_soft_labels.py --prompt_name P-04_cot_few_shot
"""

import google.generativeai as genai
import pandas as pd
import json
import os
import time
import yaml
import argparse
from pathlib import Path
from tqdm import tqdm

# --- CONFIGURACIÓN Y CONSTANTES ---

# Rutas clave del proyecto
project_root = Path(__file__).parent.parent
PROMPTS_CONFIG_FILE = project_root / 'config' / 'experiment_prompts.yaml'
DATA_DIR = project_root / 'data'
PROCESSED_DATA_DIR = DATA_DIR / '01_processed'
TEST_SETS_DIR = DATA_DIR / '02_test_sets'
INTERMEDIATE_DIR = DATA_DIR / '02_intermediate'

# Modelo de Gemini a utilizar
MODEL_NAME = "gemini-2.5-pro-latest"

# --- FUNCIONES DE AYUDA ---

def load_yaml_config(path):
    """Carga un archivo de configuración YAML."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Archivo de configuración no encontrado en {path}")
        return None

def load_and_format_prompt(template_path, cv_text, offer_text):
    """Carga una plantilla de prompt y la formatea con los datos del par."""
    try:
        with open(template_path, 'r', encoding='utf-8') as f:
            prompt_template = f.read()
        return prompt_template.format(text_of_job_posting=offer_text, text_of_cv=cv_text)
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo de prompt en {template_path}")
        return None
    except KeyError as e:
        print(f"Error: El prompt no contiene el marcador de posición esperado: {e}")
        return None

def setup_api_client():
    """Configura y valida la clave de API de Gemini."""
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("La variable de entorno GEMINI_API_KEY no está configurada.")
        genai.configure(api_key=api_key)
    except Exception as e:
        print(f"Error de configuración de API: {e}")
        exit()

# --- FUNCIÓN PRINCIPAL DE EJECUCIÓN ---

def main(prompt_name):
    """
    Función principal que orquesta la generación de soft labels.
    
    Args:
        prompt_name (str): El nombre del prompt a utilizar, debe ser una clave
                           en 'experiment_prompts.yaml'.
    """
    setup_api_client()

    # 1. Cargar la configuración de los prompts
    prompts_config = load_yaml_config(PROMPTS_CONFIG_FILE)
    if not prompts_config:
        return

    if prompt_name not in prompts_config:
        print(f"Error: El nombre de prompt '{prompt_name}' no se encontró en '{PROMPTS_CONFIG_FILE.name}'.")
        print("Opciones disponibles:", list(prompts_config.keys()))
        return
        
    chosen_prompt_config = prompts_config[prompt_name]
    prompt_template_path = project_root / chosen_prompt_config['path']
    print(f"✅ Usando prompt: '{prompt_name}' - {chosen_prompt_config['description']}")

    # 2. Cargar y preparar los datos
    print("Cargando datos de pares, CVs y ofertas...")
    pairs_df = pd.read_csv(TEST_SETS_DIR / 'test_set_pairs_to_annotate.csv', dtype={'job_id': str, 'candidate_id': str})
    offers_df = pd.read_csv(PROCESSED_DATA_DIR / 'offers_processed.csv', index_col='job_id', dtype={'job_id': str})
    cvs_df = pd.read_csv(PROCESSED_DATA_DIR / 'cvs_processed.csv', index_col='candidate_id', dtype={'candidate_id': str})

    # Asumimos que la descripción completa está en estas columnas. Ajustar si es necesario.
    offer_texts = offers_df['description'].to_dict()
    cv_texts = cvs_df['responsibilities'].to_dict()
    
    # 3. Configurar el modelo y archivo de salida
    generation_config = genai.GenerationConfig(temperature=0.1)
    model = genai.GenerativeModel(MODEL_NAME, generation_config=generation_config)
    
    # Crear un nombre de archivo de salida dinámico para el experimento
    output_filename = f"soft_labels_{prompt_name}.jsonl"
    output_file_path = INTERMEDIATE_DIR / output_filename
    INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Iniciando generación. Los resultados se guardarán en: {output_file_path}")

    # 4. Iterar y llamar a la API
    with open(output_file_path, 'w', encoding='utf-8') as f_out:
        for index, row in tqdm(pairs_df.iterrows(), total=pairs_df.shape[0], desc=f"Prompt: {prompt_name}"):
            job_id, candidate_id = row['job_id'], row['candidate_id']
            offer_text, cv_text = offer_texts.get(job_id), cv_texts.get(candidate_id)

            if not all([offer_text, cv_text]):
                print(f"Saltando par por datos faltantes (Job: {job_id}, Candidate: {candidate_id})")
                continue

            prompt = load_and_format_prompt(prompt_template_path, cv_text, offer_text)
            if not prompt:
                print("Deteniendo ejecución por error en la plantilla del prompt.")
                break
            
            try:
                response = model.generate_content(prompt)
                json_str = response.text.strip().removeprefix("```json").removesuffix("```").strip()
                result_json = json.loads(json_str)
                result_json.update({'job_id': job_id, 'candidate_id': candidate_id})
                f_out.write(json.dumps(result_json) + '\n')

            except json.JSONDecodeError:
                error_data = {'job_id': job_id, 'candidate_id': candidate_id, 'error': 'JSONDecodeError', 'raw_response': response.text}
                f_out.write(json.dumps(error_data) + '\n')
            except Exception as e:
                error_data = {'job_id': job_id, 'candidate_id': candidate_id, 'error': str(e)}
                f_out.write(json.dumps(error_data) + '\n')

            time.sleep(1.1) # Respetar el rate limit de la API

    print(f"\n✅ ¡Proceso completado para el prompt '{prompt_name}'!")

if __name__ == "__main__":
    # Configuración del parser de argumentos de la línea de comandos
    parser = argparse.ArgumentParser(description='Genera soft labels para DistilMatch usando un prompt específico.')
    parser.add_argument(
        '--prompt_name',
        type=str,
        required=True,
        help="El nombre del prompt a usar (ej: P-01_zero_shot). Debe ser una clave en 'config/experiment_prompts.yaml'."
    )
    args = parser.parse_args()
    
    # Llamar a la función principal con el argumento proporcionado
    main(args.prompt_name)