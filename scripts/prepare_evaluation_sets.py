

import pandas as pd
import json
import yaml
from pathlib import Path
from sklearn.model_selection import train_test_split
from typing import List

# --- 1. CONFIGURACIÓN Y CARGA DE PARÁMETROS ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
PROMPTS_CONFIG_PATH = PROJECT_ROOT / "config" / "experiment_prompts.yaml"

with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

# Extracción de parámetros
data_paths = config['data_paths']
gs_paths = data_paths['gold_standard']
processed_paths = data_paths['processed']
split_params = config['data_split']
student_params = config['student_model']

# Rutas de entrada y salida
GOLD_STANDARD_IDS_PATH = PROJECT_ROOT / gs_paths['full']
PROCESSED_OFFERS_PATH = PROJECT_ROOT / processed_paths['offers']
PROCESSED_CVS_PATH = PROJECT_ROOT / processed_paths['cvs']
TRAIN_CSV_PATH = PROJECT_ROOT / gs_paths['train_csv']
TEST_CSV_PATH = PROJECT_ROOT / gs_paths['test_csv']
TRAIN_JSONL_PATH = PROJECT_ROOT / gs_paths['train_jsonl']
TEST_JSONL_PATH = PROJECT_ROOT / gs_paths['test_jsonl']

# Parámetros de división y del Student
TEST_SET_SIZE = split_params['test_size']
STRATIFY_COLUMN = split_params['stratify_column']
RANDOM_STATE = config['global_seed']
STUDENT_PROMPT_ID = student_params['prompt_id']

# Nombres de columnas clave
JOB_ID_COL, CANDIDATE_ID_COL, SCORE_COL = 'job_id', 'candidate_id', 'score'
CORE_CV_COL, CORE_OFFER_COL = 'formatted_work_history', 'description'

# --- 2. FUNCIONES AUXILIARES ---

def load_prompt_template(prompt_id: str, prompts_config_path: Path) -> str:
    """Carga una plantilla de prompt desde el archivo de configuración de prompts."""
    try:
        with open(prompts_config_path, 'r', encoding='utf-8') as f:
            prompts_config = yaml.safe_load(f)
        
        prompt_info = prompts_config.get(prompt_id)
        if not prompt_info:
            raise ValueError(f"No se encontró el prompt ID '{prompt_id}' en {prompts_config_path}")

        prompt_path = PROJECT_ROOT / prompt_info['path']
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error fatal al cargar la plantilla del prompt: {e}")
        raise

def format_entity_as_text(record: pd.Series, entity_columns: List[str]) -> str:
    text_parts = []
    for col in entity_columns:
        if col in record and pd.notna(record[col]) and str(record[col]).strip() != '':
            col_name = col.replace('_', ' ').title()
            text_parts.append(f"--- {col_name} ---\n{record[col]}\n")
    return "\n".join(text_parts)

def convert_df_to_jsonl(df: pd.DataFrame, cv_cols: List[str], offer_cols: List[str], prompt_template: str, output_path: Path):
    """Convierte un DataFrame al formato JSONL, usando una plantilla de prompt."""
    print(f"Convirtiendo DataFrame a JSONL en: {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        for _, row in df.iterrows():
            cv_text_block = format_entity_as_text(row, cv_cols)
            offer_text_block = format_entity_as_text(row, offer_cols)
            
            # Rellena la plantilla del prompt con el texto correspondiente.
            prompt_text = prompt_template.format(
                text_of_cv=cv_text_block,
                text_of_job_posting=offer_text_block
            )
            response_text = f"Score: {float(row[SCORE_COL])}"
            
            record = {"prompt": prompt_text, "response": response_text}
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    print(f"Conversión a JSONL completada. {len(df)} registros guardados.")

# --- 3. PIPELINE PRINCIPAL DE EJECUCIÓN ---

def main():
    print("--- Iniciando la preparación de los sets de entrenamiento y evaluación ---")

    # Paso 1: Carga de datos y configuración
    try:
        Path(TRAIN_CSV_PATH).parent.mkdir(parents=True, exist_ok=True)
        
        print(f"Cargando plantilla de prompt del Student: '{STUDENT_PROMPT_ID}'")
        student_prompt_template = load_prompt_template(STUDENT_PROMPT_ID, PROMPTS_CONFIG_PATH)

        print("Cargando datasets fuente...")
        df_gold_ids = pd.read_csv(GOLD_STANDARD_IDS_PATH, dtype={JOB_ID_COL: str, CANDIDATE_ID_COL: str})
        df_offers = pd.read_csv(PROCESSED_OFFERS_PATH, dtype={JOB_ID_COL: str})
        df_cvs = pd.read_csv(PROCESSED_CVS_PATH, dtype={CANDIDATE_ID_COL: str})
        
        cv_columns = [col for col in df_cvs.columns if col != CANDIDATE_ID_COL]
        offer_columns = [col for col in df_offers.columns if col != JOB_ID_COL]
        
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR en la inicialización: {e}. Revisa las rutas y IDs en los ficheros de configuración.")
        return

    # Paso 2: Unión de DataFrames
    print("\nEnriqueciendo el Golden Set...")
    df_merged = pd.merge(df_gold_ids, df_offers, on=JOB_ID_COL, how='left')
    df_full_data = pd.merge(df_merged, df_cvs, on=CANDIDATE_ID_COL, how='left')

    # Paso 3: Limpieza y validación de datos
    initial_rows = len(df_full_data)
    df_full_data.dropna(subset=[CORE_CV_COL, CORE_OFFER_COL], inplace=True)
    final_rows = len(df_full_data)

    if initial_rows != final_rows:
        print(f"ADVERTENCIA: Se eliminaron {initial_rows - final_rows} filas por falta de texto esencial.")
    
    if df_full_data.empty:
        print("ERROR CRÍTICO: El DataFrame está vacío. Revisa la coincidencia de IDs.")
        return
        
    print(f"Unión y limpieza completadas. Dataset final con {len(df_full_data)} registros.")

    # Paso 4: División estratificada
    print(f"\nDividiendo los datos...")
    df_train, df_test = train_test_split(
        df_full_data, test_size=TEST_SET_SIZE, random_state=RANDOM_STATE, stratify=df_full_data[STRATIFY_COLUMN]
    )
    print(f"División completada: {len(df_train)} para entrenamiento, {len(df_test)} para test.")

    # Paso 5: Guardado de los artefactos finales
    print("\nGuardando los artefactos de datos generados...")
    df_train.to_csv(TRAIN_CSV_PATH, index=False, encoding='utf-8')
    df_test.to_csv(TEST_CSV_PATH, index=False, encoding='utf-8')
    print(" - Archivos CSV guardados.")
    
    convert_df_to_jsonl(df_train, cv_columns, offer_columns, student_prompt_template, TRAIN_JSONL_PATH)
    convert_df_to_jsonl(df_test, cv_columns, offer_columns, student_prompt_template, TEST_JSONL_PATH)
    print(" - Archivos JSONL guardados.")

    print("\n--- Proceso completado con éxito. ---")

if __name__ == "__main__":
    main()