# TFG_DistilMatch/scripts/prepare_evaluation_sets.py

"""
Script para la Preparación de los Conjuntos de Datos de Entrenamiento y Evaluación.

Propósito:
Este script es un paso crítico en el pipeline de MLOps. Toma como entrada:
1. El "Golden Set" curado (una lista de IDs de CV-Oferta y sus scores).
2. Los datasets procesados de CVs y ofertas con todo su contenido textual.

Y genera como salida los artefactos finales que alimentarán los modelos:
- `gold_standard_train.csv` / `gold_standard_test.csv`: Sets divididos con todos los datos.
- `gold_standard_train.jsonl` / `gold_standard_test.jsonl`: Formateados con el prompt
  específico del Student, listos para el fine-tuning y la evaluación.
"""
import pandas as pd
import json
import yaml
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split
from typing import List, Dict, Any
from src.utils import load_config, load_prompt_template

# --- 1. CONFIGURACIÓN Y LOGGING ---

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- 2. FUNCIONES DE UTILIDAD (CANDIDATAS PARA src/utils.py) ---

def load_config_and_resolve_paths() -> Dict[str, Any]:
    """Carga config.yaml y resuelve las rutas de entrada y salida para este script."""
    config_path = PROJECT_ROOT / "config" / "config.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Resolver rutas de entrada
    config['resolved_paths'] = {
        'gold_full': PROJECT_ROOT / config['data_paths']['gold_standard']['full_curated_csv'],
        'offers_proc': PROJECT_ROOT / config['data_paths']['processed']['offers'],
        'cvs_proc': PROJECT_ROOT / config['data_paths']['processed']['cvs'],
        'prompts_config': PROJECT_ROOT / "config" / "experiment_prompts.yaml",
        # Resolver rutas de salida
        'train_csv': PROJECT_ROOT / config['data_paths']['gold_standard']['train_csv'],
        'test_csv': PROJECT_ROOT / config['data_paths']['gold_standard']['test_csv'],
        'train_jsonl': PROJECT_ROOT / config['data_paths']['gold_standard']['train_jsonl'],
        'test_jsonl': PROJECT_ROOT / config['data_paths']['gold_standard']['test_jsonl'],
    }
    return config

def load_prompt_template(prompt_id: str, prompts_config_path: Path) -> str:
    """
    Carga la plantilla de un prompt específico desde el catálogo de prompts.
    NOTA: Candidata ideal para `src/utils.py`.
    """
    with open(prompts_config_path, 'r', encoding='utf-8') as f:
        prompts_config = yaml.safe_load(f)
    
    prompt_info = prompts_config.get(prompt_id)
    if not prompt_info:
        raise ValueError(f"Prompt ID '{prompt_id}' no encontrado en {prompts_config_path}")

    prompt_file_path = PROJECT_ROOT / prompt_info['path']
    with open(prompt_file_path, 'r', encoding='utf-8') as f:
        return f.read()

# --- 3. FUNCIONES DEL PIPELINE DE PREPARACIÓN ---

def load_source_data(paths: Dict[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Carga los tres DataFrames fuente: IDs del Golden Set, CVs y ofertas."""
    logging.info("Cargando datasets fuente...")
    df_gold_ids = pd.read_csv(paths['gold_full'], dtype={'job_id': str, 'candidate_id': str})
    df_offers = pd.read_csv(paths['offers_proc'], dtype={'job_id': str})
    df_cvs = pd.read_csv(paths['cvs_proc'], dtype={'candidate_id': str})
    logging.info(f"Cargados {len(df_gold_ids)} IDs del Golden Set, {len(df_offers)} ofertas, y {len(df_cvs)} CVs.")
    return df_gold_ids, df_offers, df_cvs

def merge_and_enrich_data(df_gold_ids: pd.DataFrame, df_offers: pd.DataFrame, df_cvs: pd.DataFrame) -> pd.DataFrame:
    """Une los DataFrames para enriquecer los pares del Golden Set con su contenido textual."""
    logging.info("Enriqueciendo el Golden Set con el contenido de CVs y ofertas...")
    df_merged = pd.merge(df_gold_ids, df_offers, on='job_id', how='left')
    df_full_data = pd.merge(df_merged, df_cvs, on='candidate_id', how='left')
    
    # Validación post-unión
    initial_rows = len(df_full_data)
    # Comprobamos que las columnas clave para el prompt no estén vacías.
    df_full_data.dropna(subset=['formatted_work_history', 'description'], inplace=True)
    if len(df_full_data) < initial_rows:
        logging.warning(f"Se eliminaron {initial_rows - len(df_full_data)} filas por falta de texto esencial (CV u oferta).")
    
    if df_full_data.empty:
        raise ValueError("El DataFrame resultante está vacío. Revisa la coincidencia de IDs entre los archivos.")
    
    logging.info(f"Enriquecimiento completado. Dataset final con {len(df_full_data)} registros válidos.")
    return df_full_data

def format_row_for_prompting(row: pd.Series, entity_cols: List[str]) -> str:
    """Formatea las columnas de una entidad (CV u oferta) en un bloque de texto estructurado."""
    text_parts = []
    for col in entity_cols:
        if col in row and pd.notna(row[col]) and str(row[col]).strip():
            # Formateamos el nombre de la columna para que sea legible.
            col_name_formatted = col.replace('_', ' ').title()
            text_parts.append(f"--- {col_name_formatted} ---\n{row[col]}")
    return "\n\n".join(text_parts)

def convert_df_to_jsonl(df: pd.DataFrame, cv_cols: List[str], offer_cols: List[str], prompt_template: str, output_path: Path):
    """Convierte un DataFrame al formato JSONL que esperan los modelos de Hugging Face."""
    logging.info(f"Convirtiendo {len(df)} filas a formato JSONL en '{output_path.name}'...")
    records = []
    for _, row in df.iterrows():
        cv_text_block = format_row_for_prompting(row, cv_cols)
        offer_text_block = format_row_for_prompting(row, offer_cols)
        
        prompt_text = prompt_template.format(text_of_cv=cv_text_block, text_of_job_posting=offer_text_block)
        response_text = f"Score: {float(row['score'])}"
        
        records.append({"prompt": prompt_text, "response": response_text})
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    logging.info("Conversión a JSONL completada.")

# --- 4. PIPELINE PRINCIPAL ---

def main():
    """Orquesta el pipeline completo de preparación de datos."""
    logging.info("--- Iniciando la preparación de los sets de entrenamiento y evaluación ---")
    try:
        config = load_config()
        project_root = Path(__file__).resolve().parent.parent
        # Construimos el diccionario 'paths' que necesita el resto del script
        paths = {
            'gold_full': project_root / config['data_paths']['gold_standard']['full_curated_csv'],
            'offers_proc': project_root / config['data_paths']['processed']['offers'],
            'cvs_proc': project_root / config['data_paths']['processed']['cvs'],
            'train_csv': project_root / config['data_paths']['gold_standard']['train_csv'],
            'test_csv': project_root / config['data_paths']['gold_standard']['test_csv'],
            'train_jsonl': project_root / config['data_paths']['gold_standard']['train_jsonl'],
            'test_jsonl': project_root / config['data_paths']['gold_standard']['test_jsonl'],
        }
        
        # Paso 1: Cargar datos fuente
        df_gold_ids, df_offers, df_cvs = load_source_data(paths)
        
        # Paso 2: Unir y validar los datos
        df_full_data = merge_and_enrich_data(df_gold_ids, df_offers, df_cvs)

        # Paso 3: División estratificada
        split_params = config['data_split']
        logging.info(f"Dividiendo los datos ({1-split_params['test_size']:.0%} train / {split_params['test_size']:.0%} test)...")
        df_train, df_test = train_test_split(
            df_full_data,
            test_size=split_params['test_size'],
            random_state=config['global_seed'],
            stratify=df_full_data[split_params['stratify_column']]
        )
        logging.info(f"División completada: {len(df_train)} para entrenamiento, {len(df_test)} para test.")

        # Paso 4: Guardar los archivos CSV
        paths['train_csv'].parent.mkdir(parents=True, exist_ok=True)
        df_train.to_csv(paths['train_csv'], index=False, encoding='utf-8')
        df_test.to_csv(paths['test_csv'], index=False, encoding='utf-8')
        logging.info(f"Archivos CSV guardados en: {paths['train_csv'].parent}")
        
        # Paso 5: Formatear y guardar los archivos JSONL
        student_prompt_template = load_prompt_template(
            config['student_model']['prompt_id']
        )
        # Identificamos dinámicamente las columnas de CV y oferta para el prompt
        cv_cols = [col for col in df_cvs.columns if col != 'candidate_id']
        offer_cols = [col for col in df_offers.columns if col != 'job_id']
        
        convert_df_to_jsonl(df_train, cv_cols, offer_cols, student_prompt_template, paths['train_jsonl'])
        convert_df_to_jsonl(df_test, cv_cols, offer_cols, student_prompt_template, paths['test_jsonl'])

        logging.info("\n--- Proceso completado con éxito. Artefactos listos para el modelo. ---")

    except (FileNotFoundError, KeyError, ValueError) as e:
        logging.error(f"Ha ocurrido un error en el pipeline: {e}", exc_info=True)

if __name__ == "__main__":
    main()