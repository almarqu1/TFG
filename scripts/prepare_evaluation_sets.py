# TFG_DistilMatch/scripts/prepare_evaluation_sets.py

"""
Script para la Preparación de los Conjuntos de Datos de Entrenamiento y Evaluación.

Propósito:
Este script es un paso crítico en el pipeline de MLOps. Toma como entrada:
1. Un archivo de mapeo "Golden Set" con IDs y scores (ej. 'gold_standard_full_curated.csv').
2. Los datasets procesados de CVs y ofertas con su contenido textual.

Y genera como salida los artefactos finales que alimentarán los modelos:
- `gold_standard_train.csv` / `gold_standard_test.csv`: Sets enriquecidos, limpios y divididos.
- `gold_standard_train.jsonl` / `gold_standard_test.jsonl`: Formateados para fine-tuning.
- `gold_standard_enriched_full.jsonl`: Una conversión de todos los pares que pudieron ser enriquecidos con éxito.
- `gold_standard_discarded_pairs.csv`: Un reporte de los pares del Golden Set que no se encontraron.
"""
import pandas as pd
import json
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split
from typing import Dict, Tuple

from src.utils import (
    load_config,
    load_prompt_template,
    format_entity_text,
    get_cv_sections,
    get_offer_sections
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
PROJECT_ROOT = Path(__file__).resolve().parent.parent

def load_source_data(paths: Dict[str, Path]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Carga los tres DataFrames fuente: IDs del Golden Set, CVs y ofertas."""
    logging.info("Cargando datasets fuente...")
    df_gold_ids = pd.read_csv(paths['gold_full_ids'], dtype={'job_id': str, 'candidate_id': str})
    df_offers = pd.read_csv(paths['offers_proc'], dtype={'job_id': str})
    df_cvs = pd.read_csv(paths['cvs_proc'], dtype={'candidate_id': str})
    logging.info(f"Cargados {len(df_gold_ids)} IDs del Golden Set, {len(df_offers)} ofertas, y {len(df_cvs)} CVs.")
    return df_gold_ids, df_offers, df_cvs

def merge_and_enrich_data(df_gold_ids: pd.DataFrame, df_offers: pd.DataFrame, df_cvs: pd.DataFrame, discarded_pairs_path: Path) -> pd.DataFrame:
    """
    Une los DataFrames para enriquecer el Golden Set. Usa 'left joins' para mantener
    todos los pares originales, identifica los que no se pueden enriquecer y los
    guarda en un archivo de reporte antes de devolver solo los pares completos.
    """
    logging.info("Enriqueciendo el Golden Set con el contenido de CVs y ofertas...")
    initial_pairs = len(df_gold_ids)

    # Paso 1: Usar 'left join' para mantener todos los IDs del Golden Set original.
    # El indicador '_merge' nos dirá si se encontró una correspondencia.
    df_merged = pd.merge(df_gold_ids, df_offers, on='job_id', how='left', indicator='offer_found')
    df_full_potential = pd.merge(df_merged, df_cvs, on='candidate_id', how='left', indicator='cv_found')

    # Paso 2: Separar los pares que se enriquecieron con éxito de los que fallaron.
    is_enriched = (df_full_potential['offer_found'] == 'both') & (df_full_potential['cv_found'] == 'both')
    df_enriched = df_full_potential[is_enriched].copy()
    df_discarded = df_full_potential[~is_enriched].copy()

    # Limpiar columnas de merge auxiliares del DataFrame enriquecido
    if not df_enriched.empty:
        df_enriched.drop(columns=['offer_found', 'cv_found'], inplace=True)
    
    final_pairs = len(df_enriched)
    discarded_count = len(df_discarded)

    if discarded_count > 0:
        logging.warning(
            f"De {initial_pairs} pares del Golden Set, {final_pairs} fueron enriquecidos con éxito. "
            f"{discarded_count} pares fueron descartados por no encontrar el CV o la oferta correspondiente."
        )
        # Crear un reporte claro de los pares descartados
        discard_report = df_discarded[['job_id', 'candidate_id', 'score']].copy()
        discard_report['reason'] = "CV or Offer ID not found in processed files"
        discarded_pairs_path.parent.mkdir(parents=True, exist_ok=True)
        discard_report.to_csv(discarded_pairs_path, index=False, encoding='utf-8')
        logging.info(f"Se ha guardado un reporte de los pares descartados en: {discarded_pairs_path}")
    
    if df_enriched.empty:
        raise ValueError("El DataFrame resultante está vacío. Ningún ID del Golden Set encontró correspondencia en los CVs y ofertas procesadas.")
    
    logging.info(f"Enriquecimiento completado. Dataset final con {final_pairs} registros válidos.")
    return df_enriched


def convert_df_to_jsonl(df: pd.DataFrame, cv_sections: Dict, offer_sections: Dict, prompt_template: str, output_path: Path):
    """Convierte un DataFrame al formato JSONL para el fine-tuning."""
    logging.info(f"Convirtiendo {len(df)} filas a formato JSONL en '{output_path.name}'...")
    records = []
    for _, row in df.iterrows():
        cv_text_block = format_entity_text(row, cv_sections)
        offer_text_block = format_entity_text(row, offer_sections)
        prompt_text = prompt_template.format(cv=cv_text_block, job_description=offer_text_block)
        response_text = f"Score: {float(row['score'])}"
        records.append({"prompt": prompt_text, "response": response_text})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    logging.info("Conversión a JSONL completada.")

def main():
    """Orquesta el pipeline completo de preparación de datos."""
    logging.info("--- Iniciando la preparación de los sets de entrenamiento y evaluación ---")
    try:
        config = load_config()
        paths = {
            'gold_full_ids': PROJECT_ROOT / config['data_paths']['gold_standard']['full_curated_csv'],
            'offers_proc': PROJECT_ROOT / config['data_paths']['processed']['offers'],
            'cvs_proc': PROJECT_ROOT / config['data_paths']['processed']['cvs'],
            'train_csv': PROJECT_ROOT / config['data_paths']['gold_standard']['train_csv'],
            'test_csv': PROJECT_ROOT / config['data_paths']['gold_standard']['test_csv'],
            'train_jsonl': PROJECT_ROOT / config['data_paths']['gold_standard']['train_jsonl'],
            'test_jsonl': PROJECT_ROOT / config['data_paths']['gold_standard']['test_jsonl'],
            'enriched_full_jsonl': PROJECT_ROOT / config['data_paths']['gold_standard']['enriched_full_jsonl'],
            'discarded_pairs_csv': PROJECT_ROOT / config['data_paths']['gold_standard']['discarded_pairs_csv'],
        }

        # Paso 1: Cargar los tres archivos fuente.
        df_gold_ids, df_offers, df_cvs = load_source_data(paths)

        # Paso 2: Unir los datos. Ahora esta función también guarda los descartes.
        df_full_data = merge_and_enrich_data(df_gold_ids, df_offers, df_cvs, paths['discarded_pairs_csv'])

        # Paso 3: División estratificada (opera solo sobre los datos enriquecidos).
        split_params = config['data_split']
        logging.info(f"Dividiendo el set enriquecido ({len(df_full_data)} filas) en train/test...")
        df_train, df_test = train_test_split(
            df_full_data,
            test_size=split_params['test_size'],
            random_state=config['global_seed'],
            stratify=df_full_data[split_params['stratify_column']]
        )
        logging.info(f"División completada: {len(df_train)} para entrenamiento, {len(df_test)} para test.")

        # Paso 4: Guardar los archivos CSV de train y test.
        paths['train_csv'].parent.mkdir(parents=True, exist_ok=True)
        df_train.to_csv(paths['train_csv'], index=False, encoding='utf-8')
        df_test.to_csv(paths['test_csv'], index=False, encoding='utf-8')
        logging.info(f"Archivos CSV de entrenamiento y test guardados en: {paths['train_csv'].parent}")

        # Paso 5: Formatear y guardar los tres archivos JSONL.
        student_prompt_template = load_prompt_template(
            config['student_model']['prompt_id']
        )
        cv_sections = get_cv_sections()
        offer_sections = get_offer_sections()

        convert_df_to_jsonl(df_train, cv_sections, offer_sections, student_prompt_template, paths['train_jsonl'])
        convert_df_to_jsonl(df_test, cv_sections, offer_sections, student_prompt_template, paths['test_jsonl'])
        convert_df_to_jsonl(df_full_data, cv_sections, offer_sections, student_prompt_template, paths['enriched_full_jsonl'])

        logging.info("\n--- Proceso completado con éxito. Artefactos listos para el modelo. ---")

    except (FileNotFoundError, KeyError, ValueError) as e:
        logging.error(f"Ha ocurrido un error en el pipeline: {e}", exc_info=True)

if __name__ == "__main__":
    main()