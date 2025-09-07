"""
Pipeline de Pre-procesamiento y Enriquecimiento de Datos.

Este script es el primer paso del pipeline de MLOps. Su responsabilidad es
transformar los datos crudos (CSV de ofertas y CVs) en datasets limpios,
estructurados y, lo más importante, enriquecidos.

Las transformaciones clave incluyen:
- Cálculo de la experiencia laboral total en años (`total_experience_years`).
- Creación de un historial laboral formateado y legible (`formatted_work_history`).
- Fusión de datos dispersos (skills, industrias) en listas consolidadas.
"""
import pandas as pd
import logging
from pathlib import Path
from datetime import datetime
import ast
from dateutil.parser import parse as parse_date
import sys

# Añadimos la raíz del proyecto para poder importar desde src/utils
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.utils import load_config # ¡Importamos nuestra función centralizada!

# --- 1. CONFIGURACIÓN Y CONSTANTES ---

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Fecha de corte estática para los cálculos de experiencia. Se usa para
# calcular la duración de los trabajos "actuales".
DATA_CUTOFF_DATE = datetime(2024, 1, 1)

# Alias para términos que indican que un trabajo es el actual.
CURRENT_JOB_ALIASES = {'current', 'present', 'ongoing', 'till date', 'none', ''}

# --- 2. FUNCIONES DE PROCESAMIENTO DE CVS ---

def _format_work_history(row: pd.Series) -> str:
    """
    Combina la información de puestos, fechas y habilidades en un único string formateado.
    Es una de las características enriquecidas clave del pipeline.
    """
    try:
        # Usamos ast.literal_eval para convertir de forma segura el string "[...]" a una lista Python.
        positions = ast.literal_eval(row['positions'])
        start_dates = ast.literal_eval(row['start_dates'])
        end_dates = ast.literal_eval(row['end_dates'])
        skills_per_job = ast.literal_eval(row['related_skils_in_job'])
    except (ValueError, SyntaxError, TypeError):
        # Si los datos no son listas válidas, devolvemos un string vacío.
        return ""

    history_entries = []
    for pos, start, end, skills in zip(positions, start_dates, end_dates, skills_per_job):
        # Normalizamos el texto de la fecha de fin para que sea consistente.
        end_str = "Present" if str(end).lower().strip() in CURRENT_JOB_ALIASES else str(end).strip().title()
        
        entry = f"- {pos} ({start} - {end_str})"
        if skills and isinstance(skills, list):
            skills_str = ", ".join(skills)
            entry += f"\n  Skills: {skills_str}"
        history_entries.append(entry)
    
    return "\n\n".join(history_entries)


def _calculate_experience_for_row(row: pd.Series) -> float:
    """
    Calcula la experiencia total en años para un candidato, aplicando una lógica de
    "capping" para evitar valores inflados por datos inconsistentes.
    """
    try:
        start_dates = ast.literal_eval(row['start_dates'])
        end_dates = ast.literal_eval(row['end_dates'])
        if not (isinstance(start_dates, list) and isinstance(end_dates, list)):
             return 0.0
    except (ValueError, SyntaxError, TypeError):
        logging.warning(f"No se pudieron parsear las listas de fechas para el candidato índice {row.name}. Se asigna 0 experiencia.")
        return 0.0

    total_experience_days = 0
    parsed_start_dates = []
    for start_str, end_str in zip(start_dates, end_dates):
        try:
            start_dt = parse_date(start_str)
            parsed_start_dates.append(start_dt)
            end_dt = DATA_CUTOFF_DATE if str(end_str).lower().strip() in CURRENT_JOB_ALIASES else parse_date(str(end_str))
            
            duration = (end_dt - start_dt).days
            if duration > 0:
                total_experience_days += duration
        except (TypeError, ValueError, AttributeError):
            # Ignora pares de fechas mal formados.
            continue
    
    if not parsed_start_dates:
        return 0.0

    # Lógica de capping: la experiencia total no puede ser mayor que el tiempo
    # transcurrido desde el inicio del primer trabajo. Esto corrige solapamientos.
    first_job_start_date = min(parsed_start_dates)
    max_possible_experience_days = (DATA_CUTOFF_DATE - first_job_start_date).days
    capped_experience_days = min(total_experience_days, max_possible_experience_days)
    
    return round(max(0, capped_experience_days) / 365.25, 2)

def process_cvs(cvs_path: Path) -> pd.DataFrame:
    """Orquesta el procesamiento y enriquecimiento completo de los datos de CVs."""
    logging.info("Iniciando el procesamiento de CVs...")
    try:
        # Leemos solo las columnas que vamos a usar para optimizar memoria.
        cvs_df = pd.read_csv(cvs_path)
    except FileNotFoundError:
        logging.error(f"Archivo de CVs no encontrado en: {cvs_path}")
        raise

    logging.info("Generando características enriquecidas: 'total_experience_years' y 'formatted_work_history'...")
    cvs_df['total_experience_years'] = cvs_df.apply(_calculate_experience_for_row, axis=1)
    cvs_df['formatted_work_history'] = cvs_df.apply(_format_work_history, axis=1)
    
    logging.info("Generando IDs únicos y limpiando columnas de origen redundantes...")
    cvs_df.reset_index(inplace=True)
    cvs_df.rename(columns={'index': 'candidate_id'}, inplace=True)
    cvs_df['candidate_id'] = 'cand_' + cvs_df['candidate_id'].astype(str)
    
    source_cols_to_drop = ['positions', 'start_dates', 'end_dates', 'related_skils_in_job', 'responsibilities']
    cvs_df.drop(columns=[col for col in source_cols_to_drop if col in cvs_df.columns], inplace=True)

    # Reordenamos las columnas para mayor claridad en el CSV de salida.
    first_cols = ['candidate_id', 'total_experience_years', 'formatted_work_history']
    other_cols = [col for col in cvs_df.columns if col not in first_cols]
    return cvs_df[first_cols + other_cols]

# --- 3. FUNCIONES DE PROCESAMIENTO DE OFERTAS ---

def process_offers(paths: dict) -> pd.DataFrame:
    """Carga, fusiona y limpia los datos de las ofertas de trabajo."""
    logging.info("Iniciando el procesamiento de ofertas...")
    try:
        offers_df = pd.read_csv(paths['offers'], dtype={'job_id': str})
        job_skills_df = pd.read_csv(paths['job_skills'], dtype={'job_id': str})
        skills_map_df = pd.read_csv(paths['skills_map'])
        job_industries_df = pd.read_csv(paths['job_industries'], dtype={'job_id': str})
        industries_map_df = pd.read_csv(paths['industries_map'])
    except FileNotFoundError as e:
        logging.error(f"Archivo no encontrado: {e.filename}. Abortando.")
        raise
        
    logging.info("Agregando skills e industrias a cada oferta...")
    skills_agg = pd.merge(job_skills_df, skills_map_df, on='skill_abr', how='left').groupby('job_id')['skill_name'].apply(list).reset_index(name='skills_list')
    industries_agg = pd.merge(job_industries_df, industries_map_df, on='industry_id', how='left').groupby('job_id')['industry_name'].apply(list).reset_index(name='industries_list')
    
    offers_processed = offers_df.merge(skills_agg, on='job_id', how='left').merge(industries_agg, on='job_id', how='left')
    
    # Aseguramos que las columnas de listas vacías sean listas y no NaN.
    offers_processed['skills_list'] = offers_processed['skills_list'].apply(lambda x: x if isinstance(x, list) else [])
    offers_processed['industries_list'] = offers_processed['industries_list'].apply(lambda x: x if isinstance(x, list) else [])
        
    return offers_processed

# --- 4. ORQUESTADOR PRINCIPAL ---

def main():
    """Punto de entrada principal del script. Carga la configuración y ejecuta los pipelines."""
    logging.info("Iniciando el pipeline de pre-procesamiento de datos.")
    
    config = load_config()
    raw_paths = config['data_paths']['raw']
    processed_paths = config['data_paths']['processed']
    
    # Construir rutas absolutas
    paths_in = {key: PROJECT_ROOT / path for key, path in raw_paths.items()}
    path_out_offers = PROJECT_ROOT / processed_paths['offers']
    path_out_cvs = PROJECT_ROOT / processed_paths['cvs']
    
    path_out_offers.parent.mkdir(parents=True, exist_ok=True)

    offers_processed = process_offers(paths_in)
    offers_processed.to_csv(path_out_offers, index=False)
    logging.info(f"Ofertas procesadas y guardadas en: {path_out_offers}")

    cvs_processed = process_cvs(paths_in['cvs'])
    cvs_processed.to_csv(path_out_cvs, index=False)
    logging.info(f"CVs procesados y guardados en: {path_out_cvs}")
    
    logging.info("¡Pre-procesamiento completado con éxito!")

if __name__ == '__main__':
    main()