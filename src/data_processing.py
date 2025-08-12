import pandas as pd
import argparse
import logging
from pathlib import Path
from datetime import datetime
import ast  # Para la evaluación segura de literales de Python (e.g., listas en formato string)
from dateutil.parser import parse as parse_date  # Parser de fechas flexible

# Configuración del logging para el seguimiento del pipeline
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- CONSTANTES GLOBALES ---

# Fecha de corte estática para los cálculos de experiencia.
DATA_CUTOFF_DATE = datetime(2024, 1, 1)

# Alias para términos que indican que un trabajo es el actual.
CURRENT_JOB_ALIASES = {'current', 'present', 'ongoing', 'till date', 'none', ''}


# --- FUNCIONES MODULARES DE PROCESAMIENTO ---

def _format_work_history(row):
    """
    Combina la información de puestos, fechas y habilidades en un único string formateado.
    """
    try:
        positions = ast.literal_eval(row['positions'])
        start_dates = ast.literal_eval(row['start_dates'])
        end_dates = ast.literal_eval(row['end_dates'])
        skills_per_job = ast.literal_eval(row['related_skils_in_job'])
    except (ValueError, SyntaxError):
        return ""

    history_entries = []
    for pos, start, end, skills in zip(positions, start_dates, end_dates, skills_per_job):
        end_str = str(end).strip().title() if str(end).lower().strip() not in CURRENT_JOB_ALIASES else "Present"
        
        entry = f"- {pos} ({start} - {end_str})"
        if skills and isinstance(skills, list):
            skills_str = ", ".join(skills)
            entry += f"\n  Skills: {skills_str}"
        history_entries.append(entry)
    
    return "\n\n".join(history_entries)


def _calculate_experience_for_row(row):
    """
    Calcula la experiencia total para una única fila de candidato, con una comprobación de sanidad.
    """
    try:
        start_dates_str = row['start_dates']
        end_dates_str = row['end_dates']
        
        if not isinstance(start_dates_str, str) or not isinstance(end_dates_str, str):
            return 0.0

        start_dates = ast.literal_eval(start_dates_str)
        end_dates = ast.literal_eval(end_dates_str)

    except (ValueError, SyntaxError):
        logging.warning(f"No se pudieron parsear las listas de fechas para el candidato índice {row.name}. Se asigna 0 experiencia.")
        return 0.0

    total_experience_days = 0
    parsed_start_dates = []
    for start_str, end_str in zip(start_dates, end_dates):
        try:
            start_dt = parse_date(start_str)
            parsed_start_dates.append(start_dt)

            if str(end_str).lower().strip() in CURRENT_JOB_ALIASES:
                end_dt = DATA_CUTOFF_DATE
            else:
                end_dt = parse_date(str(end_str))
            
            duration = (end_dt - start_dt).days
            if duration > 0:
                total_experience_days += duration
        except (TypeError, ValueError):
            continue

    if not parsed_start_dates:
        return 0.0

    first_job_start_date = min(parsed_start_dates)
    max_possible_experience_days = (DATA_CUTOFF_DATE - first_job_start_date).days

    capped_experience_days = min(total_experience_days, max_possible_experience_days)
    
    return round(max(0, capped_experience_days) / 365.25, 2)


def process_offers(offers_path, job_skills_path, skills_map_path, job_industries_path, industries_map_path):
    """Carga, fusiona y limpia los datos de las ofertas de trabajo."""
    logging.info("Iniciando el procesamiento de ofertas...")
    
    try:
        usecols_offers = ['job_id', 'title', 'description', 'formatted_experience_level', 'formatted_work_type', 'remote_allowed']
        offers_df = pd.read_csv(offers_path, usecols=usecols_offers, dtype={'job_id': str})
        job_skills_df = pd.read_csv(job_skills_path, dtype={'job_id': str})
        skills_map_df = pd.read_csv(skills_map_path)
        job_industries_df = pd.read_csv(job_industries_path, dtype={'job_id': str})
        industries_map_df = pd.read_csv(industries_map_path)
    except FileNotFoundError as e:
        logging.error(f"Archivo no encontrado: {e.filename}. Abortando.")
        raise
        
    logging.info("Agregando skills e industrias a cada oferta...")
    skills_agg = pd.merge(job_skills_df, skills_map_df, on='skill_abr', how='left').groupby('job_id')['skill_name'].apply(list).reset_index(name='skills_list')
    industries_agg = pd.merge(job_industries_df, industries_map_df, on='industry_id', how='left').groupby('job_id')['industry_name'].apply(list).reset_index(name='industries_list')
    
    offers_processed = offers_df.merge(skills_agg, on='job_id', how='left').merge(industries_agg, on='job_id', how='left')
    
    offers_processed['skills_list'] = offers_processed['skills_list'].apply(lambda x: x if isinstance(x, list) else [])
    offers_processed['industries_list'] = offers_processed['industries_list'].apply(lambda x: x if isinstance(x, list) else [])
    
    logging.info("Optimizando tipos de datos...")
    for col in ['formatted_experience_level', 'formatted_work_type']:
        offers_processed[col] = offers_processed[col].astype('category')
        
    return offers_processed

def process_cvs(cvs_path):
    """
    Carga y procesa los datos de los CVs, creando columnas enriquecidas y
    eliminando las columnas de origen redundantes.
    """
    logging.info("Iniciando el procesamiento de CVs...")
    
    try:
        usecols_cvs = [
            'career_objective', 'positions', 'start_dates', 'end_dates', 'related_skils_in_job',
            'skills', 'degree_names', 'responsibilities', 
            'major_field_of_studies', 'educational_institution_name', 
            'professional_company_names', 'extra_curricular_activity_types',
            'languages', 'certification_skills'          
        ]
        cvs_df = pd.read_csv(cvs_path, usecols=usecols_cvs)
    except (FileNotFoundError, ValueError) as e:
        logging.error(f"Error al leer el archivo o las columnas de CVs. Asegúrate de que el archivo y las columnas existen. Error: {e}")
        raise

    logging.info("Calculando la experiencia laboral total y generando historial formateado...")
    cvs_df['total_experience_years'] = cvs_df.apply(_calculate_experience_for_row, axis=1)
    cvs_df['formatted_work_history'] = cvs_df.apply(_format_work_history, axis=1)
    
    logging.info("Generando IDs únicos y limpiando columnas redundantes...")
    cvs_df.reset_index(inplace=True)
    cvs_df.rename(columns={'index': 'candidate_id'}, inplace=True)
    cvs_df['candidate_id'] = 'cand_' + cvs_df['candidate_id'].astype(str)
    
    # Estas columnas han cumplido su propósito y ahora son redundantes.
    source_cols_to_drop = ['positions', 'start_dates', 'end_dates', 'related_skils_in_job']
    cvs_df.drop(columns=source_cols_to_drop, inplace=True)

    # Definir el orden final de las columnas para mayor claridad en el archivo de salida.
    # Colocar las columnas generadas y de identificación al principio.
    first_cols = ['candidate_id', 'total_experience_years', 'formatted_work_history']
    other_cols = [col for col in cvs_df.columns if col not in first_cols]
    final_order = first_cols + other_cols
    
    cvs_processed = cvs_df[final_order]
    
    return cvs_processed

# --- ORQUESTADOR PRINCIPAL ---

def main(args):
    """Función principal que orquesta el pipeline de pre-procesamiento de datos."""
    logging.info("Iniciando el pipeline de pre-procesamiento de datos.")
    
    Path(args.offers_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.cvs_output).parent.mkdir(parents=True, exist_ok=True)

    offers_processed = process_offers(args.offers_input, args.job_skills_input, args.skills_map_input, args.job_industries_input, args.industries_map_input)
    offers_processed.to_csv(args.offers_output, index=False)
    logging.info(f"Ofertas procesadas y guardadas en: {args.offers_output}")

    cvs_processed = process_cvs(args.cvs_input)
    cvs_processed.to_csv(args.cvs_output, index=False)
    logging.info(f"CVs procesados (con esquema limpio y enriquecido) y guardados en: {args.cvs_output}")
    
    logging.info("¡Pre-procesamiento completado con éxito!")

# ==============================================================================
#  PUNTO DE ENTRADA DEL SCRIPT
# ==============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pre-procesa los datos crudos de ofertas y CVs.")
    
    project_root = Path(__file__).resolve().parent.parent
    
    parser.add_argument('--offers_input', type=str, default=str(project_root / 'data/00_raw/datasetJobs2/postings.csv'))
    parser.add_argument('--job_skills_input', type=str, default=str(project_root / 'data/00_raw/datasetJobs2/jobs/job_skills.csv'))
    parser.add_argument('--skills_map_input', type=str, default=str(project_root / 'data/00_raw/datasetJobs2/mappings/skills.csv'))
    parser.add_argument('--job_industries_input', type=str, default=str(project_root / 'data/00_raw/datasetJobs2/jobs/job_industries.csv'))
    parser.add_argument('--industries_map_input', type=str, default=str(project_root / 'data/00_raw/datasetJobs2/mappings/industries.csv'))
    parser.add_argument('--cvs_input', type=str, default=str(project_root / 'data/00_raw/datasetCV2/resume_data.csv'))
    
    parser.add_argument('--offers_output', type=str, default=str(project_root / 'data/01_processed/offers_processed.csv'))
    parser.add_argument('--cvs_output', type=str, default=str(project_root / 'data/01_processed/cvs_processed.csv'))

    args = parser.parse_args()
    main(args)