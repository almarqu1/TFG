import pandas as pd
import argparse
import logging
from pathlib import Path


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- FUNCIONES MODULARES DE PROCESAMIENTO ---

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
    """Carga y procesa los datos de los CVs, generando un ID único para cada uno."""
    logging.info("Iniciando el procesamiento de CVs...")
    
    try:
        usecols_cvs = [
            'positions', 'skills', 'degree_names', 'responsibilities', 
            'major_field_of_studies', 'educational_institution_name', 
            'professional_company_names', 'extra_curricular_activity_types',
            'languages', 'certification_skills'          
        ]
        cvs_df = pd.read_csv(cvs_path, usecols=usecols_cvs)
    except FileNotFoundError as e:
        logging.error(f"Archivo no encontrado: {e.filename}. Abortando.")
        raise

    cvs_df.reset_index(inplace=True)
    cvs_df.rename(columns={'index': 'candidate_id'}, inplace=True)
    cvs_df['candidate_id'] = 'cand_' + cvs_df['candidate_id'].astype(str)
    
    final_cv_columns = ['candidate_id'] + usecols_cvs
    cvs_processed = cvs_df[final_cv_columns]
    
    return cvs_processed

# --- ORQUESTADOR PRINCIPAL ---

def main(args):
    """Función principal que orquesta el pipeline de pre-procesamiento de datos."""
    logging.info("Iniciando el pipeline de pre-procesamiento de datos.")
    
    # Asegurarse de que los directorios de salida existan
    Path(args.offers_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.cvs_output).parent.mkdir(parents=True, exist_ok=True)

    # Procesar ofertas
    offers_processed = process_offers(args.offers_input, args.job_skills_input, args.skills_map_input, args.job_industries_input, args.industries_map_input)
    offers_processed.to_csv(args.offers_output, index=False)
    logging.info(f"Ofertas procesadas y guardadas en: {args.offers_output}")

    # Procesar CVs
    cvs_processed = process_cvs(args.cvs_input)
    cvs_processed.to_csv(args.cvs_output, index=False)
    logging.info(f"CVs procesados (con ID generado) y guardados en: {args.cvs_output}")
    
    logging.info("¡Pre-procesamiento completado con éxito!")

# ==============================================================================
#  PUNTO DE ENTRADA DEL SCRIPT
# ==============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pre-procesa los datos crudos de ofertas y CVs.")
    
    project_root = Path(__file__).resolve().parent.parent
    
    # Rutas de input
    parser.add_argument('--offers_input', type=str, default=str(project_root / 'data/00_raw/datasetJobs2/postings.csv'))
    parser.add_argument('--job_skills_input', type=str, default=str(project_root / 'data/00_raw/datasetJobs2/jobs/job_skills.csv'))
    parser.add_argument('--skills_map_input', type=str, default=str(project_root / 'data/00_raw/datasetJobs2/mappings/skills.csv'))
    parser.add_argument('--job_industries_input', type=str, default=str(project_root / 'data/00_raw/datasetJobs2/jobs/job_industries.csv'))
    parser.add_argument('--industries_map_input', type=str, default=str(project_root / 'data/00_raw/datasetJobs2/mappings/industries.csv'))
    parser.add_argument('--cvs_input', type=str, default=str(project_root / 'data/00_raw/datasetCV2/resume_data.csv'))
    
    # Rutas de output
    parser.add_argument('--offers_output', type=str, default=str(project_root / 'data/01_processed/offers_processed.csv'))
    parser.add_argument('--cvs_output', type=str, default=str(project_root / 'data/01_processed/cvs_processed.csv'))

    args = parser.parse_args()
    main(args)