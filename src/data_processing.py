import pandas as pd
import argparse

# Este script se encarga de la limpieza, fusión y optimización de los
# datasets crudos, generando archivos procesados listos para ser usados
# en el pipeline de ML.

def main(args):
    """
    Función principal que ejecuta el pipeline de pre-procesamiento y limpieza de datos.
    """
    print("Iniciando el pre-procesamiento y limpieza de datos...")

    # --- Procesamiento de Ofertas ---
    print("Procesando ofertas...")
    
    # Define y carga solo las columnas necesarias para optimizar el uso de memoria.
    columnas_necesarias_ofertas = [
        'job_id', 'title', 'description', 
        'formatted_experience_level', 'formatted_work_type', 'remote_allowed'
    ]
    offers_df = pd.read_csv(args.offers_input, usecols=columnas_necesarias_ofertas)
    
    # Carga los dataframes de mapeo para skills e industries
    job_skills_df = pd.read_csv(args.job_skills_input)
    skills_map_df = pd.read_csv(args.skills_map_input)
    job_industries_df = pd.read_csv(args.job_industries_input)
    industries_map_df = pd.read_csv(args.industries_map_input)
    
    # Agrega skills e industries en formato de lista para cada job_id
    skills_agg = pd.merge(job_skills_df, skills_map_df, on='skill_abr', how='left').groupby('job_id')['skill_name'].apply(list).reset_index(name='skills_list')
    industries_agg = pd.merge(job_industries_df, industries_map_df, on='industry_id', how='left').groupby('job_id')['industry_name'].apply(list).reset_index(name='industries_list')
    
    # Fusiona los dataframes para consolidar la información de las ofertas
    offers_processed = offers_df.merge(skills_agg, on='job_id', how='left').merge(industries_agg, on='job_id', how='left')
    
    # Asegura que las columnas de listas no contengan NaNs, sino listas vacías
    offers_processed['skills_list'] = offers_processed['skills_list'].apply(lambda x: x if isinstance(x, list) else [])
    offers_processed['industries_list'] = offers_processed['industries_list'].apply(lambda x: x if isinstance(x, list) else [])
    
    # Optimiza los tipos de datos de columnas categóricas para reducir el tamaño
    for col in ['formatted_experience_level', 'formatted_work_type']:
        offers_processed[col] = offers_processed[col].astype('category')
    
    # Guarda el archivo de ofertas procesado
    offers_processed.to_csv(args.offers_output, index=False)
    print(f"Ofertas procesadas y guardadas en: {args.offers_output}")

    # --- Procesamiento de CVs ---
    print("\nProcesando CVs...")
    
    # Define las columnas de interés del dataset de CVs
    columnas_necesarias_cvs = [
        'positions', 'skills', 
        'degree_names', 'responsibilities', 
        'major_field_of_studies', 'educational_institution_name', 
        'professional_company_names', 'extra_curricular_activity_types',
        'languages', 'certification_skills'          
    ]

    cvs_df = pd.read_csv(args.cvs_input, usecols=columnas_necesarias_cvs)

    # --- Generación de ID Único para CVs ---
    
    cvs_df.reset_index(inplace=True)
    cvs_df.rename(columns={'index': 'candidate_id'}, inplace=True)
    cvs_df['candidate_id'] = 'cand_' + cvs_df['candidate_id'].astype(str)
    
    # Reordena las columnas para poner el ID al principio
    final_cv_columns = ['candidate_id'] + columnas_necesarias_cvs
    cvs_processed = cvs_df[final_cv_columns]

    # Guarda el archivo de CVs procesado
    cvs_processed.to_csv(args.cvs_output, index=False)
    print(f"CVs procesados (con ID generado) y guardados en: {args.cvs_output}")
    
    print("\n¡Pre-procesamiento completado con éxito!")


# ==============================================================================
#  PUNTO DE ENTRADA DEL SCRIPT
# ==============================================================================

if __name__ == '__main__':
    # Configura el parser de argumentos para ejecutar el script desde la terminal
    parser = argparse.ArgumentParser(description="Pre-procesa los datos crudos de ofertas y CVs, generando archivos limpios y optimizados.")
    
    # Rutas de input
    parser.add_argument('--offers_input', type=str, default='data/00_raw/datasetJobs2/postings.csv')
    parser.add_argument('--job_skills_input', type=str, default='data/00_raw/datasetJobs2/jobs/job_skills.csv')
    parser.add_argument('--skills_map_input', type=str, default='data/00_raw/datasetJobs2/mappings/skills.csv')
    parser.add_argument('--job_industries_input', type=str, default='data/00_raw/datasetJobs2/jobs/job_industries.csv')
    parser.add_argument('--industries_map_input', type=str, default='data/00_raw/datasetJobs2/mappings/industries.csv')
    parser.add_argument('--cvs_input', type=str, default='data/00_raw/datasetCV2/resume_data.csv')
    
    # Rutas de output
    parser.add_argument('--offers_output', type=str, default='data/01_processed/offers_processed.csv')
    parser.add_argument('--cvs_output', type=str, default='data/01_processed/cvs_processed.csv')

    args = parser.parse_args()
    main(args)