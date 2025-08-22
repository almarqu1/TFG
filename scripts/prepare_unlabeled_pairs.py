# TFG_DistilMatch/scripts/prepare_unlabeled_pairs.py

"""
Script para la Preparación de Pares CV-Oferta Sin Etiquetar (Versión Enriquecida).

Propósito:
Este script toma los archivos procesados de CVs y ofertas, y en lugar de
simplemente juntar columnas, construye dos textos formateados y ricos en
información para cada par aleatorio. Estos textos (`formatted_cv` y
`formatted_offer`) están diseñados para ser insertados directamente en
el prompt del LLM, dándole todo el contexto necesario.

La aleatoriedad es controlada por la `global_seed` para reproducibilidad.
"""

import pandas as pd
import yaml
from pathlib import Path

# --- Carga de Configuración Centralizada ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

# --- Extracción de Parámetros del config ---
CV_PATH = PROJECT_ROOT / config['data_paths']['cvs_processed_csv']
OFFERS_PATH = PROJECT_ROOT / config['data_paths']['offers_processed_csv']
OUTPUT_PATH = PROJECT_ROOT / config['data_paths']['unlabeled_pairs_csv']
NUM_SAMPLES = config['silver_set_generation']['num_samples_to_generate']
RANDOM_SEED = config['global_seed']

# --- NUEVAS FUNCIONES DE FORMATEO ---

def format_cv_text(row: pd.Series) -> str:
    """Construye un único bloque de texto estructurado a partir de una fila de CV."""
    parts = []
    
    # Función auxiliar para añadir secciones solo si tienen contenido
    def add_section(title, content):
        if pd.notna(content) and str(content).strip():
            parts.append(f"--- {title.upper()} ---\n{content}")

    add_section("Total Years of Experience", row.get('total_experience_years'))
    add_section("Career Objective", row.get('career_objective'))
    add_section("Formatted Work History", row.get('formatted_work_history'))
    add_section("Skills", row.get('skills'))
    add_section("Education", f"Institución: {row.get('educational_institution_name')}\nGrado: {row.get('degree_names')}\nEspecialidad: {row.get('major_field_of_studies')}")
    add_section("Languages", row.get('languages'))
    add_section("Certifications", row.get('certification_skills'))
    
    return "\n\n".join(parts)

def format_offer_text(row: pd.Series) -> str:
    """Construye un único bloque de texto estructurado a partir de una fila de oferta."""
    parts = []

    def add_section(title, content):
        if pd.notna(content) and str(content).strip():
            parts.append(f"--- {title.upper()} ---\n{content}")
            
    # Crear una cabecera con metadatos clave
    header = (
        f"Title: {row.get('title')}\n"
        f"Experience Level: {row.get('formatted_experience_level')}\n"
        f"Work Type: {row.get('formatted_work_type')}\n"
        
    )
    parts.append(header)
    
    add_section("Description", row.get('description'))
    add_section("Required Skills", row.get('skills_list'))
    add_section("Industries", row.get('industries_list'))
    
    return "\n\n".join(parts)

def main():
    print("--- Iniciando la preparación de pares CV-Oferta (formato enriquecido) ---")

    df_cvs = pd.read_csv(CV_PATH)
    df_offers = pd.read_csv(OFFERS_PATH)

    print(f"Tomando una muestra reproducible de {NUM_SAMPLES} CVs y ofertas (semilla: {RANDOM_SEED})...")
    cvs_sample = df_cvs.sample(n=NUM_SAMPLES, random_state=RANDOM_SEED).reset_index(drop=True)
    offers_sample = df_offers.sample(n=NUM_SAMPLES, random_state=RANDOM_SEED).reset_index(drop=True)

    print("Formateando el texto enriquecido para CVs y ofertas...")
    # Aplicar las funciones de formateo para crear las nuevas columnas
    cvs_sample['formatted_cv'] = cvs_sample.apply(format_cv_text, axis=1)
    offers_sample['formatted_offer'] = offers_sample.apply(format_offer_text, axis=1)

    # Combinar en un dataframe final con solo las columnas necesarias
    df_pairs = pd.concat([
        cvs_sample[['candidate_id', 'formatted_cv']],
        offers_sample[['job_id', 'formatted_offer']]
    ], axis=1)

    print(f"Guardando {len(df_pairs)} pares formateados en: {OUTPUT_PATH}")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True) 
    df_pairs.to_csv(OUTPUT_PATH, index=False)

    print("\n--- Preparación de pares completada con éxito ---")

if __name__ == "__main__":
    main()