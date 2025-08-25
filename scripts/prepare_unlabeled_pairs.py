# TFG_DistilMatch/scripts/prepare_unlabeled_pairs.py

"""
Script para la Preparación de Pares CV-Oferta Sin Etiquetar (Formato Enriquecido).

Propósito:
Este script construye el dataset de entrada para el modelo Teacher. En lugar de
simplemente unir columnas, genera dos textos formateados y ricos en información
para cada par (`formatted_cv` y `formatted_offer`). Este paso es fundamental
para el principio "Garbage In, Garbage Out": al proporcionar al LLM un input
limpio, estructurado y completo, maximizamos la calidad de las "soft labels"
que generará.

El proceso toma una muestra aleatoria pero reproducible de los CVs y ofertas
procesados para crear los pares.
"""

import pandas as pd
import yaml
import logging
from pathlib import Path
from typing import Dict, Any

# --- 1. CONFIGURACIÓN Y LOGGING ---

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_config_and_resolve_paths() -> Dict[str, Any]:
    """Carga config.yaml y resuelve las rutas necesarias para este script."""
    project_root = Path(__file__).resolve().parent.parent
    config_path = project_root / "config" / "config.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Resolvemos las rutas que vamos a usar
    config['resolved_paths'] = {
        'cvs_processed': project_root / config['data_paths']['processed']['cvs'],
        'offers_processed': project_root / config['data_paths']['processed']['offers'],
        'unlabeled_pairs_output': project_root / config['data_paths']['processed']['unlabeled_pairs']
    }
    return config

# --- 2. LÓGICA DE FORMATEO ENRIQUECIDO ---

def format_entity_text(row: pd.Series, sections: Dict[str, Any]) -> str:
    """
    Función genérica para construir un bloque de texto estructurado a partir de una fila.

    Args:
        row: Una fila (pd.Series) de un DataFrame.
        sections: Un diccionario que define las secciones del texto.
                  Las claves son los títulos de sección.
                  Los valores pueden ser un string (nombre de columna) o una
                  función lambda que formatea varias columnas.
    """
    parts = []
    
    def add_section(title: str, content: Any):
        # Solo añadimos la sección si el contenido no es nulo o una cadena vacía.
        if pd.notna(content) and str(content).strip():
            parts.append(f"--- {title.upper()} ---\n{content}")

    for title, source in sections.items():
        if isinstance(source, str): # Si es un simple nombre de columna
            add_section(title, row.get(source))
        elif callable(source): # Si es una función de formateo
            add_section(title, source(row))
            
    return "\n\n".join(parts)

def get_cv_sections() -> Dict[str, Any]:
    """Define la estructura y el contenido de la sección de CV."""
    return {
        "Total Years of Experience": 'total_experience_years',
        "Career Objective": 'career_objective',
        "Formatted Work History": 'formatted_work_history',
        "Skills": 'skills',
        "Education": lambda r: (
            f"Institución: {r.get('educational_institution_name', 'N/A')}\n"
            f"Grado: {r.get('degree_names', 'N/A')}\n"
            f"Especialidad: {r.get('major_field_of_studies', 'N/A')}"
        ),
        "Languages": 'languages',
        "Certifications": 'certification_skills',
    }
    
def get_offer_sections() -> Dict[str, Any]:
    """Define la estructura y el contenido de la sección de Oferta."""
    return {
        "Job Details": lambda r: (
            f"Title: {r.get('title', 'N/A')}\n"
            f"Experience Level: {r.get('formatted_experience_level', 'N/A')}\n"
            f"Work Type: {r.get('formatted_work_type', 'N/A')}"
        ),
        "Description": 'description',
        "Required Skills": 'skills_list',
        "Industries": 'industries_list',
    }

# --- 3. PIPELINE PRINCIPAL ---

def main():
    """Orquesta la creación de pares CV-Oferta no etiquetados."""
    logging.info("--- Iniciando la preparación de pares CV-Oferta (formato enriquecido) ---")
    try:
        config = load_config_and_resolve_paths()
        paths = config['resolved_paths']
        
        df_cvs = pd.read_csv(paths['cvs_processed'])
        df_offers = pd.read_csv(paths['offers_processed'])

        # Parámetros para el muestreo
        num_samples = config['silver_set_generation']['num_samples_to_generate']
        random_seed = config['global_seed']

        logging.info(f"Tomando una muestra reproducible de {num_samples} CVs y ofertas (semilla: {random_seed}).")
        # Aseguramos que no pedimos más muestras de las disponibles
        n_cvs = min(num_samples, len(df_cvs))
        n_offers = min(num_samples, len(df_offers))

        cvs_sample = df_cvs.sample(n=n_cvs, random_state=random_seed).reset_index(drop=True)
        offers_sample = df_offers.sample(n=n_offers, random_state=random_seed).reset_index(drop=True)

        logging.info("Aplicando formato de texto enriquecido para CVs y ofertas...")
        
        cv_sections = get_cv_sections()
        offer_sections = get_offer_sections()

        cvs_sample['formatted_cv'] = cvs_sample.apply(lambda row: format_entity_text(row, cv_sections), axis=1)
        offers_sample['formatted_offer'] = offers_sample.apply(lambda row: format_entity_text(row, offer_sections), axis=1)

        # Combinamos las muestras en un DataFrame final con las columnas necesarias
        df_pairs = pd.concat([
            cvs_sample[['candidate_id', 'formatted_cv']],
            offers_sample[['job_id', 'formatted_offer']]
        ], axis=1)

        output_path = paths['unlabeled_pairs_output']
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_pairs.to_csv(output_path, index=False)
        logging.info(f"Guardados {len(df_pairs)} pares formateados en: {output_path}")

        logging.info("\n--- Preparación de pares completada con éxito ---")

    except (FileNotFoundError, KeyError) as e:
        logging.error(f"Error de configuración o de archivo: {e}", exc_info=True)
    except Exception as e:
        logging.error(f"Ha ocurrido un error inesperado: {e}", exc_info=True)


if __name__ == "__main__":
    main()