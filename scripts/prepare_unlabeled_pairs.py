import pandas as pd
import yaml
import logging
from pathlib import Path
import sys

# Añadir la raíz del proyecto al path para permitir la importación desde src
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# Ahora importamos las funciones desde el módulo de utilidades
from src.utils import format_entity_text, get_cv_sections, get_offer_sections

# --- 1. CONFIGURACIÓN Y LOGGING ---

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_config_and_resolve_paths() -> dict:
    """Carga config.yaml y resuelve las rutas necesarias para este script."""
    config_path = PROJECT_ROOT / "config" / "config.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    config['resolved_paths'] = {
        'cvs_processed': PROJECT_ROOT / config['data_paths']['processed']['cvs'],
        'offers_processed': PROJECT_ROOT / config['data_paths']['processed']['offers'],
        'unlabeled_pairs_output': PROJECT_ROOT / config['data_paths']['processed']['unlabeled_pairs']
    }
    return config

# --- 2. PIPELINE PRINCIPAL ---

def main():
    """Orquesta la creación de pares CV-Oferta no etiquetados."""
    logging.info("--- Iniciando la preparación de pares CV-Oferta (formato enriquecido) ---")
    try:
        config = load_config_and_resolve_paths()
        paths = config['resolved_paths']
        
        df_cvs = pd.read_csv(paths['cvs_processed'])
        df_offers = pd.read_csv(paths['offers_processed'])

        num_samples = config['silver_set_generation']['num_samples_to_generate']
        random_seed = config['global_seed']

        logging.info(f"Tomando una muestra reproducible de {num_samples} CVs y ofertas (semilla: {random_seed}).")
        n_cvs = min(num_samples, len(df_cvs))
        n_offers = min(num_samples, len(df_offers))

        cvs_sample = df_cvs.sample(n=n_cvs, random_state=random_seed).reset_index(drop=True)
        offers_sample = df_offers.sample(n=n_offers, random_state=random_seed).reset_index(drop=True)

        logging.info("Aplicando formato de texto enriquecido para CVs y ofertas...")
        
        # Obtenemos las definiciones de las secciones desde nuestro módulo centralizado
        cv_sections = get_cv_sections()
        offer_sections = get_offer_sections()

        # Aplicamos la función de formateo, también importada
        cvs_sample['formatted_cv'] = cvs_sample.apply(lambda row: format_entity_text(row, cv_sections), axis=1)
        offers_sample['formatted_offer'] = offers_sample.apply(lambda row: format_entity_text(row, offer_sections), axis=1)

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
