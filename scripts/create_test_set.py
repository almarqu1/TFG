import pandas as pd
import numpy as np
import argparse
import os
from tqdm import tqdm
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.constants import INDUSTRY_KEYWORDS_MAP
from src.text_utils import parse_string_list

def sample_targeted_pairs(offers_df, cvs_df, num_pairs):
    print(f"Iniciando muestreo dirigido para generar {num_pairs} pares...")
    pairs = set()
    
    cvs_df['searchable_text'] = cvs_df.apply(
        lambda row: ' '.join(
            str(s) for s in parse_string_list(row['positions']) + 
                           parse_string_list(row['major_field_of_studies']) +
                           parse_string_list(row['skills'])
        ).lower(),
        axis=1
    )

    ## Usamos las industrias definidas en nuestro mapa como base para la búsqueda
    available_industries = list(INDUSTRY_KEYWORDS_MAP.keys())

    pbar = tqdm(total=num_pairs, desc="Generando pares dirigidos")
    attempts = 0
    max_attempts = num_pairs * 20 # Aumentamos los intentos por si algunas industrias son raras

    while len(pairs) < num_pairs and attempts < max_attempts:
        ## Elegimos una industria de nuestra lista predefinida
        target_industry = np.random.choice(available_industries)
        
        ## Obtenemos la lista de keywords y creamos un regex
        keywords = INDUSTRY_KEYWORDS_MAP[target_industry]
        regex_pattern = '|'.join(keywords)
        
        possible_offers = offers_df[offers_df['industries_list'].apply(lambda x: target_industry in x)]
        if possible_offers.empty:
            attempts += 1
            continue
            
        ## Filtramos CVs usando el patrón de regex, ignorando mayúsculas/minúsculas
        possible_cvs = cvs_df[cvs_df['searchable_text'].str.contains(regex_pattern, na=False, case=False)]
        if possible_cvs.empty:
            attempts += 1
            continue

        offer_id = possible_offers.sample(1)['job_id'].iloc[0]
        candidate_id = possible_cvs.sample(1)['candidate_id'].iloc[0]
        
        if (offer_id, candidate_id) not in pairs:
            pairs.add((offer_id, candidate_id))
            pbar.update(1)
        attempts += 1
        
    pbar.close()
    if len(pairs) < num_pairs:
        print(f"\nAdvertencia: Solo se pudieron generar {len(pairs)} pares dirigidos de los {num_pairs} solicitados.")
    
    return pd.DataFrame(list(pairs), columns=['job_id', 'candidate_id'])


def sample_random_pairs(offers_df, cvs_df, num_pairs):
    """
    Genera pares completamente aleatorios. (Esta función no cambia)
    """
    print(f"\nIniciando muestreo aleatorio para generar {num_pairs} pares...")
    pairs = set()
    
    offer_ids = offers_df['job_id'].unique()
    candidate_ids = cvs_df['candidate_id'].unique()
    
    pbar = tqdm(total=num_pairs, desc="Generando pares aleatorios")
    while len(pairs) < num_pairs:
        offer_id = np.random.choice(offer_ids)
        candidate_id = np.random.choice(candidate_ids)
        
        if (offer_id, candidate_id) not in pairs:
            pairs.add((offer_id, candidate_id))
            pbar.update(1)

    pbar.close()
    return pd.DataFrame(list(pairs), columns=['job_id', 'candidate_id'])


def main(args):
    # Asegura reproducibilidad
    np.random.seed(args.seed)
    
    print("Cargando datos procesados...")

    converters = {
        'skills_list': parse_string_list,
        'industries_list': parse_string_list
    }
    offers_df = pd.read_csv(args.offers_processed_input, converters=converters)
    cvs_df = pd.read_csv(args.cvs_processed_input)

    num_targeted = args.num_pairs // 2
    num_random = args.num_pairs - num_targeted

    targeted_pairs_df = sample_targeted_pairs(offers_df, cvs_df, num_targeted)
    random_pairs_df = sample_random_pairs(offers_df, cvs_df, num_random)

    print("\nCombinando, eliminando duplicados y barajando...")
    final_pairs_df = pd.concat([targeted_pairs_df, random_pairs_df], ignore_index=True)
    final_pairs_df.drop_duplicates(inplace=True)
    final_pairs_df = final_pairs_df.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    final_pairs_df['category'] = ''
    final_pairs_df['score'] = ''
    final_pairs_df['annotator_id'] = ''
    final_pairs_df['justification'] = ''
    
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    final_pairs_df.to_csv(args.output_file, index=False)
    
    print("\n¡Proceso completado!")
    print(f"Se ha generado el archivo de pares para anotación: {args.output_file}")
    print(f"Total de pares únicos generados: {len(final_pairs_df)}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Crea el conjunto de test 'Gold Standard'.")
    default_offers = str(project_root / 'data/01_processed/offers_processed.csv')
    default_cvs = str(project_root / 'data/01_processed/cvs_processed.csv')
    default_output = str(project_root / 'data/02_test_sets/test_set_pairs_to_annotate.csv')
    
    parser.add_argument('--offers_processed_input', type=str, default=default_offers)
    parser.add_argument('--cvs_processed_input', type=str, default=default_cvs)
    parser.add_argument('--output_file', type=str, default=default_output)
    parser.add_argument('--num_pairs', type=int, default=360)
    parser.add_argument('--seed', type=int, default=42, help="Semilla aleatoria para reproducibilidad.")
    args = parser.parse_args()
    main(args)