# TFG_DistilMatch/scripts/generate_silver_set.py

"""
Script Paralelizado para la Generación del "Silver Set" de Soft Labels.

Propósito:
Este es el motor de la Knowledge Distillation. Utiliza el modelo "Teacher"
(previamente seleccionado por su alto rendimiento zero-shot) para etiquetar un
gran volumen de pares CV-oferta no anotados. El resultado es el "Silver Set",
un dataset masivo que enseñará al modelo "Student" a emular el razonamiento
del modelo más grande y costoso.

Arquitectura:
El script utiliza `multiprocessing.Pool` para realizar llamadas a la API de
Gemini en paralelo, maximizando el rendimiento y reduciendo drásticamente el
tiempo total de generación. Incluye una lógica de reanudación robusta para
poder interrumpir y continuar el proceso sin perder trabajo.
"""

import pandas as pd
import yaml
import json
import dotenv
import os
import logging
from pathlib import Path
from tqdm import tqdm
import google.generativeai as genai
from multiprocessing import Pool
from typing import Dict, Any
from src.utils import load_config, parse_json_from_response

# --- 1. CONFIGURACIÓN Y LOGGING ---

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(processName)s] - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# --- 2. FUNCIONES DE UTILIDAD (CANDIDATAS PARA src/utils.py) ---

def load_full_config() -> Dict[str, Any]:
    """
    Carga y combina los ficheros de configuración principales.
    NOTA: Esta es una función candidata para `src/utils.py`.
    """
    project_root = Path(__file__).resolve().parent.parent
    config_path = project_root / "config" / "config.yaml"
    prompts_config_path = project_root / "config" / "experiment_prompts.yaml"

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    with open(prompts_config_path, 'r', encoding='utf-8') as f:
        prompts_config = yaml.safe_load(f)
    
    # Inyectar la configuración de prompts en la config principal
    config['prompts'] = prompts_config
    config['project_root'] = project_root
    
    # Resolver rutas clave
    config['data_paths']['unlabeled_pairs'] = project_root / config['data_paths']['processed']['unlabeled_pairs']
    config['data_paths']['silver_standard_train'] = project_root / config['data_paths']['intermediate']['silver_standard_train']
    
    return config

# --- 3. LÓGICA DEL WORKER PARA PARALELIZACIÓN ---

# Usaremos un diccionario global para almacenar los objetos que no se pueden "picklear" (serializar)
# para pasarlos a los procesos hijos. El `initializer` se encargará de poblarlo.
worker_globals = {}

def init_worker(config: Dict[str, Any]):
    """
    Función de inicialización para cada proceso del pool. Se ejecuta una vez por worker.
    Configura la API key y el modelo de GenAI para evitar problemas de serialización.
    """
    # NOTA TFG: El `initializer` es crucial en multiprocessing. Permite que cada
    # proceso hijo cree sus propias conexiones de red y objetos complejos,
    # en lugar de intentar heredar los del proceso padre, lo cual a menudo falla.
    
    # El `processName` en el log nos ayudará a depurar si un worker específico falla.
    logging.info(f"Inicializando worker...")
    
    dotenv.load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY no configurada en el worker.")
    genai.configure(api_key=api_key)

    teacher_cfg = config['teacher_model']
    
    # Cargar el contenido del prompt una sola vez por worker
    prompt_id = teacher_cfg['prompt_id']
    prompt_path = config['project_root'] / config['prompts'][prompt_id]['path']
    with open(prompt_path, 'r', encoding='utf-8') as f:
        prompt_content = yaml.safe_load(f)

    # Almacenamos los objetos en el diccionario global del worker
    worker_globals['prompt_template'] = prompt_content['user_prompt_template']
    worker_globals['model'] = genai.GenerativeModel(
        model_name=teacher_cfg['model_name'],
        system_instruction=prompt_content['system_instruction'],
        generation_config={"temperature": teacher_cfg['temperature']}
    )
    worker_globals['gen_cfg'] = config['silver_set_generation']
    logging.info(f"Worker inicializado exitosamente.")

def process_row(row_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    La función de trabajo principal que se ejecuta en cada proceso.
    Procesa una única fila (un par CV-oferta), llama a la API y devuelve
    un diccionario con el resultado.
    """
    # Accedemos a los objetos desde el diccionario global del worker
    model = worker_globals['model']
    prompt_template = worker_globals['prompt_template']
    gen_cfg = worker_globals['gen_cfg']
    
    try:
        max_words = gen_cfg['max_words_per_field']
        # Truncamos los textos para evitar exceder el límite de contexto del modelo.
        truncated_cv = ' '.join(str(row_data['formatted_cv']).split()[:max_words])
        truncated_job = ' '.join(str(row_data['formatted_offer']).split()[:max_words])
        prompt = prompt_template.format(text_of_cv=truncated_cv, text_of_job_posting=truncated_job)
        
        # Llamada a la API
        response = model.generate_content(prompt, request_options={"timeout": 180})
        raw_response_text = response.text
        
        # Parseo de la respuesta
        parsed_json = parse_json_from_response(raw_response_text)
        
        output_record = {'candidate_id': row_data['candidate_id'], 'job_id': row_data['job_id']}
        if parsed_json and 'score' in parsed_json:
            output_record.update({
                'status': 'SUCCESS',
                'teacher_score': parsed_json.get('score'),
                'teacher_justification': parsed_json.get('justification')
            })
        else:
            output_record.update({'status': 'PARSE_FAILURE', 'raw_response': raw_response_text})

    except Exception as e:
        # Capturamos cualquier excepción (de API, de red, etc.)
        logging.warning(f"Fallo en API para par {row_data['candidate_id']}-{row_data['job_id']}: {e}")
        output_record = {
            'candidate_id': row_data['candidate_id'],
            'job_id': row_data['job_id'],
            'status': 'API_FAILURE',
            'error_details': str(e)
        }
        
    return output_record

# --- 4. ORQUESTACIÓN PRINCIPAL ---

def main():
    """Pipeline que orquesta la generación de soft labels en paralelo."""
    try:
        cfg = load_config()
        project_root = Path(__file__).resolve().parent.parent
        prompts_config_path = project_root / "config" / "experiment_prompts.yaml"
        with open(prompts_config_path, 'r', encoding='utf-8') as f:
            prompts_config = yaml.safe_load(f)
        cfg['prompts'] = prompts_config
        cfg['project_root'] = project_root
        cfg['data_paths']['unlabeled_pairs'] = project_root / cfg['data_paths']['processed']['unlabeled_pairs']
        cfg['data_paths']['silver_standard_train'] = project_root / cfg['data_paths']['intermediate']['silver_standard_train']
    except (KeyError, FileNotFoundError, ValueError) as e:
        logging.error(f"Error Crítico de Configuración: {e}", exc_info=True)
        return

    gen_cfg = cfg['silver_set_generation']
    input_csv_path = cfg['data_paths']['unlabeled_pairs']
    output_path = cfg['data_paths']['silver_standard_train']
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logging.info("--- Iniciando Generación del Silver Set (Versión Multiprocessing) ---")
    
    df_unlabeled = pd.read_csv(input_csv_path)
    df_unlabeled.dropna(subset=['formatted_cv', 'formatted_offer'], inplace=True)
    
    # Recortar al número de muestras deseado si es menor que el total.
    num_samples = min(gen_cfg['num_samples_to_generate'], len(df_unlabeled))
    df_unlabeled = df_unlabeled.head(num_samples)

    # Lógica de reanudación
    processed_ids = set()
    if output_path.exists():
        with open(output_path, 'r', encoding='utf-8') as f:
            for line in f:
                try: 
                    rec = json.loads(line)
                    if 'candidate_id' in rec and 'job_id' in rec:
                        processed_ids.add((rec['candidate_id'], rec['job_id']))
                except json.JSONDecodeError: continue
        logging.info(f"Reanudando: Se han encontrado {len(processed_ids)} registros ya procesados.")
    
    df_to_process = df_unlabeled[~df_unlabeled.apply(lambda r: (r['candidate_id'], r['job_id']) in processed_ids, axis=1)]
    if df_to_process.empty:
        logging.info("🎉 ¡Proceso ya completado! No hay nuevos pares que procesar.")
        return
        
    # --- Lógica de Multiprocessing ---
    # AJUSTA ESTE VALOR según tu límite de RPM y la capacidad de tu máquina.
    NUM_PROCESSES = 30 
    
    logging.info(f"🚀 Generando {len(df_to_process)} nuevas soft labels usando {NUM_PROCESSES} procesos en paralelo.")
    
    rows_to_process = df_to_process.to_dict('records')

    # `pool.imap_unordered` es la opción más eficiente. Devuelve los resultados
    # en cuanto están listos, sin esperar a que terminen los demás, lo que permite
    # que la barra de progreso y la escritura en disco sean constantes.
    try:
        with open(output_path, 'a', encoding='utf-8') as f_out:
            with Pool(processes=NUM_PROCESSES, initializer=init_worker, initargs=(cfg,)) as pool:
                results_iterator = pool.imap_unordered(process_row, rows_to_process)
                
                for result in tqdm(results_iterator, total=len(rows_to_process)):
                    if result:
                        f_out.write(json.dumps(result, ensure_ascii=False) + '\n')
                        f_out.flush() # Forzar escritura para máxima resiliencia.

        logging.info("\n✅ --- Generación completada con éxito --- ✅")
    except Exception as e:
        logging.error(f"Ha ocurrido un error fatal durante el procesamiento en paralelo: {e}", exc_info=True)


if __name__ == "__main__":
    # La protección `if __name__ == "__main__"` es OBLIGATORIA para que multiprocessing
    # funcione correctamente, especialmente en Windows y macOS.
    main()