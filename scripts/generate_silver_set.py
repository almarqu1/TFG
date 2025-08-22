"""
Script Definitivo y Paralelizado para la Generación del "Silver Set" de Soft Labels.

Este script utiliza el módulo `multiprocessing` de Python para lograr un paralelismo
real y robusto. Es la solución óptima para realizar llamadas masivas a una API, 
maximizando la tasa de peticiones por minuto (RPM) y acelerando drásticamente el proceso.

Características Clave:
- Paralelismo Real: Usa un pool de procesos para ejecutar múltiples llamadas a la API
  simultáneamente, garantizando una alta tasa de rendimiento.
- Compatibilidad Total: Lee todos los parámetros directamente del `config.yaml` del
  proyecto sin necesidad de modificaciones.
- Depuración Exhaustiva: Ante cualquier fallo (API o parseo), se guarda un informe
  detallado en el archivo de salida.
- Resiliencia y Reanudación: Guarda cada resultado inmediatamente y reanuda el proceso
  si se interrumpe, garantizando que no se pierda trabajo.
"""
import pandas as pd
import yaml
import json
import dotenv
import os
import time
from pathlib import Path
from tqdm import tqdm
import google.generativeai as genai
from multiprocessing import Pool

# --- 1. FUNCIONES DE AYUDA DEDICADAS ---

def cargar_configuracion() -> dict:
    """Carga y combina las configuraciones, validando las rutas y el prompt_id."""
    project_root = Path(__file__).resolve().parent.parent
    
    config_path = project_root / "config" / "config.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    prompts_config_path = project_root / "config" / "experiment_prompts.yaml"
    with open(prompts_config_path, 'r', encoding='utf-8') as f:
        prompts_config = yaml.safe_load(f)
    
    prompt_id = config['teacher_model']['prompt_id']
    if prompt_id not in prompts_config:
        raise KeyError(f"El prompt_id '{prompt_id}' no se encontró en {prompts_config_path}.")

    prompt_file_path = project_root / prompts_config[prompt_id]['path']
    with open(prompt_file_path, 'r', encoding='utf-8') as f:
        config['prompt_content'] = yaml.safe_load(f)

    config['project_root'] = project_root
    return config

def parsear_json_de_respuesta(texto_crudo: str) -> dict | None:
    """Intenta extraer y decodificar un bloque JSON de una cadena de texto."""
    try:
        texto = texto_crudo.strip()
        if texto.startswith("```"):
            texto = '\n'.join(texto.split('\n')[1:-1])
        
        json_start = texto.find('{')
        json_end = texto.rfind('}')
        if json_start == -1: return None
        
        return json.loads(texto[json_start : json_end + 1])
    except (json.JSONDecodeError, ValueError):
        return None

def llamar_api_gemini(model, prompt: str) -> tuple[bool, str]:
    """Llama a la API de forma síncrona (bloqueante)."""
    try:
        response = model.generate_content(prompt, request_options={"timeout": 180})
        
        if not hasattr(response, 'candidates') or not response.candidates:
            return (False, 'API_FAILURE: La respuesta no contiene candidatos.')
        
        candidate = response.candidates[0]
        finish_reason = candidate.finish_reason.name
        
        if finish_reason not in ["STOP", "MAX_TOKENS"]:
            return (False, f"API_FAILURE: Generación detenida por razón '{finish_reason}'.")
            
        if not hasattr(candidate, 'content') or not hasattr(candidate.content, 'parts') or not candidate.content.parts:
            return (False, f"API_FAILURE: La respuesta no contiene 'parts' de contenido (Razón: {finish_reason}).")
            
        full_text = "".join(part.text for part in candidate.content.parts if hasattr(part, 'text'))
        return (True, full_text)

    except Exception as e:
        return (False, f"API_FAILURE: Excepción durante la llamada a la API: {e}")

# --- 2. LÓGICA DEL WORKER PARA PARALELIZACIÓN ---

# Se definen como variables globales para que sean accesibles por los procesos hijos
model = None
gen_cfg = None
prompt_content = None

def init_worker(config):
    """
    Función de inicialización para cada proceso del pool.
    Configura la API key y el modelo de GenAI de forma independiente en cada worker
    para evitar problemas de serialización de objetos complejos.
    """
    global model, gen_cfg, prompt_content
    
    dotenv.load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key: raise ValueError("GEMINI_API_KEY no configurada en el worker.")
    genai.configure(api_key=api_key)

    teacher_cfg = config['teacher_model']
    gen_cfg = config['silver_set_generation']
    prompt_content = config['prompt_content']
    
    model = genai.GenerativeModel(
        model_name=teacher_cfg['model_name'],
        system_instruction=prompt_content['system_instruction'],
        generation_config={"temperature": teacher_cfg['temperature']}
    )

def process_row(row):
    """
    La función de trabajo principal que se ejecuta en cada proceso.
    Procesa una única fila (un par CV-oferta), llama a la API y devuelve
    un diccionario con el resultado.
    """
    user_prompt_template = prompt_content['user_prompt_template']
    
    max_words = gen_cfg['max_words_per_field']
    truncated_cv = ' '.join(str(row['formatted_cv']).split()[:max_words])
    truncated_job = ' '.join(str(row['formatted_offer']).split()[:max_words])
    prompt = user_prompt_template.format(text_of_cv=truncated_cv, text_of_job_posting=truncated_job)
    
    exito_api, contenido_api = llamar_api_gemini(model, prompt)
    
    output_record = {'candidate_id': row['candidate_id'], 'job_id': row['job_id']}

    if not exito_api:
        output_record.update({'status': 'API_FAILURE', 'error_details': contenido_api})
    else:
        parsed_json = parsear_json_de_respuesta(contenido_api)
        if parsed_json is None:
            output_record.update({'status': 'PARSE_FAILURE', 'raw_response': contenido_api})
        else:
            output_record.update({
                'status': 'SUCCESS',
                'teacher_score': parsed_json.get('score'),
                'teacher_justification': parsed_json.get('justification')
            })
    
    return output_record

# --- 3. ORQUESTACIÓN PRINCIPAL DEL SCRIPT ---

def main():
    """Pipeline que usa multiprocessing para generar soft labels en paralelo."""
    try:
        cfg = cargar_configuracion()
    except (KeyError, FileNotFoundError, ValueError) as e:
        print(f"❌ Error Crítico de Configuración: {e}")
        return

    project_root = cfg['project_root']
    gen_cfg = cfg['silver_set_generation']
    input_csv_path = project_root / cfg['data_paths']['unlabeled_pairs_csv']
    output_path = project_root / gen_cfg['output_jsonl_path']

    print(f"--- Iniciando Generación del Silver Set (Versión MULTIPROCESSING) ---")
    
    df_unlabeled = pd.read_csv(input_csv_path).head(gen_cfg['num_samples_to_generate'])
    df_unlabeled.dropna(subset=['formatted_cv', 'formatted_offer'], inplace=True)

    # Lógica de reanudación: se ejecuta antes de la paralelización para determinar el trabajo pendiente.
    processed_ids = set()
    if output_path.exists():
        with open(output_path, 'r', encoding='utf-8') as f_read:
            for line in f_read:
                try: 
                    rec = json.loads(line)
                    if 'candidate_id' in rec and 'job_id' in rec:
                        processed_ids.add((rec['candidate_id'], rec['job_id']))
                except json.JSONDecodeError: continue
        print(f"🔄 Reanudando: Se han encontrado {len(processed_ids)} registros procesados.")
    
    df_to_process = df_unlabeled[~df_unlabeled.apply(lambda r: (r['candidate_id'], r['job_id']) in processed_ids, axis=1)]
    if df_to_process.empty:
        print("🎉 ¡Proceso ya completado!")
        return
        
    # --- Lógica de Multiprocessing ---
    
    # AJUSTA ESTE VALOR. Define cuántas llamadas a la API se realizarán simultáneamente.
    # Comienza con un valor conservador (ej. 30) y auméntalo según tu límite de RPM (ej. 150)
    # y la capacidad de tu máquina.
    NUM_PROCESSES = 30 
    
    print(f"🚀 Generando {len(df_to_process)} nuevas soft labels usando {NUM_PROCESSES} procesos en paralelo.")
    
    rows_to_process = df_to_process.to_dict('records')

    with open(output_path, 'a', encoding='utf-8') as f_out:
        # Se crea el pool de procesos. El `initializer` se asegura de que cada proceso
        # llame a `init_worker` una vez al empezar.
        with Pool(processes=NUM_PROCESSES, initializer=init_worker, initargs=(cfg,)) as pool:
            # `pool.imap_unordered` es la opción más eficiente. Procesa las tareas en paralelo
            # y devuelve los resultados en cuanto están listos, permitiendo que la barra de
            # progreso se actualice de forma realista y constante.
            results_iterator = pool.imap_unordered(process_row, rows_to_process)
            
            for result in tqdm(results_iterator, total=len(rows_to_process)):
                if result:
                    f_out.write(json.dumps(result, ensure_ascii=False) + '\n')
                    f_out.flush() # Forzar la escritura al disco para máxima resiliencia.

    print(f"\n✅ --- Generación completada --- ✅")


if __name__ == "__main__":
    # La protección `if __name__ == "__main__"` es OBLIGATORIA para que `multiprocessing`
    # funcione correctamente en Windows y macOS. Asegura que la creación del pool de
    # procesos solo se ejecute en el script principal.
    main()