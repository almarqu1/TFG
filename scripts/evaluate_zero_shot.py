import pandas as pd
import yaml
import torch
import re
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig
from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr

# --- Carga de Configuración Centralizada ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

# --- Extracción de Parámetros desde el config ---
TEST_DATASET_PATH = PROJECT_ROOT / config['data_paths']['gold_standard']['test_jsonl']
OUTPUT_DIR = PROJECT_ROOT / config['output_paths']['reports']
STUDENT_BASELINE_RESULTS_PATH = PROJECT_ROOT / config['output_paths']['eval_results']['student_baseline']
STUDENT_MODEL_NAME = config['student_model']['base_model_name']

OUTPUT_DIR.mkdir(parents=True, exist_ok=True) 

def parse_score(text_response: str) -> float | None:
    if not text_response:
        return None
    match = re.search(r"Score:\s*([0-9]+\.?[0-9]*)", text_response, re.IGNORECASE)
    if match:
        try:
            return float(match.group(1))
        except (ValueError, IndexError):
            return None
    return None

def load_model_and_tokenizer(model_name: str):
    print(f"Cargando modelo: {model_name}...")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    model.generation_config = GenerationConfig(
        max_new_tokens=150,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id
    )

    
    print("Modelo y tokenizador cargados y configurados para evaluación determinista.")
    return model, tokenizer

def load_dataset(dataset_path: Path) -> pd.DataFrame:
    try:
        print(f"Cargando dataset desde: {dataset_path}")
        return pd.read_json(dataset_path, lines=True)
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo de dataset en {dataset_path}")
        return pd.DataFrame()

def run_evaluation(model, tokenizer, dataset: pd.DataFrame) -> pd.DataFrame:
    predictions = []

    for index, row in tqdm(dataset.iterrows(), total=dataset.shape[0], desc=f"Evaluando {STUDENT_MODEL_NAME}"):
        prompt = row['prompt']
        true_score = row['response']
        
        messages = [{"role": "user", "content": prompt}]
        text_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        model_inputs = tokenizer([text_input], return_tensors="pt").to(model.device)
        
        generated_ids = model.generate(**model_inputs)
        
        response_text = tokenizer.batch_decode(generated_ids[:, model_inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
        
        predicted_score = parse_score(response_text)
        
        predictions.append({
            'prompt': prompt,
            'true_score': true_score,
            'predicted_score': predicted_score,
            'full_response': response_text
        })
        
    return pd.DataFrame(predictions)

def calculate_and_print_metrics(results_df: pd.DataFrame):
    """Calcula y muestra las métricas de rendimiento, limpiando los datos primero."""
    # Hacemos una copia para trabajar de forma segura
    df = results_df.copy()
    df['true_score'] = df['true_score'].apply(
        lambda x: parse_score(str(x)) if isinstance(x, str) else float(x)
    )

    # Ahora que ambas columnas son numéricas (o None/NaN), eliminamos las filas inválidas.
    df.dropna(subset=['predicted_score', 'true_score'], inplace=True)
    
    if df.empty:
        print("No se pudieron calcular las métricas: no hay predicciones o valores reales válidos después de la limpieza.")
        return

    # Aseguramos que el tipo de dato sea float para sklearn
    y_true = df['true_score'].astype(float)
    y_pred = df['predicted_score'].astype(float)

    # Calculamos las métricas
    mae = mean_absolute_error(y_true, y_pred)
    spearman_corr, _ = spearmanr(y_true, y_pred)
    
    print("\n--- Resultados de la Evaluación Baseline ---")
    print(f"Error Absoluto Medio (MAE): {mae:.4f}")
    print(f"Correlación de Spearman (ρ): {spearman_corr:.4f}")
    print("-----------------------------------------")

def main():
    print("--- Script de Evaluación Zero-Shot: Baseline del Student ---")
    model, tokenizer = load_model_and_tokenizer(STUDENT_MODEL_NAME)
    test_dataset = load_dataset(TEST_DATASET_PATH)
    if test_dataset.empty:
        return
    results_df = run_evaluation(model, tokenizer, test_dataset)
    print(f"\nGuardando resultados detallados en: {STUDENT_BASELINE_RESULTS_PATH}")
    results_df.to_csv(STUDENT_BASELINE_RESULTS_PATH, index=False)
    calculate_and_print_metrics(results_df)
    print("\nEvaluación del baseline completada con éxito.")

if __name__ == "__main__":
    main()