# TFG_DistilMatch/src/evaluate_reasoning.py
# VERSIÓN ADAPTADA PARA MODELOS ENTRENADOS CON "EXPLANATION-TUNING"

"""
Script de Evaluación para Modelos de Razonamiento

Propósito:
Evaluar un modelo Student que fue entrenado para generar una justificación
completa además del score (como el modelo v4).

Diferencias Clave con evaluate.py:
1.  Carga el prompt específico usado durante el entrenamiento (ej. S-04).
2.  Construye el input para la inferencia replicando el formato exacto del
    entrenamiento (formato de "relleno de huecos").
3.  Aumenta significativamente `max_new_tokens` para permitir que el modelo
    genere la justificación completa sin ser cortado.
"""

import torch
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import os
import argparse
import sys

# --- Configuración del Path ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

# Importamos nuestras utilidades
from src.utils import load_config, parse_score_from_string, load_prompt_template
from src.constants import CATEGORY_TO_SCORE, SCORE_TO_CATEGORY

# Importamos métricas (sin cambios aquí)
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import spearmanr, pearsonr

# La función calculate_metrics es idéntica a la de evaluate.py, la mantenemos.
def calculate_metrics(df: pd.DataFrame) -> dict:
    """Calcula y devuelve un diccionario con todas las métricas de evaluación."""
    eval_df = df.dropna(subset=['true_score', 'predicted_score'])
    if len(eval_df) == 0:
        print("Advertencia: No se pudo parsear ninguna predicción válida. No se pueden calcular métricas.")
        return { "Num Valid Predictions": 0, "Total Samples": len(df) }
    y_true = eval_df['true_score']
    y_pred = eval_df['predicted_score']
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = mse ** 0.5
    spearman_corr, _ = spearmanr(y_true, y_pred)
    pearson_corr, _ = pearsonr(y_true, y_pred)
    eval_df['true_category'] = eval_df['true_score'].map(SCORE_TO_CATEGORY)
    eval_df['predicted_category'] = eval_df['predicted_score'].apply(
        lambda score: SCORE_TO_CATEGORY.get(min(SCORE_TO_CATEGORY.keys(), key=lambda k: abs(k - score)))
    )
    accuracy = (eval_df['true_category'] == eval_df['predicted_category']).mean()
    return {
        "MAE": mae, "RMSE": rmse, "Spearman Correlation": spearman_corr,
        "Pearson Correlation": pearson_corr, "Categorical Accuracy": accuracy,
        "Num Valid Predictions": len(eval_df), "Total Samples": len(df)
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluar un modelo Student fine-tuneado para razonamiento.")
    parser.add_argument("--model_dir", type=str, required=True, help="Directorio con los adaptadores LoRA entrenados.")
    # --- NUEVO ARGUMENTO ---
    # Necesitamos saber qué prompt usar para la evaluación.
    parser.add_argument("--prompt_id", type=str, required=True, help="ID del prompt usado en el entrenamiento (ej. S-04_student_reasoning_structured).")
    parser.add_argument("--output_suffix", type=str, default="reasoning_model", help="Sufijo para los archivos de salida.")
    args = parser.parse_args()

    print("--- [Paso 1/7] Cargando configuración del proyecto ---")
    config = load_config()
    model_config = config['student_model']
    data_paths = config['data_paths']

    print(f"--- [Paso 2/7] Preparando tokenizer para '{model_config['base_model_name']}' ---")
    tokenizer = AutoTokenizer.from_pretrained(model_config['base_model_name'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("--- [Paso 3/7] Cargando modelo base y fusionando adaptadores LoRA ---")
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    base_model = AutoModelForCausalLM.from_pretrained(model_config['base_model_name'], quantization_config=bnb_config, device_map="auto")
    print(f"   -> Cargando adaptadores desde: {args.model_dir}")
    model = PeftModel.from_pretrained(base_model, args.model_dir)
    model = model.eval()

    print(f"--- [Paso 4/7] Cargando Golden Standard Test Set desde '{data_paths['gold_standard']['test_jsonl']}' ---")
    test_dataset = load_dataset("json", data_files=data_paths['gold_standard']['test_jsonl'], split="train")

    # --- LÓGICA DE PROMPTING MODIFICADA ---
    print(f"--- [Paso 5/7] Generando predicciones con el prompt '{args.prompt_id}' ---")
    prompt_template = load_prompt_template(args.prompt_id)
    
    predictions = []
    for example in tqdm(test_dataset, desc="Evaluando con razonamiento"):
        # El `example['prompt']` del golden set contiene el texto formateado de CV y Oferta.
        # Lo usamos para rellenar la plantilla de razonamiento.
        # La plantilla S-04 contiene los placeholders {cv} y {job_description}. El prompt del golden set
        # contiene el texto completo. Para evitar problemas, simplemente concatenamos la instrucción
        # de la plantilla con el prompt del golden set.
        
        # Extraemos la parte de la instrucción y el inicio de la respuesta esperada
        instruction_part = prompt_template.split("[CV]")[0]
        full_prompt_text = instruction_part + example['prompt'] + "\n\n[ANALYSIS]\n"
        
        inputs = tokenizer(full_prompt_text, return_tensors="pt").to(model.device)

        # --- MAX_NEW_TOKENS MODIFICADO ---
        # Aumentamos el límite para dar espacio a la justificación completa.
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=512, pad_token_id=tokenizer.eos_token_id)
        
        # Decodificamos solo la parte generada, no el input
        generated_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        # La respuesta completa es lo que el modelo generó.
        full_response_text = "[ANALYSIS]\n" + generated_text

        predictions.append({
            "prompt": full_prompt_text,
            "true_score": parse_score_from_string(example['response']),
            "raw_response": full_response_text,
            "predicted_score": parse_score_from_string(full_response_text)
        })

    results_df = pd.DataFrame(predictions)

    print("--- [Paso 6/7] Calculando métricas de rendimiento ---")
    metrics = calculate_metrics(results_df)
    print("\n--- INFORME DE EVALUACIÓN FINAL ---")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")
    print("-------------------------------------\n")

    print("--- [Paso 7/7] Guardando resultados detallados e informe ---")
    output_dir = config['output_paths']['reports']
    os.makedirs(output_dir, exist_ok=True)
    details_path = os.path.join(output_dir, f"evaluation_details_{args.output_suffix}.csv")
    results_df.to_csv(details_path, index=False)
    print(f"✅ Predicciones detalladas guardadas en: {details_path}")
    report_path = os.path.join(output_dir, f"evaluation_report_{args.output_suffix}.md")
    with open(report_path, "w") as f:
        f.write(f"# Informe de Evaluación - Modelo: {args.output_suffix}\n\n")
        f.write(pd.DataFrame([metrics]).to_markdown(index=False))
    print(f"✅ Informe de métricas guardado en: {report_path}")

if __name__ == "__main__":
    main()
