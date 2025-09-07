"""
Script de Evaluación Final

Propósito:
Evaluar el rendimiento de un modelo Student fine-tuneado (con adaptadores LoRA)
en el conjunto de test "sagrado" (Golden Standard Test Set).

Pasos Clave:
1.  Carga la configuración global para obtener las rutas y nombres de modelos.
2.  Carga el modelo base (Qwen) cuantizado en 4-bit.
3.  Carga los adaptadores LoRA entrenados y los fusiona con el modelo base.
4.  Carga el Golden Standard Test Set.
5.  Itera sobre el test set, generando una predicción de score para cada par CV-Oferta.
6.  Parsea las respuestas del modelo para extraer el score numérico.
7.  Calcula un conjunto de métricas de rendimiento (MAE, Spearman, etc.)
    comparando las predicciones con las etiquetas reales (ground truth).
8.  Guarda los resultados detallados y el informe de métricas.
"""

import torch
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import os
import argparse

# Importamos nuestras utilidades
from src.utils import load_config, parse_score_from_string
from src.constants import CATEGORY_TO_SCORE, SCORE_TO_CATEGORY

# Importamos métricas de evaluación
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import spearmanr, pearsonr

def calculate_metrics(df: pd.DataFrame) -> dict:
    """Calcula y devuelve un diccionario con todas las métricas de evaluación."""
    
    # Filtramos para evitar errores si alguna predicción falló en el parseo
    eval_df = df.dropna(subset=['true_score', 'predicted_score'])
    
    if len(eval_df) == 0:
        print("Advertencia: No se pudo parsear ninguna predicción válida. No se pueden calcular métricas.")
        return {}

    y_true = eval_df['true_score']
    y_pred = eval_df['predicted_score']

    # Métricas de Regresión
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = mse ** 0.5

    # Métricas de Correlación/Ranking
    spearman_corr, _ = spearmanr(y_true, y_pred)
    pearson_corr, _ = pearsonr(y_true, y_pred)

    # Métricas Categóricas
    eval_df['true_category'] = eval_df['true_score'].map(SCORE_TO_CATEGORY)
    eval_df['predicted_category'] = eval_df['predicted_score'].apply(
        lambda score: SCORE_TO_CATEGORY.get(
            min(SCORE_TO_CATEGORY.keys(), key=lambda k: abs(k - score))
        )
    )
    accuracy = (eval_df['true_category'] == eval_df['predicted_category']).mean()

    return {
        "MAE": mae,
        "RMSE": rmse,
        "Spearman Correlation": spearman_corr,
        "Pearson Correlation": pearson_corr,
        "Categorical Accuracy": accuracy,
        "Num Valid Predictions": len(eval_df),
        "Total Samples": len(df)
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluar un modelo Student fine-tuneado.")
    parser.add_argument(
        "--model_dir", 
        type=str, 
        required=True,
        help="Directorio que contiene los adaptadores LoRA entrenados (ej. 'models/distilmatch_qwen_lora_v1/final_checkpoint')."
    )
    parser.add_argument(
        "--output_suffix",
        type=str,
        default="final_tuned",
        help="Sufijo para los archivos de salida del informe (ej. 'score_only' o 'with_reasoning')."
    )
    args = parser.parse_args()

    print("--- [Paso 1/7] Cargando configuración del proyecto ---")
    config = load_config()
    model_config = config['student_model'] # Usamos una config base para el nombre del modelo
    data_paths = config['data_paths']

    print(f"--- [Paso 2/7] Preparando tokenizer para '{model_config['base_model_name']}' ---")
    tokenizer = AutoTokenizer.from_pretrained(model_config['base_model_name'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("--- [Paso 3/7] Cargando modelo base cuantizado y fusionando adaptadores LoRA ---")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        model_config['base_model_name'],
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    print(f"   -> Cargando adaptadores desde: {args.model_dir}")
    model = PeftModel.from_pretrained(base_model, args.model_dir)
    model = model.eval() # Poner el modelo en modo de evaluación (importante)

    print(f"--- [Paso 4/7] Cargando Golden Standard Test Set desde '{data_paths['gold_standard']['test_jsonl']}' ---")
    test_dataset = load_dataset("json", data_files=data_paths['gold_standard']['test_jsonl'], split="train")

    print("--- [Paso 5/7] Generando predicciones en el test set... ---")
    predictions = []
    for example in tqdm(test_dataset, desc="Evaluando"):
        # Construimos el prompt de la misma forma que en la evaluación zero-shot
        system_message = "You are an expert HR analyst. Your task is to score the compatibility between a CV and a job offer on a scale from 0 to 100. Provide only the final score in the format 'Score: [number]'."
        user_prompt = example['prompt'] # El JSONL ya tiene el prompt formateado
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt},
        ]

        # Usamos el chat_template para asegurar el formato correcto
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

        with torch.no_grad(): # Desactivamos el cálculo de gradientes para la inferencia
            outputs = model.generate(**inputs, max_new_tokens=20)
        
        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Guardamos todo para un análisis detallado
        predictions.append({
            "prompt": user_prompt,
            "true_score": parse_score_from_string(example['response']),
            "raw_response": response_text,
            "predicted_score": parse_score_from_string(response_text)
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
    
    # Guardar predicciones detalladas
    details_path = os.path.join(output_dir, f"evaluation_details_{args.output_suffix}.csv")
    results_df.to_csv(details_path, index=False)
    print(f"✅ Predicciones detalladas guardadas en: {details_path}")

    # Guardar informe de métricas
    report_path = os.path.join(output_dir, f"evaluation_report_{args.output_suffix}.md")
    with open(report_path, "w") as f:
        f.write(f"# Informe de Evaluación Final - Modelo: {args.output_suffix}\n\n")
        f.write(pd.DataFrame([metrics]).to_markdown(index=False))
    print(f"✅ Informe de métricas guardado en: {report_path}")

if __name__ == "__main__":
    main()