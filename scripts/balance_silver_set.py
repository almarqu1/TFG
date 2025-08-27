# TFG_DistilMatch/scripts/balance_silver_set.py

"""
Script para Balancear el Silver Set de Entrenamiento

Propósito:
Abordar el problema de desequilibrio de clases extremo (95% 'NO FIT')
identificado en el Silver Set generado por el Teacher. Este script crea
una nueva versión del dataset de entrenamiento utilizando la técnica de
submuestreo aleatorio (Random Undersampling) de la clase mayoritaria.

El resultado es un dataset balanceado que fuerza al modelo Student a aprender
los patrones de todas las categorías, en lugar de colapsar a la predicción
más frecuente.
"""

import json
import os
import sys
import pandas as pd
import random

# --- Configuración del Path ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from src.utils import load_config

def main():
    """
    Función principal que carga, balancea y guarda el dataset.
    """
    # 1. Cargar Configuración
    try:
        config = load_config()
        input_path = config['data_paths']['intermediate']['silver_standard_train']
        # Definimos una nueva ruta de salida para el dataset balanceado
        output_path = os.path.join(os.path.dirname(input_path), "silver_standard_train_balanced.jsonl")
    except Exception as e:
        print(f"❌ Error al cargar la configuración: {e}")
        return

    print(f"⚖️ Iniciando el balanceo del dataset: {input_path}")

    # 2. Cargar el Dataset Desequilibrado
    full_input_path = os.path.join(PROJECT_ROOT, input_path)
    if not os.path.exists(full_input_path):
        print(f"❌ Error: No se encontró el archivo de entrada en '{full_input_path}'.")
        return

    # Usamos una lista de diccionarios para mantener el formato JSONL
    data = []
    with open(full_input_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))

    df = pd.DataFrame(data)
    print(f"Cargados {len(df)} ejemplos. Distribución original:")
    print(df['teacher_score'].value_counts(normalize=True).sort_index())

    # 3. Lógica de Balanceo (Random Undersampling)
    # Identificar la clase con el menor número de muestras (sin contar la mayoritaria)
    counts = df['teacher_score'].value_counts()
    majority_class_score = counts.idxmax()
    
    # Encontremos el tamaño de la segunda clase más pequeña como objetivo
    # para no perder demasiados datos si una clase es muy muy pequeña.
    # Por ejemplo, si tenemos 10 '95' y 200 '70', apuntamos a 200.
    target_samples_per_class = counts.drop(majority_class_score).max()
    print(f"\nObjetivo: ~{target_samples_per_class} muestras por clase.")

    balanced_dfs = []
    for score in df['teacher_score'].unique():
        class_df = df[df['teacher_score'] == score]
        if score == majority_class_score:
            # Submuestreamos la clase mayoritaria
            sample_df = class_df.sample(n=target_samples_per_class, random_state=config['global_seed'])
            balanced_dfs.append(sample_df)
        else:
            # Mantenemos todas las muestras de las clases minoritarias
            balanced_dfs.append(class_df)

    df_balanced = pd.concat(balanced_dfs).sample(frac=1, random_state=config['global_seed']).reset_index(drop=True)
    
    print("\nDataset balanceado exitosamente. Nueva distribución:")
    print(df_balanced['teacher_score'].value_counts(normalize=True).sort_index())
    print(f"Tamaño total del nuevo dataset: {len(df_balanced)} ejemplos.")

    # 4. Guardar el Nuevo Dataset Balanceado en formato JSONL
    full_output_path = os.path.join(PROJECT_ROOT, output_path)
    with open(full_output_path, 'w', encoding='utf-8') as f:
        for record in df_balanced.to_dict('records'):
            f.write(json.dumps(record) + '\n')
    
    print(f"\n✅ Dataset balanceado guardado en: {full_output_path}")
    print("\nAhora puedes usar esta ruta en tu config.yaml o script de entrenamiento.")

if __name__ == "__main__":
    main()