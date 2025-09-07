# TFG_DistilMatch/scripts/analyze_silver_set_distribution.py

"""
Script de Análisis de Distribución del Silver Set

Este script realiza un análisis cuantitativo del 'Silver Set' generado
por el modelo Teacher. Su objetivo principal es verificar la hipótesis de
desequilibrio de clases, un paso crucial para entender los resultados del
primer experimento de fine-tuning.

Funcionalidades:
1. Carga las rutas de datos desde el `config.yaml` central.
2. Lee el archivo `silver_standard_train.jsonl`.
3. Utiliza la función `parse_score_from_string` de `src.utils` para extraer
   de forma consistente los scores de cada ejemplo.
4. Calcula y muestra en la terminal una tabla con la distribución de frecuencias
   absolutas y relativas de cada score.
5. Genera y guarda un gráfico de barras visualmente claro en la carpeta de
   reportes, que puede ser utilizado directamente en la memoria del TFG.
"""

import json
import os
import sys
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# --- Configuración del Path ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from src.utils import load_config

def main():
    """
    Función principal para ejecutar el análisis.
    """
    # 1. Cargar configuración del proyecto
    try:
        config = load_config()
        #silver_set_path = config['data_paths']['intermediate']['silver_standard_train_balanced']
        silver_set_path = config['data_paths']['intermediate']['silver_standard_train']
        output_report_dir = config['output_paths']['reports']
        #chart_filename = 'balanced_silver_set_class_distribution.png'
        chart_filename = 'silver_set_class_distribution.png'
    except Exception as e:
        print(f"❌ Error al cargar la configuración: {e}")
        return

    print(f"🔬 Analizando el archivo: {silver_set_path}")
    
    full_silver_path = os.path.join(PROJECT_ROOT, silver_set_path)
    if not os.path.exists(full_silver_path):
        print(f"❌ Error: No se encontró el archivo del Silver Set en '{full_silver_path}'.")
        return

    # 2. Procesar el archivo y extraer scores (MODIFICACIÓN CLAVE)
    scores = []
    with open(full_silver_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                # Leemos directamente el score del campo 'teacher_score'
                score = data.get('teacher_score') 
                if score is not None and isinstance(score, (int, float)):
                    scores.append(float(score))
            except json.JSONDecodeError:
                print(f"⚠️ Advertencia: Se omitió una línea mal formada en el JSONL.")
            except KeyError:
                print(f"⚠️ Advertencia: Se omitió una línea sin la clave 'teacher_score'.")

    if not scores:
        print("❌ No se pudieron extraer scores válidos del archivo. Revisa que el campo 'teacher_score' exista.")
        return
        
    print(f"\n✅ Se han procesado {len(scores)} ejemplos con scores válidos.")

    # 3. Calcular y mostrar la distribución en la terminal
    score_counts = Counter(scores)
    total_count = len(scores)

    print("\n--- Distribución de Clases en el Silver Set ---")
    df_dist = pd.DataFrame(
        [(score, count, (count / total_count) * 100) for score, count in score_counts.items()],
        columns=['Score', 'Count', 'Percentage']
    ).sort_values('Score').reset_index(drop=True)
    print(df_dist.to_string())
    print("-------------------------------------------------")

    # 4. Generar y guardar el gráfico
    full_report_dir = os.path.join(PROJECT_ROOT, output_report_dir)
    os.makedirs(full_report_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 7))
    ax = sns.barplot(x='Score', y='Count', data=df_dist, palette='viridis', order=df_dist['Score'])
    
    for p in ax.patches:
        percentage = f'{100 * p.get_height() / total_count:.1f}%'
        x = p.get_x() + p.get_width() / 2
        y = p.get_height()
        ax.annotate(percentage, (x, y), ha='center', va='bottom', fontsize=11, color='black', xytext=(0, 5), textcoords='offset points')
    
    plt.title('Distribución de Scores en el Silver Set (Desequilibrio de Clases)', fontsize=16, pad=20)
    plt.xlabel('Score Asignado por el Teacher', fontsize=12)
    plt.ylabel('Número de Ejemplos', fontsize=12)
    
    chart_path = os.path.join(full_report_dir, chart_filename)
    plt.savefig(chart_path, bbox_inches='tight')
    print(f"\n📊 Gráfico de distribución guardado en: {chart_path}")

if __name__ == '__main__':
    main()