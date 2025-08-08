import pandas as pd
from sklearn.metrics import cohen_kappa_score
import argparse
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.constants import ORDERED_CATEGORIES

def main(args):
    print("Calculando el Acuerdo Inter-Anotador (Kappa de Cohen Ponderado)...")

    # Cargar los dos archivos de anotaciones
    try:
        df1 = pd.read_csv(args.file1)
        df2 = pd.read_csv(args.file2)
    except FileNotFoundError as e:
        print(f"Error: No se pudo encontrar el archivo {e.filename}")
        return

    # Fusionar los dos dataframes usando los IDs para alinear las anotaciones
    merged_df = pd.merge(
        df1, df2,
        on=['job_id', 'candidate_id'],
        suffixes=('_1', '_2')
    )

    # Filtrar solo los pares que ambos han anotado
    annotated_pairs = merged_df.dropna(subset=['category_1', 'category_2'])
    if len(annotated_pairs) == 0:
        print("No se encontraron pares anotados por ambos. Asegúrate de que los archivos son correctos.")
        return

    labels1 = annotated_pairs['category_1']
    labels2 = annotated_pairs['category_2']

    # Calcular Kappa de Cohen ponderado (cuadrático es ideal para escalas ordinales)
    kappa = cohen_kappa_score(labels1, labels2, labels=ORDERED_CATEGORIES, weights='quadratic')

    print(f"\nSe compararon {len(annotated_pairs)} pares anotados por ambos.")
    print(f"Acuerdo Inter-Anotador (Kappa de Cohen Ponderado Cuadrático): {kappa:.4f}")

    if kappa >= 0.70:
        print("✅ ¡Excelente! El acuerdo es alto (κ ≥ 0.70). El conjunto de datos es fiable.")
    elif kappa >= 0.50:
        print("⚠️ El acuerdo es moderado. Se recomienda una sesión de revisión para alinear criterios.")
    else:
        print("❌ Advertencia: El acuerdo es bajo. Es crucial revisar la guía de anotación y recalibrar.")
        
    # Opcional: Guardar los desacuerdos para una fácil revisión
    disagreements = annotated_pairs[annotated_pairs['category_1'] != annotated_pairs['category_2']]
    disagreements_file = project_root / 'data/02_test_sets/disagreements.csv'
    disagreements.to_csv(disagreements_file, index=False)
    print(f"\nSe encontraron {len(disagreements)} desacuerdos. Se han guardado en: {disagreements_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Calcula el Kappa de Cohen entre dos archivos de anotación.")
    parser.add_argument('file1', type=str, help="Ruta al primer archivo de anotaciones CSV.")
    parser.add_argument('file2', type=str, help="Ruta al segundo archivo de anotaciones CSV.")
    args = parser.parse_args()
    main(args)