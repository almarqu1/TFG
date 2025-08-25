# TFG_DistilMatch/scripts/clean_failures.py

"""
Script de Mantenimiento para Limpiar el Silver Set de Fallos de API.

Propósito:
Durante la generación masiva de soft labels, es inevitable que algunas llamadas
a la API fallen (por timeouts, sobrecarga, etc.). Este script revisa el archivo
JSONL de salida y elimina de forma segura todas las entradas que fueron marcadas
como 'API_FAILURE', dejando el archivo listo para un segundo intento de generación
solo con los pares pendientes.

Es una herramienta clave para hacer el pipeline de datos más robusto y resiliente.
"""
import json
import yaml
import logging
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Any
from src.utils import load_config

# CONFIGURACIÓN INICIAL Y LOGGING

# Configuración del logging para mostrar mensajes informativos en la consola.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# LÓGICA PRINCIPAL DE LIMPIEZA

def clean_jsonl_failures(file_path: Path):
    """
    Lee un archivo JSONL, filtra las líneas con 'status': 'API_FAILURE',
    y reemplaza el archivo original con la versión limpia de forma segura.

    Args:
        file_path: La ruta (Path) al archivo JSONL a limpiar.
    """
    if not file_path.exists():
        logging.warning(f"El archivo '{file_path.name}' no existe. No hay nada que limpiar.")
        return

    # Si el script falla a mitad de camino, no corrompemos el archivo original.
    temp_file_path = file_path.with_suffix(f"{file_path.suffix}.tmp")

    lines_read = 0
    lines_kept = 0
    lines_removed = 0

    logging.info(f"--- Iniciando limpieza de {file_path.name} ---")

    try:
        # Contamos las líneas para tener una barra de progreso precisa
        with open(file_path, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for _ in f)
        
        if total_lines == 0:
            logging.info("El archivo está vacío. No se requiere limpieza.")
            return

        # Leemos el original, escribimos los registros válidos en el temporal.
        with open(file_path, 'r', encoding='utf-8') as f_read, \
             open(temp_file_path, 'w', encoding='utf-8') as f_write:
            
            for line in tqdm(f_read, total=total_lines, desc="🧹 Limpiando archivo"):
                lines_read += 1
                try:
                    record = json.loads(line)
                    # Mantenemos la línea si el 'status' NO es 'API_FAILURE' para revisión.
                    if record.get("status") != "API_FAILURE":
                        f_write.write(line)
                        lines_kept += 1
                    else:
                        lines_removed += 1
                except json.JSONDecodeError:
                    # Si una línea está corrupta y no es un JSON válido, la mantenemos.
                    logging.warning(f"Línea {lines_read} mal formada (no es JSON válido). Se mantendrá.")
                    f_write.write(line)
                    lines_kept += 1
        
        # Si todo el proceso fue exitoso, reemplazamos el archivo original con el temporal.
        temp_file_path.replace(file_path)

        logging.info("--- ✅ Reporte de Limpieza Finalizado ---")
        logging.info(f"Líneas totales leídas: {lines_read}")
        logging.info(f"Líneas conservadas:    {lines_kept}")
        logging.info(f"Líneas eliminadas:     {lines_removed}")
        logging.info(f"¡Archivo '{file_path.name}' limpiado con éxito!")
        logging.info("Ahora puedes re-ejecutar 'generate_silver_set.py' para procesar los pares pendientes.")

    except Exception as e:
        logging.error(f"Ocurrió un error inesperado durante la limpieza: {e}", exc_info=True)
        # Si algo sale mal, eliminamos el archivo temporal para no dejar artefactos.
        if temp_file_path.exists():
            temp_file_path.unlink()

# PUNTO DE ENTRADA DEL SCRIPT 

if __name__ == "__main__":
    try:
        config = load_config()
        project_root = Path(__file__).resolve().parent.parent
        silver_set_path = project_root / config['data_paths']['intermediate']['silver_standard_train']
        clean_jsonl_failures(silver_set_path)
    except (FileNotFoundError, KeyError) as e:
        logging.error(f"Error de Configuración: {e}")
    except Exception as e:
        logging.error(f"Ha ocurrido un error en la ejecución del script: {e}")
