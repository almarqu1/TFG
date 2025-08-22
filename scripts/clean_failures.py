import json
import os
import yaml
from pathlib import Path
from tqdm import tqdm

def cargar_ruta_salida() -> Path:
    """
    Carga la configuración para obtener la ruta del archivo JSONL que necesita limpieza.
    """
    project_root = Path(__file__).resolve().parent.parent
    config_path = project_root / "config" / "config.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de configuración en: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Validar que la ruta esperada exista en la config
    try:
        output_relative_path = config['silver_set_generation']['output_jsonl_path']
        return project_root / output_relative_path
    except KeyError as e:
        raise KeyError(f"La clave {e} no se encontró en el config.yaml. Asegúrate de que la ruta 'silver_set_generation.output_jsonl_path' esté definida.")

def limpiar_jsonl(file_path: Path):
    """
    Lee un archivo JSONL, elimina las líneas con 'status': 'API_FAILURE',
    y reemplaza el archivo original con la versión limpia.
    """
    if not file_path.exists():
        print(f"🤷‍♂️ El archivo de entrada no existe en '{file_path}'. No hay nada que limpiar.")
        return

    # Usamos un archivo temporal para la nueva versión limpia
    temp_file_path = file_path.with_suffix(f"{file_path.suffix}.tmp")

    lines_read = 0
    lines_kept = 0
    lines_removed = 0

    print(f"--- Iniciando limpieza de {file_path.name} ---")

    try:
        # Contar líneas totales para la barra de progreso
        with open(file_path, 'r', encoding='utf-8') as f_read:
            total_lines = sum(1 for line in f_read)

        # Leer, filtrar y escribir en el archivo temporal
        with open(file_path, 'r', encoding='utf-8') as f_read, \
             open(temp_file_path, 'w', encoding='utf-8') as f_write:
            
            for line in tqdm(f_read, total=total_lines, desc="🧹 Limpiando archivo"):
                lines_read += 1
                try:
                    record = json.loads(line)
                    # La condición clave: Mantenemos la línea si el status NO es API_FAILURE
                    if record.get("status") != "API_FAILURE":
                        f_write.write(line)
                        lines_kept += 1
                    else:
                        lines_removed += 1
                except json.JSONDecodeError:
                    # Si una línea está corrupta, la mantenemos por si acaso para revisión manual
                    print(f"\nAdvertencia: Se encontró una línea mal formada en la línea {lines_read}. Se mantendrá en el archivo.")
                    f_write.write(line)
                    lines_kept += 1
        
        # Reemplazar el archivo original con el temporal solo si todo ha ido bien
        os.replace(temp_file_path, file_path)

        print("\n--- Reporte de Limpieza ---")
        print(f"Líneas totales leídas: {lines_read}")
        print(f"✅ Líneas conservadas:    {lines_kept}")
        print(f"❌ Líneas eliminadas:     {lines_removed}")
        print(f"✨ ¡Archivo '{file_path.name}' limpiado con éxito! ✨")
        print("Ahora puedes volver a ejecutar el script principal para procesar los fallos.")

    except Exception as e:
        print(f"\n❌ Ocurrió un error durante la limpieza: {e}")
        # Si algo falla, eliminamos el archivo temporal para no dejar basura
        if temp_file_path.exists():
            os.remove(temp_file_path)

if __name__ == "__main__":
    try:
        jsonl_path = cargar_ruta_salida()
        limpiar_jsonl(jsonl_path)
    except (FileNotFoundError, KeyError) as e:
        print(f"❌ Error de configuración: {e}")
