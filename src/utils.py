# TFG_DistilMatch/src/utils.py

"""
Módulo de Utilidades Compartidas

Este fichero centraliza funciones auxiliares que son utilizadas por múltiples
scripts a lo largo del proyecto. El objetivo es seguir el principio DRY
(Don't Repeat Yourself), mejorar la mantenibilidad y asegurar que la lógica
común (como la carga de configuración o el parseo de respuestas) sea consistente.
"""

import yaml
import re
import json
from pathlib import Path
from typing import Dict, Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def load_config() -> Dict[str, Any]:
    """
    Carga el archivo de configuración principal (config.yaml).
    """
    config_path = PROJECT_ROOT / "config" / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Archivo de configuración no encontrado en: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def load_prompt_template(prompt_id: str) -> str:
    """
    Carga la plantilla de un prompt específico desde el catálogo de prompts.
    """
    prompts_config_path = PROJECT_ROOT / "config" / "experiment_prompts.yaml"
    with open(prompts_config_path, 'r', encoding='utf-8') as f:
        prompts_config = yaml.safe_load(f)
    
    prompt_info = prompts_config.get(prompt_id)
    if not prompt_info:
        raise ValueError(f"Prompt ID '{prompt_id}' no encontrado en {prompts_config_path}")

    prompt_file_path = PROJECT_ROOT / prompt_info['path']
    with open(prompt_file_path, 'r', encoding='utf-8') as f:
        return f.read()

def parse_score_from_string(text_response: str) -> Optional[float]:
    """
    Extrae un score numérico de una cadena de texto (ej. "Score: 95.0").
    """
    if not isinstance(text_response, str):
        return None
    match = re.search(r"Score:\s*(\d+\.?\d*)", text_response, re.IGNORECASE)
    if match:
        try:
            return float(match.group(1))
        except (ValueError, IndexError):
            return None
    return None

def parse_json_from_response(raw_text: str) -> Optional[Dict[str, Any]]:
    """
    Extrae un bloque JSON de una cadena de texto, incluso si está en un bloque de código.
    """
    if not isinstance(raw_text, str): return None
    text = raw_text.strip()
    if text.startswith("```"):
        text = '\n'.join(text.split('\n')[1:-1])
    try:
        json_start = text.find('{')
        json_end = text.rfind('}')
        if json_start == -1 or json_end == -1: return None
        return json.loads(text[json_start : json_end + 1])
    except (json.JSONDecodeError, ValueError):
        return None