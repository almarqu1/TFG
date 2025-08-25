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
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, Callable

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
    
def format_entity_text(row: pd.Series, sections: Dict[str, Any]) -> str:
    """
    Función genérica para construir un bloque de texto estructurado a partir de una fila.

    Args:
        row: Una fila (pd.Series) de un DataFrame.
        sections: Un diccionario que define las secciones del texto.
                  Las claves son los títulos de sección.
                  Los valores pueden ser un string (nombre de columna) o una
                  función lambda que formatea varias columnas.
    """
    parts = []
    
    def add_section(title: str, content: Any):
        # Solo añadimos la sección si el contenido no es nulo, NaN, o una cadena vacía.
        if pd.notna(content) and str(content).strip():
            parts.append(f"--- {title.upper()} ---\n{str(content).strip()}")

    for title, source in sections.items():
        try:
            if isinstance(source, str): # Si es un simple nombre de columna
                add_section(title, row.get(source))
            elif callable(source): # Si es una función de formateo
                add_section(title, source(row))
        except Exception:
            # Captura errores si una columna esperada por una lambda no existe, etc.
            # Esto hace la función más robusta.
            continue
            
    return "\n\n".join(parts)

def get_cv_sections() -> Dict[str, Any]:
    """Define la estructura y el contenido de la sección de CV."""
    return {
        "Total Years of Experience": 'total_experience_years',
        "Career Objective": 'career_objective',
        "Formatted Work History": 'formatted_work_history',
        "Skills": 'skills',
        "Education": lambda r: (
            f"Institución: {r.get('educational_institution_name', 'N/A')}\n"
            f"Grado: {r.get('degree_names', 'N/A')}\n"
            f"Especialidad: {r.get('major_field_of_studies', 'N/A')}"
        ),
        "Languages": 'languages',
        "Certifications": 'certification_skills',
    }
    
def get_offer_sections() -> Dict[str, Any]:
    """Define la estructura y el contenido de la sección de Oferta."""
    return {
        "Job Details": lambda r: (
            f"Title: {r.get('title', 'N/A')}\n"
            f"Experience Level: {r.get('formatted_experience_level', 'N/A')}\n"
            f"Work Type: {r.get('formatted_work_type', 'N/A')}"
        ),
        "Description": 'description',
        "Required Skills": 'skills_list',
        "Industries": 'industries_list',
    }