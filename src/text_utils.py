# src/text_utils.py
import pandas as pd
import ast
from typing import List, Any

def parse_string_list(s: Any) -> List[Any]:
    """
    Convierte de forma segura un string que representa una lista a una lista de Python.
    - Si la entrada ya es una lista, la devuelve.
    - Si es un string, intenta evaluarlo.
    - En cualquier otro caso (NaN, None, error), devuelve una lista vacía.
    """
    if isinstance(s, list):
        return s
    if isinstance(s, str):
        try:
            return ast.literal_eval(s)
        except (ValueError, SyntaxError):
            return []
    return []

def crear_input_oferta(row: pd.Series) -> str:
    """Construye un string estructurado a partir de una fila de datos de una oferta."""
    exp_level = row.get('formatted_experience_level', 'No especificado')
    exp_level = 'No especificado' if pd.isna(exp_level) else exp_level

    # Limpia la descripción para evitar ruido
    description = str(row.get('description', '')).split('Job Type:')[0].strip()

    # Construye las partes del texto con etiquetas claras
    parts = [
        f"[TITLE] {row.get('title', 'N/A')}",
        f"[EXPERIENCE] {exp_level}",
        f"[SKILLS] {', '.join(map(str, row.get('skills_list', [])))}",
        f"[INDUSTRIES] {', '.join(map(str, row.get('industries_list', [])))}",
        f"[DESCRIPTION] {description}"
    ]
    # Filtra partes vacías (ej. si no hay skills) y las une
    return " ".join(part for part in parts if part.split('] ')[1])

def crear_input_cv(row: pd.Series) -> str:
    """Construye un string estructurado a partir de una fila de datos de un CV."""
    tagged_lists = [
        ("POSITION", parse_string_list(row.get('positions'))),
        ("SKILLS", parse_string_list(row.get('skills'))),
        ("EDUCATION", parse_string_list(row.get('degree_names'))),
        ("UNIVERSITY", parse_string_list(row.get('educational_institution_name'))),
        ("PREVIOUS_COMPANIES", parse_string_list(row.get('professional_company_names'))),
        ("CERTIFICATIONS", parse_string_list(row.get('certification_skills'))),
        ("LANGUAGES", parse_string_list(row.get('languages'))),
        ("EXTRACURRICULAR", parse_string_list(row.get('extra_curricular_activity_types')))
    ]
    parts = [f"[{tag}] {', '.join(map(str, items))}" for tag, items in tagged_lists if items]

    # Añade el resumen de experiencia si existe
    responsibilities = row.get('responsibilities')
    if responsibilities and pd.notna(responsibilities):
        parts.append(f"[EXPERIENCE_SUMMARY] {responsibilities}")
        
    return " ".join(parts)