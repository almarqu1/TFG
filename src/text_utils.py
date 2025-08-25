# TFG_DistilMatch/src/text_utils.py

"""
Módulo de Utilidades para la Manipulación de Texto.

Contiene funciones específicas para parsear y transformar campos de texto
que se encuentran en los datasets del proyecto.
"""
import ast
from typing import List, Any

def parse_string_list(s: Any) -> List[Any]:
    """
    Convierte de forma segura un string que representa una lista a una lista de Python.

    Esta función es robusta y maneja varios casos de entrada:
    - Si la entrada ya es una lista, la devuelve sin cambios.
    - Si es un string (ej. "['skill1', 'skill2']"), intenta evaluarlo como un literal de Python.
    - Si es NaN, None, o un string mal formado, devuelve una lista vacía para evitar errores.

    Args:
        s: El objeto a convertir.

    Returns:
        Una lista de Python.
    """
    if isinstance(s, list):
        return s
    if isinstance(s, str):
        try:
            # ast.literal_eval es la forma segura de evaluar strings que contienen
            # literales de Python (strings, números, listas, tuplas, dicts).
            # A diferencia de eval(), no ejecuta código malicioso.
            evaluated = ast.literal_eval(s)
            return evaluated if isinstance(evaluated, list) else []
        except (ValueError, SyntaxError, MemoryError):
            # Si el string no es una lista válida, devolvemos una lista vacía.
            return []
    # Para cualquier otro tipo de entrada (ej. float NaN), devolvemos lista vacía.
    return []

# NOTA: Las funciones `crear_input_oferta` y `crear_input_cv` han sido eliminadas
# durante la refactorización, ya que su lógica de formateo ha sido reemplazada
# por las funciones de formato enriquecido que se encuentran en los scripts del pipeline
# (como `scripts/prepare_unlabeled_pairs.py`), las cuales son más flexibles y
# están mejor alineadas con el formato final de los prompts.