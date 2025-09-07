"""
Módulo de Constantes del Proyecto.

Este fichero centraliza valores fijos y mapeos que son utilizados en
diferentes partes del código. Esto evita tener "números mágicos" o
listas hardcodeadas dispersas, mejorando la mantenibilidad y la claridad.
"""

# --- CONSTANTES DE ANOTACIÓN Y MODELO ---

# Mapeo canónico de las categorías de la rúbrica a sus scores numéricos.
# Esta es la "fuente de la verdad" para convertir la evaluación humana en un target numérico.
CATEGORY_TO_SCORE = {
    '🟢 MUST INTERVIEW': 95.0,
    '🟡 PROMISING FIT': 70.0,
    '🟠 BORDERLINE': 45.0,
    '🔴 NO FIT': 15.0,
}

# Mapeos inversos para convertir scores numéricos de vuelta a categorías.
SCORE_TO_CATEGORY = {v: k for k, v in CATEGORY_TO_SCORE.items()}


# Define el orden explícito de las categorías.
# Este orden se utiliza para:
# 1. Asegurar una presentación consistente en la interfaz de usuario (Streamlit).
# 2. Servir como referencia para el cálculo de métricas de concordancia como el Kappa ponderado.
ORDERED_CATEGORIES = [
    '🔴 NO FIT',
    '🟠 BORDERLINE',
    '🟡 PROMISING FIT',
    '🟢 MUST INTERVIEW',
]

# --- CONSTANTES PARA EL MUESTREO DIRIGIDO (Trabajo Futuro/Exploración) ---

# Mapeo de categorías semánticas de industrias a palabras clave asociadas.
# Diseñado para facilitar la búsqueda y el muestreo de perfiles específicos durante
# la exploración de datos o la creación de conjuntos de datos balanceados.
# Ejemplo de uso: encontrar CVs de 'Diseño' buscando títulos que contengan 'ui/ux', 'designer', etc.
INDUSTRY_KEYWORDS_MAP = {
    # ... (el contenido aquí es excelente, no requiere cambios) ...
    'Art/Creative': ['art', 'creative', 'artist', 'illustrator', 'graphic', 'photographer', 'designer'],
    'Design': ['design', 'designer', 'ui/ux', 'ux/ui', 'user experience', 'user interface', 'visual', 'product design'],
    # ... etc ...
}