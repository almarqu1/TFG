import streamlit as st
import pandas as pd
import ast
from pathlib import Path
import sys

# --- CONFIGURACIÓN Y CONSTANTES ---
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.constants import CATEGORY_TO_SCORE, ORDERED_CATEGORIES
from src.text_utils import parse_string_list

st.set_page_config(layout="wide", page_title="DistilMatch Annotation Tool")


# Color para los encabezados de las secciones de datos para mejorar la legibilidad.
HEADER_COLOR = "#17A2B8"

# Rutas de los archivos de datos.
DATA_DIR = project_root / 'data' # CAMBIO: Usar project_root para rutas más robustas
PROCESSED_DATA_DIR = DATA_DIR / '01_processed'
TEST_SETS_DIR = DATA_DIR / '02_test_sets'
OFFERS_FILE = PROCESSED_DATA_DIR / 'offers_processed.csv'
CVS_FILE = PROCESSED_DATA_DIR / 'cvs_processed.csv'
PAIRS_FILE = TEST_SETS_DIR / 'test_set_pairs_to_annotate.csv'

# --- FUNCIONES DE AYUDA Y CARGA DE DATOS ---



@st.cache_data
def load_data():
    """Carga los DataFrames de ofertas y CVs, convirtiendo las columnas de listas."""
    try:
        converters = {'skills_list': parse_string_list, 'industries_list': parse_string_list}
        offers_df = pd.read_csv(OFFERS_FILE, converters=converters)
        cvs_df = pd.read_csv(CVS_FILE)
        offers_df['job_id'] = offers_df['job_id'].astype(str)
        cvs_df['candidate_id'] = cvs_df['candidate_id'].astype(str)
        return offers_df, cvs_df
    except FileNotFoundError as e:
        st.error(f"Error al cargar datos: {e.filename}. Asegúrate de que los archivos existen.")
        return None, None

def display_list_as_bullets(title, items, color):
    """Muestra una lista como viñetas en Markdown, con un encabezado de color."""
    if items and isinstance(items, list) and len(items) > 0:
        st.markdown(f'<strong style="color: {color};">{title}</strong>', unsafe_allow_html=True)
        st.markdown("\n".join([f"- {item}" for item in items]))

# --- INICIALIZACIÓN DEL ESTADO DE LA APP ---

if 'current_index' not in st.session_state:
    st.session_state.current_index = 0
    try:
        annotations_df = pd.read_csv(PAIRS_FILE)
        annotations_df['job_id'] = annotations_df['job_id'].astype(str)
        annotations_df['candidate_id'] = annotations_df['candidate_id'].astype(str)
        st.session_state.annotations_df = annotations_df
        for col in ['category', 'score', 'annotator_id', 'justification']:
            if col not in annotations_df.columns:
                annotations_df[col] = ''
            else:
                annotations_df[col] = annotations_df[col].fillna('')
    except FileNotFoundError:
        st.error(f"Error: No se encontró '{PAIRS_FILE.name}'. Ejecuta `src/create_test_set.py` primero.")
        st.stop()

# --- INTERFAZ PRINCIPAL ---

st.title("DistilMatch - Herramienta de Anotación  📝")
st.markdown("Revisa cada par de CV-Oferta y asigna una categoría de compatibilidad.")

offers, cvs = load_data()
if offers is None or 'annotations_df' not in st.session_state:
    st.stop()

# --- BARRA LATERAL (SIDEBAR) PARA CONTROLES ---
with st.sidebar:
    st.header("Control Panel")
    annotator_id = st.text_input("Tu ID de Anotador:", key="annotator_id_input")
    total_pairs = len(st.session_state.annotations_df)
    annotated_count = (st.session_state.annotations_df['category'] != '').sum()
    st.metric("Progreso", f"{annotated_count} / {total_pairs}")
    st.progress(annotated_count / total_pairs)
    st.divider()
    output_filename = f"annotations_{annotator_id}.csv" if annotator_id else "annotations_output.csv"
    st.download_button(
        label="📥 Descargar Anotaciones (CSV)",
        data=st.session_state.annotations_df.to_csv(index=False).encode('utf-8'),
        file_name=output_filename, mime='text/csv'
    )
    st.caption("Guarda tu progreso descargando el archivo.")

if not annotator_id:
    st.warning("Por favor, introduce tu ID de Anotador en la barra lateral para continuar.")
    st.stop()

# --- VISUALIZACIÓN DE DATOS ---

idx = st.session_state.current_index
if idx >= total_pairs:
    st.success("¡Felicidades! 🎉 Has anotado todos los pares.")
    st.balloons()
    st.stop()

current_pair = st.session_state.annotations_df.iloc[idx]
job_id, candidate_id = current_pair['job_id'], current_pair['candidate_id']
offer_details = offers[offers['job_id'] == job_id].iloc[0]
cv_details = cvs[cvs['candidate_id'] == candidate_id].iloc[0]

col1, col2 = st.columns(2)
with col1:
    st.subheader("📄 Oferta de Trabajo")
    st.markdown(f"**ID:** `{job_id}`")
    st.markdown(f"### {offer_details.get('title', 'N/A')}")
    
    # Renderiza el encabezado "Nivel de Experiencia" con color.
    exp_level = offer_details.get('formatted_experience_level', 'No especificado')
    st.markdown(f'<strong style="color: {HEADER_COLOR};">Nivel de Experiencia:</strong> {exp_level}', unsafe_allow_html=True)
    
    # Llama a la función de ayuda para mostrar listas con encabezados de colores.
    display_list_as_bullets("Skills Requeridas:", offer_details.get('skills_list'), color=HEADER_COLOR)
    display_list_as_bullets("Industrias:", offer_details.get('industries_list'), color=HEADER_COLOR)
    
    with st.expander("Ver descripción completa"):
        st.markdown(offer_details.get('description', 'No hay descripción.'))

with col2:
    st.subheader("👤 Currículum Vitae")
    st.markdown(f"**ID:** `{candidate_id}`")
    
    display_list_as_bullets("Posiciones:", parse_string_list(cv_details.get('positions')), color=HEADER_COLOR)
    display_list_as_bullets("Educación:", parse_string_list(cv_details.get('degree_names')), color=HEADER_COLOR)
    display_list_as_bullets("Universidades:", parse_string_list(cv_details.get('educational_institution_name')), color=HEADER_COLOR)
    st.markdown("---")
    
    with st.expander("Ver Skills, Empresas y Certificaciones"):
        display_list_as_bullets("Skills:", parse_string_list(cv_details.get('skills')), color=HEADER_COLOR)
        display_list_as_bullets("Empresas Anteriores:", parse_string_list(cv_details.get('professional_company_names')), color=HEADER_COLOR)
        display_list_as_bullets("Certificaciones:", parse_string_list(cv_details.get('certification_skills')), color=HEADER_COLOR)
        display_list_as_bullets("Idiomas:", parse_string_list(cv_details.get('languages')), color=HEADER_COLOR)
    
    responsibilities = cv_details.get('responsibilities')
    if responsibilities and isinstance(responsibilities, str):
        with st.expander("Ver Resumen de Experiencia"):
            st.markdown(responsibilities)

st.divider()

# --- FORMULARIO DE ANOTACIÓN Y NAVEGACIÓN ---
st.subheader(f"Tu Evaluación (Par {idx + 1}/{total_pairs})")

current_annotation = st.session_state.annotations_df.iloc[idx]
default_category_index = ORDERED_CATEGORIES.index(current_annotation['category']) if current_annotation['category'] in ORDERED_CATEGORIES else 2 # 'MAYBE' por defecto

category = st.radio(
    "Selecciona la categoría de compatibilidad:", options=ORDERED_CATEGORIES, index=default_category_index,
    horizontal=True, key=f"cat_{idx}"
)
justification = st.text_area(
    "Justificación (opcional pero recomendado):", value=current_annotation.get('justification', ''), key=f"just_{idx}"
)

nav_col1, nav_col2, nav_col3 = st.columns([1, 2, 1])
with nav_col1:
    if st.button("⬅️ Anterior", use_container_width=True, disabled=(idx == 0)):
        st.session_state.current_index -= 1
        st.rerun()

with nav_col2:
    if st.button("💾 Guardar y Siguiente ➡️", type="primary", use_container_width=True):
        st.session_state.annotations_df.loc[idx, 'category'] = category
        st.session_state.annotations_df.loc[idx, 'score'] = CATEGORY_TO_SCORE[category]
        st.session_state.annotations_df.loc[idx, 'justification'] = justification
        st.session_state.annotations_df.loc[idx, 'annotator_id'] = annotator_id
        st.session_state.current_index += 1
        st.rerun()

with nav_col3:
    if st.button("Saltar ➡️", use_container_width=True):
        st.session_state.current_index += 1
        st.rerun()