import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# --- CONFIGURACIÓN Y CONSTANTES ---

# Añade el directorio raíz del proyecto al path para poder importar módulos de 'src'
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.constants import CATEGORY_TO_SCORE, ORDERED_CATEGORIES
from src.text_utils import parse_string_list

# Configuración de la página de Streamlit para usar todo el ancho
st.set_page_config(layout="wide", page_title="DistilMatch Annotation Tool")

# Constantes de la UI para mantener un estilo consistente
HEADER_COLOR = "#17A2B8"

# Rutas a los archivos de datos, construidas de forma robusta con pathlib
DATA_DIR = project_root / 'data'
PROCESSED_DATA_DIR = DATA_DIR / '01_processed'
TEST_SETS_DIR = DATA_DIR / '02_test_sets'
OFFERS_FILE = PROCESSED_DATA_DIR / 'offers_processed.csv'
CVS_FILE = PROCESSED_DATA_DIR / 'cvs_processed.csv'
PAIRS_FILE = TEST_SETS_DIR / 'test_set_pairs_to_annotate.csv'

# --- FUNCIONES DE AYUDA Y CARGA DE DATOS ---

@st.cache_data
def load_data():
    """
    Carga los DataFrames de ofertas y CVs desde los archivos CSV procesados.
    Utiliza el decorador @st.cache_data para optimizar el rendimiento, evitando
    recargar los datos en cada re-ejecución del script.
    """
    try:
        converters = {'skills_list': parse_string_list, 'industries_list': parse_string_list}
        offers_df = pd.read_csv(OFFERS_FILE, converters=converters, dtype={'job_id': str})
        cvs_df = pd.read_csv(CVS_FILE, dtype={'candidate_id': str})
        return offers_df, cvs_df
    except FileNotFoundError as e:
        st.error(f"Error al cargar datos: {e.filename}. Asegúrate de que los archivos procesados existen.")
        return None, None

def display_list_as_bullets(title, items, color):
    """
    Renderiza una lista de Python como una lista de viñetas en Markdown con un
    encabezado coloreado para mejorar la legibilidad.
    """
    if items and isinstance(items, list) and len(items) > 0:
        st.markdown(f'<strong style="color: {color};">{title}</strong>', unsafe_allow_html=True)
        st.markdown("\n".join([f"- {item}" for item in items]))

# --- INICIALIZACIÓN DEL ESTADO DE LA APLICACIÓN ---

# Este bloque se ejecuta solo una vez por sesión de usuario para inicializar el estado.
if 'initialized' not in st.session_state:
    try:
        # Carga la plantilla base de pares a anotar
        base_pairs_df = pd.read_csv(PAIRS_FILE, dtype={'job_id': str, 'candidate_id': str})

        # Prepara el DataFrame con columnas vacías para las futuras anotaciones
        for col in ['category', 'score', 'annotator_id', 'justification']:
            base_pairs_df[col] = ''
        
        # Guarda el DataFrame y el índice inicial en el estado de la sesión
        st.session_state.annotations_df = base_pairs_df
        st.session_state.current_index = 0
        st.session_state.initialized = True  # Marca como inicializado
        st.session_state.last_processed_file = None # Bandera para la carga de archivos

    except FileNotFoundError:
        st.error(f"Error: No se encontró '{PAIRS_FILE.name}'. Ejecuta `scripts/create_test_set.py` primero.")
        st.stop()


# --- INTERFAZ PRINCIPAL ---

st.title("DistilMatch - Herramienta de Anotación 📝")
st.markdown("Revisa cada par de CV-Oferta y asigna una categoría de compatibilidad.")

offers, cvs = load_data()
if offers is None:
    st.stop()

# --- BARRA LATERAL (SIDEBAR) PARA CONTROLES ---
with st.sidebar:
    st.header("Control Panel")
    annotator_id = st.text_input("Tu ID de Anotador:", key="annotator_id_input")

    st.divider()
    st.subheader("Retomar Progreso")
    
    # Widget para que el usuario suba su archivo de progreso
    uploaded_file = st.file_uploader(
        "Carga tu archivo de anotaciones (.csv)",
        type=['csv']
    )

    # Para evitar reprocesar el mismo archivo en cada rerun, se usa una bandera en session_state.
    # El bloque solo se ejecuta si se carga un archivo NUEVO.
    if uploaded_file is not None and uploaded_file.name != st.session_state.get('last_processed_file'):
        try:
            # 1. Cargar el CSV subido por el usuario
            loaded_annotations_df = pd.read_csv(uploaded_file, dtype={'job_id': str, 'candidate_id': str})
            
            # 2. Cargar la plantilla base para asegurar que la estructura es correcta
            base_pairs_df = pd.read_csv(PAIRS_FILE, dtype={'job_id': str, 'candidate_id': str})
            
            # 3. Fusionar la plantilla con las anotaciones cargadas
            cols_to_merge = ['job_id', 'candidate_id', 'category', 'score', 'justification', 'annotator_id']
            relevant_loaded_df = loaded_annotations_df[[col for col in cols_to_merge if col in loaded_annotations_df.columns]]
            merged_df = pd.merge(base_pairs_df[['job_id', 'candidate_id']], relevant_loaded_df, on=['job_id', 'candidate_id'], how='left')
            
            # 4. Limpiar los valores NaN que resultan del merge y rellenar columnas
            for col in ['category', 'annotator_id', 'justification']:
                merged_df[col] = merged_df[col].replace({np.nan: ''})
            merged_df['score'] = merged_df['score'].apply(lambda x: '' if pd.isna(x) else x)

            # 5. Actualizar el estado de la sesión con los datos cargados
            st.session_state.annotations_df = merged_df
            st.session_state.last_processed_file = uploaded_file.name  # Actualizar la bandera

            # 6. Mover el índice actual al primer par que necesita ser anotado
            annotated_mask = (st.session_state.annotations_df['category'] != '') & (st.session_state.annotations_df['category'].notna())
            st.session_state.current_index = 0 if annotated_mask.all() else annotated_mask.idxmin()

            st.success("✅ Progreso cargado. Saltando al primer par pendiente.")
            st.rerun()  # Forzar un refresco de la UI para mostrar el nuevo estado

        except Exception as e:
            st.error(f"Error al procesar el archivo: {e}")
            st.session_state.last_processed_file = None # Resetear la bandera en caso de error

    st.divider()
    
    # Visualización del progreso de anotación
    total_pairs = len(st.session_state.annotations_df)
    annotated_count = ((st.session_state.annotations_df['category'] != '') & (st.session_state.annotations_df['category'].notna())).sum()
    st.metric("Progreso", f"{annotated_count} / {total_pairs}")
    if total_pairs > 0:
        st.progress(annotated_count / total_pairs)
    st.divider()

    # Botón para descargar el estado actual de las anotaciones
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

# --- VISUALIZACIÓN DE DATOS DEL PAR ACTUAL ---
idx = st.session_state.current_index
if idx >= len(st.session_state.annotations_df):
    st.success("¡Felicidades! 🎉 Has anotado todos los pares.")
    st.balloons()
    st.stop()

# Recupera los IDs del par actual
current_pair = st.session_state.annotations_df.iloc[idx]
job_id, candidate_id = current_pair['job_id'], current_pair['candidate_id']

# Busca los detalles completos en los DataFrames principales
offer_details = offers[offers['job_id'] == job_id].iloc[0]
cv_details = cvs[cvs['candidate_id'] == candidate_id].iloc[0]

# Estructura en dos columnas para la visualización de la oferta y el CV
col1, col2 = st.columns(2)
with col1:
    st.subheader("📄 Oferta de Trabajo")
    st.markdown(f"**ID:** `{job_id}`")
    st.markdown(f"### {offer_details.get('title', 'N/A')}")
    exp_level = offer_details.get('formatted_experience_level', 'No especificado')
    st.markdown(f'<strong style="color: {HEADER_COLOR};">Nivel de Experiencia:</strong> {exp_level}', unsafe_allow_html=True)
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
    if cv_details.get('responsibilities'):
        with st.expander("Ver Resumen de Experiencia"):
            st.markdown(cv_details.get('responsibilities'))

st.divider()

# --- FORMULARIO DE ANOTACIÓN Y NAVEGACIÓN ---
st.subheader(f"Tu Evaluación (Par {idx + 1}/{total_pairs})")

# Popula el formulario con los datos guardados para el par actual, si existen
current_annotation = st.session_state.annotations_df.iloc[idx]
default_category_index = 2 # 'MAYBE' por defecto
if current_annotation.get('category') in ORDERED_CATEGORIES:
    default_category_index = ORDERED_CATEGORIES.index(current_annotation['category'])

category = st.radio(
    "Selecciona la categoría de compatibilidad:", options=ORDERED_CATEGORIES, index=default_category_index,
    horizontal=True, key=f"cat_{idx}"
)
justification = st.text_area(
    "Justificación (opcional pero recomendado):", value=current_annotation.get('justification', ''), key=f"just_{idx}"
)

# Botones de navegación en tres columnas
nav_col1, nav_col2, nav_col3 = st.columns([1, 2, 1])
with nav_col1:
    if st.button("⬅️ Anterior", use_container_width=True, disabled=(idx == 0)):
        st.session_state.current_index -= 1
        st.rerun()

with nav_col2:
    if st.button("💾 Guardar y Siguiente ➡️", type="primary", use_container_width=True):
        # Guardar la anotación actual en el DataFrame del estado de sesión
        st.session_state.annotations_df.loc[idx, 'category'] = category
        st.session_state.annotations_df.loc[idx, 'score'] = CATEGORY_TO_SCORE[category]
        st.session_state.annotations_df.loc[idx, 'justification'] = justification
        st.session_state.annotations_df.loc[idx, 'annotator_id'] = annotator_id
        
        # Lógica para saltar al siguiente par *no anotado* para agilizar el trabajo
        annotated_mask = (st.session_state.annotations_df['category'] != '') & (st.session_state.annotations_df['category'].notna())
        if annotated_mask.all():
            if st.session_state.current_index < total_pairs - 1:
                st.session_state.current_index += 1
        else:
            st.session_state.current_index = annotated_mask.idxmin()
        st.rerun()

with nav_col3:
    if st.button("Saltar ➡️", use_container_width=True):
        st.session_state.current_index += 1
        st.rerun()