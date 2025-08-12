"""
Herramienta de Exploración y Búsqueda para la Creación de un "Golden Set".

Esta aplicación de Streamlit combina la exploración aleatoria con la búsqueda
dirigida. El usuario puede filtrar ofertas por palabras clave en el título y
luego explorar aleatoriamente dentro de ese subconjunto, permitiendo la
creación eficiente de un Golden Set de alta calidad.
"""

import streamlit as st
import pandas as pd
from pathlib import Path
import sys
import time

# --- CONFIGURACIÓN Y CONSTANTES ---
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.constants import CATEGORY_TO_SCORE, ORDERED_CATEGORIES
from src.text_utils import parse_string_list

st.set_page_config(layout="wide", page_title="DistilMatch - Golden Set Explorer")

HEADER_COLOR = "#fd7e14" # Nuevo color para destacar

# Rutas a los datasets completos
PROCESSED_DATA_DIR = project_root / 'data' / '01_processed'
OFFERS_FILE = PROCESSED_DATA_DIR / 'offers_processed.csv'
CVS_FILE = PROCESSED_DATA_DIR / 'cvs_processed.csv'

# --- FUNCIONES DE AYUDA ---
@st.cache_data
def load_source_datasets():
    """Carga los DataFrames completos de ofertas y CVs."""
    try:
        offers_df = pd.read_csv(OFFERS_FILE, dtype={'job_id': str})
        cvs_df = pd.read_csv(CVS_FILE, dtype={'candidate_id': str})
        return offers_df, cvs_df
    except FileNotFoundError as e:
        st.error(f"Error al cargar datos: {e.filename}.")
        return None, None

def display_record_details(title, record, color):
    """Muestra todos los campos de un registro de forma estructurada."""
    st.markdown(f'## <span style="color: {color};">{title}</span>', unsafe_allow_html=True)
    for col, val in record.items():
        if col in ['display_label']: continue
        
        col_name = col.replace('_', ' ').title()
        parsed_list = parse_string_list(val)

        if isinstance(parsed_list, list) and len(parsed_list) > 0:
            st.markdown(f"**{col_name}:**")
            st.markdown("\n".join([f"- `{item}`" for item in parsed_list]))
        elif pd.notna(val) and val != '':
            st.markdown(f"**{col_name}:**")
            st.markdown(f"> {val}")

# --- INICIALIZACIÓN DEL ESTADO ---
if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    all_offers, all_cvs = load_source_datasets()
    if all_offers is not None:
        st.session_state.all_offers = all_offers
        st.session_state.all_cvs = all_cvs
        st.session_state.current_offer = all_offers.sample(1).iloc[0]
        st.session_state.current_cv = all_cvs.sample(1).iloc[0]
    st.session_state.golden_pairs_df = pd.DataFrame(columns=[
        'job_id', 'candidate_id', 'category', 'score', 'justification', 'annotator_id'
    ])

# --- INTERFAZ PRINCIPAL ---
st.title("DistilMatch - Explorador y Buscador 🔍🎲")
st.markdown("Usa la barra lateral para buscar ofertas por título o para explorar aleatoriamente.")

if 'all_offers' not in st.session_state:
    st.error("Los datasets no pudieron ser cargados. La aplicación no puede continuar.")
    st.stop()

# --- BARRA LATERAL (SIDEBAR) PARA CONTROLES ---
with st.sidebar:
    st.header("🕹️ Panel de Control")
    annotator_id = st.text_input("Tu ID de Anotador:", key="annotator_id_input")
    
    st.divider()
    st.header("🔬 Controles de Exploración")

    # NUEVO: Barra de Búsqueda
    search_query = st.text_input("Buscar en Título de Oferta:", placeholder="Ej: Data Scientist, Audit...")

    # Lógica de filtrado
    if search_query:
        mask = st.session_state.all_offers['title'].str.contains(search_query, case=False, na=False)
        filtered_offers_df = st.session_state.all_offers[mask]
        st.info(f"Mostrando ofertas aleatorias de **{len(filtered_offers_df)}** resultados encontrados.")
    else:
        filtered_offers_df = st.session_state.all_offers

    if search_query and filtered_offers_df.empty:
        st.warning("No se encontraron ofertas. Los botones usarán el dataset completo.")
        filtered_offers_df = st.session_state.all_offers # Fallback al dataset completo

    # Botones de exploración (ahora usan el DataFrame filtrado)
    if st.button("🔄 Nuevo Par Aleatorio", use_container_width=True, type="primary"):
        st.session_state.current_offer = filtered_offers_df.sample(1).iloc[0]
        st.session_state.current_cv = st.session_state.all_cvs.sample(1).iloc[0]
        st.rerun()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("📄 Nueva Oferta", use_container_width=True):
            st.session_state.current_offer = filtered_offers_df.sample(1).iloc[0]
            st.rerun()
    with col2:
        if st.button("👤 Nuevo CV", use_container_width=True):
            st.session_state.current_cv = st.session_state.all_cvs.sample(1).iloc[0]
            st.rerun()
            
    st.divider()
    
    st.header("📋 Golden Set Creado")
    st.metric("Pares Creados", len(st.session_state.golden_pairs_df))
    if not st.session_state.golden_pairs_df.empty:
        st.dataframe(st.session_state.golden_pairs_df[['job_id', 'category']], use_container_width=True)

    st.download_button(
        label="📥 Descargar Golden Set (CSV)",
        data=st.session_state.golden_pairs_df.to_csv(index=False).encode('utf-8'),
        file_name="explored_golden_set.csv",
        mime='text/csv'
    )

if not annotator_id:
    st.warning("Por favor, introduce tu ID de Anotador en la barra lateral para continuar.")
    st.stop()

# --- VISTA PRINCIPAL ---
offer_details = st.session_state.current_offer
cv_details = st.session_state.current_cv
col_offer, col_cv = st.columns(2, gap="large")

with col_offer:
    display_record_details("Oferta de Trabajo", offer_details, color=HEADER_COLOR)
with col_cv:
    display_record_details("Currículum Vitae", cv_details, color=HEADER_COLOR)

st.divider()

# --- Formulario de Anotación ---
st.subheader("Tu Evaluación para este Par")
with st.form("annotation_form"):
    category = st.radio("Selecciona la categoría:", options=ORDERED_CATEGORIES, index=2, horizontal=True)
    justification = st.text_area("Justificación (opcional):")
    
    submitted = st.form_submit_button("➕ Añadir Par al Golden Set", type="primary", use_container_width=True)
    
    if submitted:
        offer_id = offer_details['job_id']
        cv_id = cv_details['candidate_id']
        
        is_duplicate = not st.session_state.golden_pairs_df[
            (st.session_state.golden_pairs_df['job_id'] == offer_id) &
            (st.session_state.golden_pairs_df['candidate_id'] == cv_id)
        ].empty
        
        if is_duplicate:
            st.warning("⚠️ Este par ya ha sido añadido.")
        else:
            new_row = pd.DataFrame([{'job_id': offer_id, 'candidate_id': cv_id, 'category': category, 'score': CATEGORY_TO_SCORE[category], 'justification': justification, 'annotator_id': annotator_id}])
            st.session_state.golden_pairs_df = pd.concat([st.session_state.golden_pairs_df, new_row], ignore_index=True)
            st.success(f"✅ ¡Par añadido! Total: {len(st.session_state.golden_pairs_df)}. Se cargará un nuevo par aleatorio...")
            st.balloons()
            
            # Cargar un nuevo par aleatorio, respetando el filtro de búsqueda si existe
            st.session_state.current_offer = filtered_offers_df.sample(1).iloc[0]
            st.session_state.current_cv = st.session_state.all_cvs.sample(1).iloc[0]
            
            time.sleep(1.5)
            st.rerun()