import streamlit as st
import pandas as pd
from pathlib import Path
import sys
import time

# --- CONFIGURACIÓN Y CONSTANTES ---
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.constants import CATEGORY_TO_SCORE, ORDERED_CATEGORIES
from src.text_utils import parse_string_list

st.set_page_config(layout="wide", page_title="DistilMatch - Golden Set Builder")

HEADER_COLOR = "#fd7e14"

# Rutas a los datasets procesados completos
PROCESSED_DATA_DIR = project_root / 'data' / '01_processed'
OFFERS_FILE = PROCESSED_DATA_DIR / 'offers_processed.csv'
CVS_FILE = PROCESSED_DATA_DIR / 'cvs_processed.csv'

# --- FUNCIONES DE AYUDA ---
@st.cache_data
def load_source_datasets():
    """Carga los DataFrames completos de ofertas y CVs desde los archivos procesados."""
    try:
        offers_df = pd.read_csv(OFFERS_FILE, dtype={'job_id': str})
        cvs_df = pd.read_csv(CVS_FILE, dtype={'candidate_id': str})
        return offers_df, cvs_df
    except FileNotFoundError as e:
        st.error(f"Error crítico al cargar datos: {e.filename}. Asegúrate de haber ejecutado el script de pre-procesamiento.")
        return None, None

def display_record_details(title, record, color):
    """Muestra los detalles de un registro con formato inteligente y adaptable a temas."""
    st.markdown(f'## <span style="color: {color};">{title}</span>', unsafe_allow_html=True)
    for col, val in record.items():
        if pd.isna(val) or str(val).strip() == '':
            continue

        col_name = col.replace('_', ' ').title()

        if col == 'total_experience_years':
            st.markdown(f"**{col_name}:** `{val}` años")
        elif col in ['formatted_work_history', 'description', 'career_objective']:
            st.markdown(f"**{col_name}:**")
            style = f"border-left: 3px solid {color}; padding: 10px; border-radius: 5px; white-space: pre-wrap; font-size: 0.9em;"
            st.markdown(f"<div style='{style}'>{val}</div>", unsafe_allow_html=True)
        else:
            parsed_list = parse_string_list(val)
            if isinstance(parsed_list, list) and len(parsed_list) > 0:
                st.markdown(f"**{col_name}:**")
                st.markdown("\n".join([f"- `{item}`" for item in parsed_list]))
            else:
                if col not in ['job_id', 'candidate_id']:
                    st.markdown(f"**{col_name}:** {val}")

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
    st.session_state.file_loaded = False

# --- BARRA LATERAL (SIDEBAR) ---
with st.sidebar:
    st.header("🕹️ Panel de Control")
    annotator_id = st.text_input("Tu ID de Anotador:", value="expert_01", key="annotator_id_input")
    st.divider()
    st.header("🔬 Controles de Exploración")
    search_query = st.text_input("Buscar en Título de Oferta:", placeholder="Ej: Data Scientist...")
    if search_query:
        mask = st.session_state.all_offers['title'].str.contains(search_query, case=False, na=False)
        filtered_offers_df = st.session_state.all_offers[mask]
        st.info(f"Explorando **{len(filtered_offers_df)}** ofertas.")
    else:
        filtered_offers_df = st.session_state.all_offers
    if search_query and filtered_offers_df.empty:
        st.warning("No se encontraron ofertas. Usando dataset completo.")
        filtered_offers_df = st.session_state.all_offers
    if st.button("🔄 Nuevo Par Aleatorio", use_container_width=True, type="primary"):
        st.session_state.current_offer = filtered_offers_df.sample(1).iloc[0]
        st.session_state.current_cv = st.session_state.all_cvs.sample(1).iloc[0]
        st.rerun()
    c1, c2 = st.columns(2)
    c1.button("📄 Nueva Oferta", use_container_width=True, on_click=lambda: st.session_state.update(current_offer=filtered_offers_df.sample(1).iloc[0]))
    c2.button("👤 Nuevo CV", use_container_width=True, on_click=lambda: st.session_state.update(current_cv=st.session_state.all_cvs.sample(1).iloc[0]))
    st.divider()
    st.header("📋 Golden Set Actual")
    uploaded_file = st.file_uploader("Continuar Anotando (Cargar CSV):", type="csv")
    if uploaded_file and not st.session_state.file_loaded:
        st.session_state.golden_pairs_df = pd.read_csv(uploaded_file)
        st.session_state.file_loaded = True
        st.success(f"Cargados {len(st.session_state.golden_pairs_df)} pares.")
        st.rerun()
    st.metric("Pares Creados", len(st.session_state.golden_pairs_df))
    if not st.session_state.golden_pairs_df.empty:
        st.dataframe(st.session_state.golden_pairs_df[['candidate_id', 'job_id', 'category']].tail(), use_container_width=True)
    st.download_button("📥 Descargar Golden Set (CSV)", st.session_state.golden_pairs_df.to_csv(index=False).encode('utf-8'), "golden_standard_full.csv", "text/csv")

# --- INTERFAZ PRINCIPAL ---
st.title("DistilMatch - Golden Set Builder 🧑‍🔬")
st.markdown("Usa la barra lateral para **buscar**, **explorar** y **gestionar tu sesión** de anotación.")
if 'all_offers' not in st.session_state:
    st.error("Los datasets no se han cargado. La aplicación no puede continuar.")
    st.stop()
col_offer, col_cv = st.columns(2, gap="large")
with col_offer:
    display_record_details("Oferta de Trabajo", st.session_state.current_offer, color=HEADER_COLOR)
with col_cv:
    display_record_details("Currículum Vitae", st.session_state.current_cv, color=HEADER_COLOR)
st.divider()

# --- Formulario de Anotación ---
with st.form("annotation_form"):
    st.subheader("Tu Evaluación para este Par")
    

    with st.expander("📖 Ver Guía Rápida de Anotación"):
        st.markdown("""
        - **🟢 MUST INTERVIEW**: Prioridad máxima. El candidato cumple o excede los requisitos clave. ¡Contactar ya!
        - **🟡 PROMISING FIT**: Fuerte potencial. Hay suficientes señales positivas para justificar una llamada de screening.
        - **🟠 BORDERLINE**: Un "no" probable, pero con alguna cualidad redentora. Guardar "por si acaso".
        - **⚫ NO FIT**: Descarte claro. Falta algún requisito no negociable.
        """)
        

    category = st.radio(
        "Selecciona la categoría orientada a la acción:", 
        options=ORDERED_CATEGORIES, 
        index=1,  # Default a 'PROMISING FIT'
        horizontal=True
    )
    justification = st.text_area("Justificación (opcional):", height=100)
    
    submitted = st.form_submit_button("➕ Añadir Par al Golden Set", type="primary", use_container_width=True)
    
    if submitted:
        offer_id = st.session_state.current_offer['job_id']
        cv_id = st.session_state.current_cv['candidate_id']
        is_duplicate = not st.session_state.golden_pairs_df[(st.session_state.golden_pairs_df['job_id'] == offer_id) & (st.session_state.golden_pairs_df['candidate_id'] == cv_id)].empty
        
        if is_duplicate:
            st.warning("⚠️ Este par ya ha sido añadido.")
        else:
            new_row = pd.DataFrame([{'job_id': offer_id, 'candidate_id': cv_id, 'category': category, 'score': CATEGORY_TO_SCORE[category], 'justification': justification, 'annotator_id': annotator_id}])
            st.session_state.golden_pairs_df = pd.concat([st.session_state.golden_pairs_df, new_row], ignore_index=True)
            st.success(f"✅ ¡Par añadido! Total: {len(st.session_state.golden_pairs_df)}. Se cargará un nuevo par...")
            st.balloons()
            st.session_state.current_offer = filtered_offers_df.sample(1).iloc[0]
            st.session_state.current_cv = st.session_state.all_cvs.sample(1).iloc[0]
            time.sleep(1)
            st.rerun()