import streamlit as st
import pandas as pd
from pathlib import Path
import sys
import time
import yaml
import pyperclip
import os

# --- CONFIGURACIÓN Y CONSTANTES DEL PROYECTO ---

# Se establece la ruta raíz del proyecto para poder importar módulos propios.
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# Se importan constantes y utilidades desde el código fuente del proyecto.
from src.constants import CATEGORY_TO_SCORE, ORDERED_CATEGORIES
from src.text_utils import parse_string_list

# Configuración inicial de la página de Streamlit para que ocupe todo el ancho.
st.set_page_config(layout="wide", page_title="DistilMatch - Golden Set Builder")

# Definición de un color base para los encabezados para mantener la consistencia visual.
HEADER_COLOR = "#fd7e14"

# Rutas a los archivos de datos ya procesados.
PROCESSED_DATA_DIR = project_root / 'data' / '01_processed'
OFFERS_FILE = PROCESSED_DATA_DIR / 'offers_processed.csv'
CVS_FILE = PROCESSED_DATA_DIR / 'cvs_processed.csv'

# Rutas y configuración para la funcionalidad de generación de prompts.
CONFIG_PROMPTS_FILE = project_root / 'config' / 'experiment_prompts.yaml'
TARGET_PROMPT_ID = 'P-06_assisted_curation' # ID del prompt a utilizar.

# --- FUNCIONES DE AYUDA (HELPERS) ---

@st.cache_data
def load_prompt_template(prompt_id: str, config_path: Path) -> str:
    """
    Carga de forma eficiente la plantilla de un prompt desde los archivos del proyecto.
    Utiliza el archivo de configuración YAML para encontrar la ruta del prompt deseado.
    La función está cacheada para evitar leer los archivos repetidamente.
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            prompts_config = yaml.safe_load(f)
        
        prompt_info = prompts_config.get(prompt_id)
        if not prompt_info:
            st.error(f"Error: No se encontró el prompt con ID '{prompt_id}' en {config_path}")
            return None

        prompt_path = project_root / prompt_info['path']
        if not os.path.exists(prompt_path):
            st.error(f"Error: El archivo del prompt '{prompt_path}' no existe.")
            return None

        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read()

    except Exception as e:
        st.error(f"Ocurrió un error al cargar el prompt: {e}")
        return None

def get_record_as_text(record: pd.Series) -> str:
    """
    Convierte una fila de un DataFrame (un CV o una oferta) en un bloque de texto
    formateado, ideal para ser insertado en un prompt para un LLM.
    """
    text_parts = []
    for col, val in record.items():
        if pd.isna(val) or str(val).strip() == '' or col in ['job_id', 'candidate_id']:
            continue
        
        col_name = col.replace('_', ' ').title()
        text_parts.append(f"--- {col_name} ---\n{val}\n")
    
    return "\n".join(text_parts)

@st.cache_data
def load_source_datasets():
    """
    Carga los DataFrames completos de ofertas y CVs.
    La directiva @st.cache_data asegura que esta operación costosa solo se ejecute una vez.
    """
    try:
        offers_df = pd.read_csv(OFFERS_FILE, dtype={'job_id': str})
        cvs_df = pd.read_csv(CVS_FILE, dtype={'candidate_id': str})
        return offers_df, cvs_df
    except FileNotFoundError as e:
        st.error(f"Error crítico al cargar datos: {e.filename}. Asegúrate de haber ejecutado el pipeline de pre-procesamiento.")
        return None, None

def display_record_details(title, record, color):
    """
    Renderiza de forma visualmente atractiva los detalles de un registro (CV u oferta).
    Aplica formatos especiales para diferentes tipos de campos (años, listas, texto largo).
    """
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

# --- INICIALIZACIÓN DEL ESTADO DE LA SESIÓN ---

# Este bloque se ejecuta solo una vez al iniciar la aplicación.
# Se encarga de cargar los datos y establecer los valores iniciales.
if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    all_offers, all_cvs = load_source_datasets()
    if all_offers is not None:
        st.session_state.all_offers = all_offers
        st.session_state.all_cvs = all_cvs
        # Muestra un par aleatorio al empezar.
        st.session_state.current_offer = all_offers.sample(1).iloc[0]
        st.session_state.current_cv = all_cvs.sample(1).iloc[0]
    # Crea el DataFrame vacío para guardar las anotaciones.
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
    # Campo de búsqueda para filtrar ofertas por título.
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

    # Botones para cambiar el par actual (completo, solo oferta, o solo CV).
    if st.button("🔄 Nuevo Par Aleatorio", use_container_width=True, type="primary"):
        st.session_state.current_offer = filtered_offers_df.sample(1).iloc[0]
        st.session_state.current_cv = st.session_state.all_cvs.sample(1).iloc[0]
        st.rerun()
    c1, c2 = st.columns(2)
    c1.button("📄 Nueva Oferta", use_container_width=True, on_click=lambda: st.session_state.update(current_offer=filtered_offers_df.sample(1).iloc[0]))
    c2.button("👤 Nuevo CV", use_container_width=True, on_click=lambda: st.session_state.update(current_cv=st.session_state.all_cvs.sample(1).iloc[0]))
    
    st.divider()
    
    st.header("📋 Golden Set Actual")
    # Funcionalidad para cargar un archivo CSV y continuar una sesión de anotación.
    uploaded_file = st.file_uploader("Continuar Anotando (Cargar CSV):", type="csv")
    if uploaded_file and not st.session_state.file_loaded:
        st.session_state.golden_pairs_df = pd.read_csv(uploaded_file)
        st.session_state.file_loaded = True
        st.success(f"Cargados {len(st.session_state.golden_pairs_df)} pares.")
        st.rerun()
    
    # Muestra estadísticas y una vista previa del Golden Set que se está creando.
    st.metric("Pares Creados", len(st.session_state.golden_pairs_df))
    if not st.session_state.golden_pairs_df.empty:
        st.dataframe(st.session_state.golden_pairs_df[['candidate_id', 'job_id', 'category']].tail(), use_container_width=True)
    
    # Botón para descargar el trabajo realizado.
    st.download_button("📥 Descargar Golden Set (CSV)", st.session_state.golden_pairs_df.to_csv(index=False).encode('utf-8'), "gold_standard_full.csv", "text/csv")


# --- INTERFAZ PRINCIPAL ---

st.title("DistilMatch - Golden Set Builder 🧑‍🔬")
st.markdown("Usa la barra lateral para **buscar**, **explorar** y **gestionar tu sesión** de anotación.")

# Detiene la ejecución si los datos no se pudieron cargar al inicio.
if 'all_offers' not in st.session_state:
    st.error("Los datasets no se han cargado. La aplicación no puede continuar.")
    st.stop()

# Muestra el par CV-Oferta actual en dos columnas.
col_offer, col_cv = st.columns(2, gap="large")
with col_offer:
    display_record_details("Oferta de Trabajo", st.session_state.current_offer, color=HEADER_COLOR)
with col_cv:
    display_record_details("Currículum Vitae", st.session_state.current_cv, color=HEADER_COLOR)

st.divider()

# --- SECCIÓN DE GENERACIÓN DE PROMPTS ---

st.subheader("🧪 Validación Externa con LLM")
st.markdown("Copia el prompt completo para este par y pégalo en una interfaz de LLM (ChatGPT, Claude, etc.) para obtener una segunda opinión.")

# Carga la plantilla del prompt definida en la sección de constantes.
prompt_template = load_prompt_template(TARGET_PROMPT_ID, CONFIG_PROMPTS_FILE)

if prompt_template:
    # Botón principal para la funcionalidad de copiado.
    if st.button(f"📋 Copiar Prompt ({TARGET_PROMPT_ID})", use_container_width=True):
        # 1. Convierte los datos del CV y la oferta a texto plano.
        offer_text = get_record_as_text(st.session_state.current_offer)
        cv_text = get_record_as_text(st.session_state.current_cv)
        
        # 2. Inserta los textos en la plantilla del prompt.
        final_prompt = prompt_template.format(text_of_job_posting=offer_text, text_of_cv=cv_text)
        
        # 3. Utiliza la librería pyperclip para copiar el resultado al portapapeles.
        try:
            pyperclip.copy(final_prompt)
            st.success("¡Prompt copiado al portapapeles!")
        except Exception as e:
            st.error(f"Error al copiar: {e}. (En Linux, puede que necesites 'sudo apt-get install xclip')")
            # Como plan B, muestra el prompt en un área de texto para copiado manual.
            st.text_area("Copia manual:", final_prompt, height=200)

    # Permite al usuario ver la plantilla que se está utilizando.
    with st.expander("Ver plantilla del prompt actual"):
        st.text(prompt_template)
else:
    st.warning("No se pudo cargar la plantilla del prompt. La función de copiado está deshabilitada.")

st.divider()

# --- FORMULARIO DE ANOTACIÓN ---

# `st.form` agrupa los elementos y asegura que solo se envíen al pulsar el botón.
with st.form("annotation_form"):
    st.subheader("Tu Evaluación para este Par")
    
    with st.expander("📖 Ver Guía Rápida de Anotación"):
        st.markdown("""
        - **🟢 MUST INTERVIEW**: Prioridad máxima. El candidato cumple o excede los requisitos clave. ¡Contactar ya!
        - **🟡 PROMISING FIT**: Fuerte potencial. Hay suficientes señales positivas para justificar una llamada de screening.
        - **🟠 BORDERLINE**: Un "no" probable, pero con alguna cualidad redentora. Guardar "por si acaso".
        - **🔴 NO FIT**: Descarte claro. Falta algún requisito no negociable.
        """)

    # Controles para que el anotador ingrese su evaluación.
    category = st.radio(
        "Selecciona la categoría orientada a la acción:", 
        options=ORDERED_CATEGORIES, 
        index=1,  # Por defecto, se selecciona 'PROMISING FIT'.
        horizontal=True
    )
    justification = st.text_area("Justificación (opcional):", height=100)
    
    # Botón de envío del formulario.
    submitted = st.form_submit_button("➕ Añadir Par al Golden Set", type="primary", use_container_width=True)
    
    if submitted:
        offer_id = st.session_state.current_offer['job_id']
        cv_id = st.session_state.current_cv['candidate_id']
        
        # Comprueba si el par ya ha sido anotado para evitar duplicados.
        is_duplicate = not st.session_state.golden_pairs_df[
            (st.session_state.golden_pairs_df['job_id'] == offer_id) & 
            (st.session_state.golden_pairs_df['candidate_id'] == cv_id)
        ].empty
        
        if is_duplicate:
            st.warning("⚠️ Este par ya ha sido añadido.")
        else:
            # Si no es un duplicado, crea una nueva fila y la añade al DataFrame.
            new_row = pd.DataFrame([{
                'job_id': offer_id, 
                'candidate_id': cv_id, 
                'category': category, 
                'score': CATEGORY_TO_SCORE[category], 
                'justification': justification, 
                'annotator_id': annotator_id
            }])
            st.session_state.golden_pairs_df = pd.concat([st.session_state.golden_pairs_df, new_row], ignore_index=True)
            
            # Proporciona feedback visual y sonoro al usuario.
            st.success(f"✅ ¡Par añadido! Total: {len(st.session_state.golden_pairs_df)}. Se cargará un nuevo par...")
            st.balloons()
            
            # Carga un nuevo par aleatorio para agilizar el flujo de trabajo.
            if search_query and not filtered_offers_df.empty:
                 st.session_state.current_offer = filtered_offers_df.sample(1).iloc[0]
            else:
                 st.session_state.current_offer = st.session_state.all_offers.sample(1).iloc[0]
            st.session_state.current_cv = st.session_state.all_cvs.sample(1).iloc[0]
            
            # Espera un segundo y recarga la página para mostrar el nuevo par.
            time.sleep(1)
            st.rerun()
