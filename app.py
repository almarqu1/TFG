# TFG_DistilMatch/app.py (Versión 4.1 - Corrección de KeyError)

import streamlit as st
import pandas as pd
from pathlib import Path
import yaml
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import sys
import math
import time

# --- CONFIGURACIÓN Y CONSTANTES DEL PROYECTO ---

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from src.utils import (
    load_config as load_project_config,
    load_prompt_template,
    parse_json_from_response,
    format_entity_text,
    get_cv_sections,
    get_offer_sections
)

# --- INICIALIZACIÓN Y GESTIÓN DE ESTADO ---

if 'selected_offer' not in st.session_state:
    st.session_state.selected_offer = None
if 'selected_candidates' not in st.session_state:
    st.session_state.selected_candidates = []

if 'offer_page' not in st.session_state:
    st.session_state.offer_page = 0
if 'candidate_page' not in st.session_state:
    st.session_state.candidate_page = 0
if 'last_offer_search' not in st.session_state:
    st.session_state.last_offer_search = ""
if 'last_cv_search' not in st.session_state:
    st.session_state.last_cv_search = ""

# --- FUNCIONES DE LÓGICA DE LA APP ---

def select_offer(offer_series: pd.Series):
    st.session_state.selected_offer = offer_series

def clear_offer():
    st.session_state.selected_offer = None

def add_candidate(candidate_id: str):
    if candidate_id not in st.session_state.selected_candidates:
        st.session_state.selected_candidates.append(candidate_id)

def remove_candidate(candidate_id: str):
    if candidate_id in st.session_state.selected_candidates:
        st.session_state.selected_candidates.remove(candidate_id)

def change_page(page_key, delta):
    st.session_state[page_key] += delta

# --- LÓGICA DEL MODELO ---
@st.cache_resource
def load_config():
    return load_project_config()

CONFIG = load_config()

@st.cache_data
def load_source_datasets():
    try:
        offers_path = PROJECT_ROOT / CONFIG['data_paths']['processed']['offers']
        cvs_path = PROJECT_ROOT / CONFIG['data_paths']['processed']['cvs']
        offers_df = pd.read_csv(offers_path, dtype={'job_id': str})
        cvs_df = pd.read_csv(cvs_path, dtype={'candidate_id': str})
        return offers_df, cvs_df
    except FileNotFoundError as e:
        st.error(f"Error crítico al cargar datos: {e.filename}.")
        return None, None

def format_prompt_messages(offer: pd.Series, cv: pd.Series) -> list[dict]:
    offer_text = format_entity_text(offer, get_offer_sections())
    cv_text = format_entity_text(cv, get_cv_sections())
    system_prompt = """You are an expert Talent Acquisition Specialist. Your task is to analyze a job posting and a candidate's CV and provide a structured JSON assessment.
Your entire response must be a single JSON object with the following keys:
- "strengths": A string detailing what makes the candidate a good fit.
- "concerns_and_gaps": A string detailing what they are missing or potential issues.
- "verdict": A string with one of the following exact values: "MUST INTERVIEW", "PROMISING FIT", "BORDERLINE", "NO FIT".
- "score": An integer from 0 to 100 representing the compatibility."""
    user_prompt = f"### JOB POSTING ###\n{offer_text}\n\n### CANDIDATE CV ###\n{cv_text}"
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

@st.cache_resource(show_spinner="Cargando modelo DistilMatch...")
def load_model_for_inference(experiment_id: str):
    base_model_name = CONFIG['student_model']['base_model_name']
    lora_adapters_path = PROJECT_ROOT / CONFIG['experiments'][experiment_id]['output_dir'] / "final_checkpoint"
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(base_model_name, quantization_config=bnb_config, device_map="auto", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = PeftModel.from_pretrained(model, str(lora_adapters_path))
    model = model.merge_and_unload()
    model.eval()
    st.success(f"Modelo '{experiment_id}' cargado y listo.")
    return model, tokenizer

def run_inference(model, tokenizer, offer: pd.Series, cv: pd.Series) -> dict:
    messages = format_prompt_messages(offer, cv)
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + "```json\n"
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=1024, do_sample=False, pad_token_id=tokenizer.eos_token_id, temperature=0.1)
    generated_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    assistant_response = "```json\n" + generated_text
    parsed_output = parse_json_from_response(assistant_response)
    if parsed_output is None:
        return {"error": "Failed to parse JSON from model output.", "raw_output": assistant_response}
    return parsed_output

# --- Componentes de la UI ---
def render_paginated_view(results_df: pd.DataFrame, page_state_key: str, render_item_func: callable, items_per_page: int = 5):
    if results_df.empty: return
    total_items, total_pages = len(results_df), math.ceil(len(results_df) / items_per_page)
    current_page = st.session_state[page_state_key]
    start_idx, end_idx = current_page * items_per_page, min((current_page + 1) * items_per_page, total_items)
    for _, row in results_df.iloc[start_idx:end_idx].iterrows(): render_item_func(row)
    st.divider()
    col1, col2, col3 = st.columns([1, 2, 1])
    col1.button(f"⬅️ Anterior", key=f"prev_{page_state_key}", on_click=change_page, args=(page_state_key, -1), disabled=(current_page == 0))
    col2.write(f"Página {current_page + 1} de {total_pages}")
    col3.button(f"Siguiente ➡️", key=f"next_{page_state_key}", on_click=change_page, args=(page_state_key, 1), disabled=(current_page >= total_pages - 1))

# --- INTERFAZ DE STREAMLIT ---
st.set_page_config(layout="wide", page_title="DistilMatch - Demo")
st.title("🧪 DistilMatch: Demo de Inferencia")

offers_df, cvs_df = load_source_datasets()
if offers_df is None or cvs_df is None: st.stop()

with st.sidebar:
    st.header("⚙️ Panel de Análisis")
    st.subheader("1. Elige el Modelo")
    experiment_options = {k: v['description'] for k, v in CONFIG['experiments'].items()}
    selected_experiment = st.selectbox("Modelo a utilizar:", options=list(experiment_options.keys()), format_func=lambda x: f"{x}", index=3)
    st.divider()
    st.subheader("2. Tu Selección Actual")
    if st.session_state.selected_offer is not None:
        offer = st.session_state.selected_offer
        st.success(f"**Oferta:** {offer['title']}")
        st.button("Quitar Oferta", on_click=clear_offer, use_container_width=True)
    else:
        st.info("Aún no has seleccionado una oferta.")
    st.metric("Candidatos en la lista", len(st.session_state.selected_candidates))
    if st.session_state.selected_candidates:
        for cand_id in st.session_state.selected_candidates:
            col1, col2 = st.columns([3, 1])
            col1.write(f"- `{cand_id}`")
            col2.button("X", key=f"remove_{cand_id}", on_click=remove_candidate, args=(cand_id,))
    else:
        st.info("Aún no has añadido candidatos.")
    st.divider()
    st.subheader("3. Ejecutar Análisis")
    is_ready_to_run = st.session_state.selected_offer is not None and len(st.session_state.selected_candidates) > 0
    run_button = st.button("🚀 Analizar Candidatos", type="primary", use_container_width=True, disabled=not is_ready_to_run)

if not run_button:
    st.markdown("Usa los exploradores para buscar y añadir una oferta y candidatos a tu panel de análisis en la barra lateral.")
    offer_col, cv_col = st.columns(2, gap="large")
    with offer_col:
        st.header("🔎 Explorador de Ofertas")
        search_offer = st.text_input("Buscar ofertas por título...", key="search_offer_input")
        if search_offer != st.session_state.last_offer_search:
            st.session_state.offer_page = 0
            st.session_state.last_offer_search = search_offer
        if search_offer:
            mask = offers_df['title'].str.contains(search_offer, case=False, na=False)
            results = offers_df[mask].reset_index(drop=True)
            st.write(f"**{len(results)}** ofertas encontradas.")
            def render_offer_item(row):
                with st.expander(f"{row['title']}"):
                    st.markdown(f"**ID:** `{row['job_id']}`"); st.markdown(f"**Experiencia:** {row.get('formatted_experience_level', 'N/A')}")
                    st.button("Seleccionar esta Oferta", key=f"select_{row['job_id']}", on_click=select_offer, args=(row,))
            render_paginated_view(results, 'offer_page', render_offer_item)
    with cv_col:
        st.header("👤 Explorador de Candidatos")
        search_cv = st.text_input("Buscar candidatos por skills...", key="search_cv_input")
        if search_cv != st.session_state.last_cv_search:
            st.session_state.candidate_page = 0
            st.session_state.last_cv_search = search_cv
        if search_cv:
            mask = cvs_df['skills'].str.contains(search_cv, case=False, na=False)
            results = cvs_df[mask].reset_index(drop=True)
            st.write(f"**{len(results)}** candidatos encontrados.")
            def render_cv_item(row):
                with st.expander(f"Candidato `{row['candidate_id']}`"):
                    st.markdown(f"**Objetivo Profesional:**"); st.info(row['career_objective'])
                    st.markdown(f"**Skills:**"); st.warning(row['skills'])
                    st.button("Añadir a la Lista", key=f"add_{row['candidate_id']}", on_click=add_candidate, args=(row['candidate_id'],))
            render_paginated_view(results, 'candidate_page', render_cv_item)
else:
    # --- VISTA DE RESULTADOS (LÓGICA CORREGIDA) ---
    model, tokenizer = load_model_for_inference(selected_experiment)
    selected_offer_data = st.session_state.selected_offer
    selected_cvs_data = cvs_df[cvs_df['candidate_id'].isin(st.session_state.selected_candidates)]
    st.header(f"Resultados del Análisis (Modelo: {selected_experiment})")
    st.subheader(f"Comparando contra la oferta: {selected_offer_data['title']}")

    results_list = []
    num_candidates = len(selected_cvs_data)
    progress_bar = st.progress(0, text="Iniciando análisis...")
    for i, (_, cv_row) in enumerate(selected_cvs_data.iterrows()):
        cv_id = cv_row['candidate_id']
        progress_bar.progress((i + 1) / num_candidates, text=f"Analizando candidato {cv_id} ({i+1}/{num_candidates})...")
        result = run_inference(model, tokenizer, selected_offer_data, cv_row)
        results_list.append({'candidate_id': cv_id, **result})
        time.sleep(0.1)
    progress_bar.empty()

    results_df = pd.DataFrame(results_list)

    # ===== BLOQUE DE CÓDIGO CORREGIDO =====
    failed_analyses = pd.DataFrame()
    successful_analyses = pd.DataFrame()

    if not results_df.empty:
        if 'error' in results_df.columns:
            # Caso 1: Al menos un análisis falló
            successful_analyses = results_df[results_df['error'].isna()].copy()
            failed_analyses = results_df[results_df['error'].notna()]
        else:
            # Caso 2: Todos los análisis fueron exitosos
            successful_analyses = results_df.copy()
    # ===== FIN DEL BLOQUE CORREGIDO =====

    if not successful_analyses.empty:
        successful_analyses['score'] = pd.to_numeric(successful_analyses['score'], errors='coerce').fillna(0)
        ranked_df = successful_analyses.sort_values(by='score', ascending=False).reset_index(drop=True)

        st.subheader("🏆 Ranking de Candidatos")
        st.dataframe(ranked_df[['candidate_id', 'score', 'verdict']], use_container_width=True, hide_index=True)

        st.subheader("📋 Análisis Detallado (Ordenado por Ranking)")
        for _, row in ranked_df.iterrows():
            with st.expander(f"**Candidato:** `{row['candidate_id']}` | **Puntuación:** {int(row['score'])} | **Veredicto:** {row['verdict']}"):
                st.markdown("✅ **Fortalezas Clave:**")
                st.success(row.get('strengths', 'N/A'))
                st.markdown("⚠️ **Posibles Inquietudes y Brechas:**")
                st.warning(row.get('concerns_and_gaps', 'N/A'))
    
    if not failed_analyses.empty:
        st.subheader("❌ Errores en el Análisis")
        for _, row in failed_analyses.iterrows():
            with st.expander(f"Error al procesar al candidato `{row['candidate_id']}`"):
                st.error(row['error'])
                st.code(row.get('raw_output', 'No raw output available.'), language='json')
