import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import os
import argparse
import sys

# --- Configuración del Path ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from src.utils import (
    load_config,
    load_prompt_template,
    format_entity_text,
    get_cv_sections,
    get_offer_sections
)

# Hacemos el tokenizer global para que las funciones de mapeo puedan acceder a él
tokenizer = None

def prepare_dataset_from_scratch(dataset_path: str, config: dict, is_reasoning_experiment: bool):
    """
    Función para datasets (como el Silver Set) que necesitan ser construidos
    haciendo un merge de varias fuentes de datos.
    """
    print("   -> Modo de carga Silver Set: Se requiere reconstrucción de prompts.")
    
    # Cargar los dataframes necesarios
    main_df = pd.read_json(os.path.join(PROJECT_ROOT, dataset_path), lines=True)
    cvs_df = pd.read_csv(os.path.join(PROJECT_ROOT, config['data_paths']['processed']['cvs']))
    offers_df = pd.read_csv(os.path.join(PROJECT_ROOT, config['data_paths']['processed']['offers']))
    
    # Aplicar formato enriquecido
    print("   -> Aplicando formato enriquecido a CVs y ofertas...")
    cvs_df['cv_text'] = cvs_df.apply(lambda row: format_entity_text(row, get_cv_sections()), axis=1)
    offers_df['offer_text'] = offers_df.apply(lambda row: format_entity_text(row, get_offer_sections()), axis=1)

    # Combinar los dataframes
    print("   -> Combinando datos para crear los pares de entrenamiento...")
    merged_df = pd.merge(main_df, cvs_df[['candidate_id', 'cv_text']], on='candidate_id', how='inner')
    full_train_df = pd.merge(merged_df, offers_df[['job_id', 'offer_text']], on='job_id', how='inner')
    
    # Cargar la plantilla del prompt
    prompt_template = load_prompt_template(config['student_model']['prompt_id'])
    
    # Construir el texto final para el entrenamiento
    def create_full_text(row):
        # El prompt_template ya se ha cargado fuera de esta función anidada.
        # Formateamos el prompt con los textos del CV y la oferta.
        prompt = prompt_template.format(cv=row['cv_text'], job_description=row['offer_text'])
        
        # Ahora, construimos la "respuesta" que el modelo debe aprender a generar.
        if is_reasoning_experiment:
            # Para el experimento de razonamiento (v4), la respuesta es la justificación completa.
            just_dict = row['teacher_justification']
            
            # Formateamos la justificación a un string legible, asegurándonos de manejar casos vacíos.
            strengths = "\n".join([f"- {s}" for s in just_dict.get('strengths', [])]) if just_dict.get('strengths') else "N/A"
            gaps = "\n".join([f"- {g}" for g in just_dict.get('concerns_and_gaps', [])]) if just_dict.get('concerns_and_gaps') else "N/A"
            potential = "\n".join([f"- {p}" for p in just_dict.get('potential', [])]) if just_dict.get('potential') else "N/A"
            summary = just_dict.get('final_summary', 'N/A')
            
            # La respuesta es la cadena de texto completa que sigue al prompt.
            response = (
                f"Strengths:\n{strengths}\n\n"
                f"Concerns and Gaps:\n{gaps}\n\n"
                f"Potential:\n{potential}\n\n"
                f"Final Summary: {summary}\n\n"
                f"Score: {row['teacher_score']}"
            )
        else:
            # Para los experimentos "Score-Only" (v3), la respuesta es solo el score.
            response = f"Score: {row['teacher_score']}"
            
        # El texto completo para el entrenamiento es la concatenación del prompt y la respuesta esperada.
        # El modelo aprenderá a generar la 'response' cuando se le dé el 'prompt'.
        return prompt + response
        
    full_train_df['full_text'] = full_train_df.apply(create_full_text, axis=1)
    
    return Dataset.from_pandas(full_train_df[['full_text']])


def prepare_preformatted_dataset(dataset_path: str):
    """

    Función para datasets (como el Golden Set) que ya vienen pre-formateados
    con las columnas 'prompt' y 'response'.
    """
    print("   -> Modo de carga Golden Set: Los prompts ya están pre-formateados.")
    df = pd.read_json(os.path.join(PROJECT_ROOT, dataset_path), lines=True)
    df['full_text'] = df['prompt'] + df['response']
    return Dataset.from_pandas(df[['full_text']])


def main(experiment_name: str):
    """Función principal que ejecuta el pipeline de entrenamiento para un experimento específico."""
    
    global tokenizer # Permitir que esta función modifique el tokenizer global

    print(f"--- [Paso 1/7] Cargando configuración para el experimento: '{experiment_name}' ---")
    config = load_config()
    try:
        exp_config = config['experiments'][experiment_name]
    except KeyError:
        print(f"❌ Error: El experimento '{experiment_name}' no se encontró en config.yaml.")
        return
    
    model_config = config['student_model']
    train_params = exp_config['training_params']
    
    os.environ["WANDB_PROJECT"] = "TFG_DistilMatch"
    os.environ["WANDB_RUN_NAME"] = exp_config['run_name']

    print(f"--- [Paso 2/7] Preparando tokenizer para '{model_config['base_model_name']}' ---")
    tokenizer = AutoTokenizer.from_pretrained(model_config['base_model_name'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"--- [Paso 3/7] Cargando y procesando dataset desde '{exp_config['dataset_path']}' ---")
    dataset_path = exp_config['dataset_path']

    is_reasoning = "reasoning" in experiment_name.lower()

    # Lógica condicional para cargar el dataset correcto
    if "gold_standard" in dataset_path:
        dataset = prepare_preformatted_dataset(dataset_path)
    elif "silver_standard" in dataset_path:
        dataset = prepare_dataset_from_scratch(dataset_path, config, is_reasoning_experiment=is_reasoning)

    else:
        raise ValueError(f"Ruta de dataset no reconocida: {dataset_path}")

    # Tokenización
    def tokenize_function(examples):
        return tokenizer(examples['full_text'], truncation=True, max_length=2048) # Aumentado el max_length

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
    print(f"Dataset procesado. Número de ejemplos: {len(tokenized_dataset)}")
    
    print("--- [Paso 4/7] Configurando modelo base con cuantización y LoRA ---")
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(model_config['base_model_name'], quantization_config=bnb_config, device_map="auto")
    model = prepare_model_for_kbit_training(model)
    
    lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], lora_dropout=0.1, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("--- [Paso 5/7] Definiendo los argumentos de entrenamiento ---")
    # Asegurarse de que el batch size es al menos 1
    per_device_batch_size = min(train_params['batch_size'], 4) # Ajustado para VRAM
    if train_params['batch_size'] < per_device_batch_size:
        gradient_accumulation_steps = 1
    else:
        gradient_accumulation_steps = train_params['batch_size'] // per_device_batch_size
    
    training_args = TrainingArguments(
        output_dir=exp_config['output_dir'], 
        num_train_epochs=train_params['num_epochs'], 
        per_device_train_batch_size=per_device_batch_size, 
        gradient_accumulation_steps=gradient_accumulation_steps, 
        learning_rate=float(train_params['learning_rate']), 
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        logging_strategy="steps", 
        logging_steps=10, 
        save_strategy="epoch", 
        report_to="wandb"
    )
    
    print("--- [Paso 6/7] Creando el Trainer ---")
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(model=model, args=training_args, train_dataset=tokenized_dataset, tokenizer=tokenizer, data_collator=data_collator)

    print(f"--- [Paso 7/7] Iniciando fine-tuning para '{experiment_name}'... ---")
    trainer.train()
    print("✅ Entrenamiento completado con éxito.")

    final_save_path = os.path.join(exp_config['output_dir'], "final_checkpoint")
    trainer.save_model(final_save_path)
    print(f"✅ Adaptadores LoRA finales guardados en: {final_save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ejecutar un experimento de fine-tuning específico.")
    parser.add_argument("experiment_name", type=str, help="El nombre del experimento a ejecutar (definido en config.yaml).")
    args = parser.parse_args()
    main(args.experiment_name)