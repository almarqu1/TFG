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
# --- Import PeftModel ---
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
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

# --- FUNCIÓN DE PREPARACIÓN DE DATOS PARA SILVER SET ---
# (Esta función no necesita cambios, ya que maneja los casos v1, v3 y v4)
def prepare_dataset_from_scratch(dataset_path: str, config: dict, prompt_id: str, is_reasoning_experiment: bool):
    """
    Función para datasets (como el Silver Set) que necesitan ser construidos
    haciendo un merge de varias fuentes de datos.
    """
    print("   -> Modo de carga Silver Set: Se requiere reconstrucción de prompts.")
    main_df = pd.read_json(os.path.join(PROJECT_ROOT, dataset_path), lines=True)
    cvs_df = pd.read_csv(os.path.join(PROJECT_ROOT, config['data_paths']['processed']['cvs']))
    offers_df = pd.read_csv(os.path.join(PROJECT_ROOT, config['data_paths']['processed']['offers']))
    cvs_df['cv_text'] = cvs_df.apply(lambda row: format_entity_text(row, get_cv_sections()), axis=1)
    offers_df['offer_text'] = offers_df.apply(lambda row: format_entity_text(row, get_offer_sections()), axis=1)
    merged_df = pd.merge(main_df, cvs_df[['candidate_id', 'cv_text']], on='candidate_id', how='inner')
    full_train_df = pd.merge(merged_df, offers_df[['job_id', 'offer_text']], on='job_id', how='inner')
    
    prompt_template = load_prompt_template(prompt_id)
    
    def create_full_text(row):
        prompt = prompt_template.format(cv=row['cv_text'], job_description=row['offer_text'])
        if is_reasoning_experiment:
            just_dict = row['teacher_justification']
            strengths = "\n".join([f"- {s}" for s in just_dict.get('strengths', [])]) if just_dict.get('strengths') else "N/A"
            gaps = "\n".join([f"- {g}" for g in just_dict.get('concerns_and_gaps', [])]) if just_dict.get('concerns_and_gaps') else "N/A"
            potential = "\n".join([f"- {p}" for p in just_dict.get('potential', [])]) if just_dict.get('potential') else "N/A"
            summary = just_dict.get('final_summary', 'N/A')
            response = (
                f"Strengths:\n{strengths}\n\n"
                f"Concerns and Gaps:\n{gaps}\n\n"
                f"Potential:\n{potential}\n\n"
                f"Final Summary: {summary}\n\n"
                f"Score: {row['teacher_score']}"
            )
        else:
            response = f"Score: {row['teacher_score']}"
        return prompt + response
        
    full_train_df['full_text'] = full_train_df.apply(create_full_text, axis=1)
    return Dataset.from_pandas(full_train_df[['full_text']])

# --- FUNCIÓN DE PREPARACIÓN DE DATOS PARA GOLD SET (MODIFICADA) ---
def prepare_gold_set_for_finetuning(dataset_path: str, prompt_id: str):
    """
    Función ADAPTADA para el Golden Set.
    Construye el texto de entrenamiento usando el prompt de RAZONAMIENTO (`prompt_id`)
    pero la respuesta de SOLO-SCORE de los datos (`response`).
    """
    print("   -> Modo de carga Golden Set: Construyendo texto para calibración de score.")
    df = pd.read_json(os.path.join(PROJECT_ROOT, dataset_path), lines=True)
    
    # Cargamos la plantilla de razonamiento (ej. S-04)
    prompt_template = load_prompt_template(prompt_id)
    instruction_part = prompt_template.split("[CV]")[0]
    
    # Para cada fila:
    # 1. Tomamos la instrucción del prompt de razonamiento.
    # 2. Le añadimos el contenido del CV y la Oferta (que viene en la columna 'prompt' del JSONL).
    # 3. Le añadimos el sufijo '[ANALYSIS]\n' para que el modelo sepa que debe empezar a "pensar".
    # 4. Concatenamos la respuesta final, que es SOLO el score del JSONL.
    df['full_text'] = instruction_part + df['prompt'] + "\n\n[ANALYSIS]\n" + df['response']
    
    return Dataset.from_pandas(df[['full_text']])


def main(experiment_name: str):
    """Función principal que ejecuta el pipeline de entrenamiento para un experimento específico."""
    
    global tokenizer

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
    os.environ["WANDB_RUN_NAME"] = exp_config.get('run_name', experiment_name)

    print(f"--- [Paso 2/7] Preparando tokenizer para '{model_config['base_model_name']}' ---")
    tokenizer = AutoTokenizer.from_pretrained(model_config['base_model_name'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"--- [Paso 3/7] Cargando y procesando dataset desde '{exp_config['dataset_path']}' ---")
    dataset_path = exp_config['dataset_path']
    prompt_id = exp_config.get('prompt_id', config['student_model']['prompt_id'])
    print(f"   -> Usando prompt_id: {prompt_id}")

    # Lógica condicional para cargar y procesar el dataset
    if "gold_standard" in dataset_path:
        # Usamos la nueva función específica para el Golden Set
        dataset = prepare_gold_set_for_finetuning(dataset_path, prompt_id)
    elif "silver_standard" in dataset_path:
        is_reasoning = "reasoning" in experiment_name.lower()
        dataset = prepare_dataset_from_scratch(dataset_path, config, prompt_id, is_reasoning_experiment=is_reasoning)
    else:
        raise ValueError(f"Ruta de dataset no reconocida: {dataset_path}")

    def tokenize_function(examples):
        return tokenizer(examples['full_text'], truncation=True, max_length=2048)

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
    print(f"Dataset procesado. Número de ejemplos: {len(tokenized_dataset)}")
    
    print("--- [Paso 4/7] Configurando modelo base con cuantización y LoRA ---")
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(model_config['base_model_name'], quantization_config=bnb_config, device_map="auto")
    model = prepare_model_for_kbit_training(model)
    
    base_lora_path = exp_config.get("base_lora_adapters_path")
    if base_lora_path:
        print(f"   -> Cargando adaptadores LoRA existentes desde: {base_lora_path}")
        model = PeftModel.from_pretrained(model, base_lora_path, is_trainable=True)
    else:
        print("   -> Creando nuevos adaptadores LoRA.")
        lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], lora_dropout=0.1, bias="none", task_type="CAUSAL_LM")
        model = get_peft_model(model, lora_config)
    
    model.print_trainable_parameters()

    print("--- [Paso 5/7] Definiendo los argumentos de entrenamiento ---")
    per_device_batch_size = train_params['batch_size']
    gradient_accumulation_steps = 1
    
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
    
    # Renombrar el experimento v5 para mayor claridad
    if args.experiment_name == "v5_silver_and_gold_reasoning":
        print("Advertencia: El nombre 'v5_silver_and_gold_reasoning' está obsoleto.")
        print("Usando la nueva configuración 'v5_gold_refinement'. Asegúrate de que está en tu config.yaml.")
        args.experiment_name = "v5_gold_refinement"

    main(args.experiment_name)