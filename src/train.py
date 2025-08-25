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

# Importamos TODAS nuestras utilidades personalizadas
from src.utils import (
    load_config, 
    load_prompt_template, 
    format_entity_text, 
    get_cv_sections, 
    get_offer_sections
)

def main():
    """Función principal que ejecuta todo el pipeline de entrenamiento."""
    
    print("--- [Paso 1/8] Cargando configuración del proyecto ---")
    config = load_config()
    model_config = config['student_model']
    train_config = config['training']
    data_paths = config['data_paths']
    os.environ["WANDB_RUN_NAME"] = f"qwen-4b-full-silver-set-e{train_config['num_epochs']}-lr{train_config['learning_rate']}"
    print(f"--- [Paso 2/8] Preparando tokenizer y prompt para '{model_config['base_model_name']}' ---")
    tokenizer = AutoTokenizer.from_pretrained(model_config['base_model_name'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    prompt_template = load_prompt_template(model_config['prompt_id'])

    print(f"--- [Paso 3/8] Cargando y combinando datasets para reconstruir los pares de entrenamiento ---")
    silver_set_df = pd.read_json(data_paths['intermediate']['silver_standard_train'], lines=True)
    #silver_set_df = silver_set_df.head(100)  # Para pruebas rápidas, usar solo los primeros 100 ejemplos
    cvs_df = pd.read_csv(data_paths['processed']['cvs'])
    offers_df = pd.read_csv(data_paths['processed']['offers'])

    print("   -> Aplicando formato enriquecido a CVs y ofertas (usando funciones de src.utils)...")
    cvs_df['formatted_cv'] = cvs_df.apply(lambda row: format_entity_text(row, get_cv_sections()), axis=1)
    offers_df['formatted_offer'] = offers_df.apply(lambda row: format_entity_text(row, get_offer_sections()), axis=1)

    print("   -> Combinando Silver Set con textos de CVs y Ofertas...")
    merged_df = pd.merge(silver_set_df, cvs_df[['candidate_id', 'formatted_cv']], on='candidate_id', how='inner')
    full_train_df = pd.merge(merged_df, offers_df[['job_id', 'formatted_offer']], on='job_id', how='inner')
    
    final_df = full_train_df[['formatted_cv', 'formatted_offer', 'teacher_score']]
    dataset = Dataset.from_pandas(final_df)

    def format_and_tokenize(example):
        full_text = prompt_template.format(
        cv=example['formatted_cv'], 
        job_description=example['formatted_offer']
        ) + f"Score: {example['teacher_score']}"
        return tokenizer(full_text)

    tokenized_dataset = dataset.map(format_and_tokenize, remove_columns=dataset.column_names)
    print(f"Dataset reconstruido y procesado. Número de ejemplos: {len(tokenized_dataset)}")
    
    print("--- [Paso 4/8] Configurando cuantización en 4-bit (BitsAndBytes) ---")
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)

    print(f"--- [Paso 5/8] Cargando modelo base '{model_config['base_model_name']}' con cuantización ---")
    model = AutoModelForCausalLM.from_pretrained(model_config['base_model_name'], quantization_config=bnb_config, device_map="auto")
    model = prepare_model_for_kbit_training(model)

    print("--- [Paso 6/8] Configurando adaptadores LoRA ---")
    lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], lora_dropout=0.1, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_config)
    print("Resumen del modelo con adaptadores LoRA:")
    model.print_trainable_parameters()

    print("--- [Paso 7/8] Definiendo los argumentos y creando el Trainer ---")
    per_device_batch_size = 2 
    gradient_accumulation_steps = train_config['batch_size'] // per_device_batch_size
    training_args = TrainingArguments(output_dir=model_config['output_dir'], num_train_epochs=train_config['num_epochs'], per_device_train_batch_size=per_device_batch_size, gradient_accumulation_steps=gradient_accumulation_steps, learning_rate=float(train_config['learning_rate']), bf16=True, logging_strategy="steps", logging_steps=25, save_strategy="epoch", report_to="wandb", gradient_checkpointing=True)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(model=model, 
                      args=training_args, 
                      train_dataset=tokenized_dataset, 
                      tokenizer=tokenizer,
                      data_collator=data_collator)

    print("--- [Paso 8/8] Iniciando el fine-tuning... ---")
    trainer.train()
    print("✅ Entrenamiento completado con éxito.")

    final_save_path = os.path.join(model_config['output_dir'], "final_checkpoint")
    trainer.save_model(final_save_path)
    print(f"✅ Adaptadores LoRA finales guardados en: {final_save_path}")

if __name__ == "__main__":
    main()