import os
import gc
import yaml
import torch
from datasets import Dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

def load_config(config_path="./configs/train_config.yaml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def get_dataset_chunk(path, index, total_chunks=12, test_size=0.2, seed=42):
    dataset = load_from_disk(path)
    total_size = len(dataset)
    chunk_size = total_size // total_chunks
    start = index * chunk_size
    end = start + chunk_size
    chunk = dataset.select(range(start, min(end, total_size)))
    return chunk.train_test_split(test_size=test_size, seed=seed)

def main():
    # 1. Chargement de la configuration
    cfg = load_config()
    os.makedirs(cfg['training']['output_dir'], exist_ok=True)

    # 2. Chargement du Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        cfg['model']['base_model'],
        trust_remote_code=True,
        cache_dir=cfg['model']['cache_dir']
    )
    tokenizer.pad_token = tokenizer.eos_token

    # Preuve technique : Tokenisation en auto-complétion pure (pas d'instruction)
    def tokenize_function(example):
        result = tokenizer(
            example["content"],
            padding="max_length",
            truncation=True,
            max_length=cfg['training']['max_seq_length'],
            return_tensors='pt'
        )
        result["labels"] = result["input_ids"].clone()
        return result

    # 3. Chargement des données
    print(f"Chargement du chunk {cfg['dataset']['chunk_index']}...")
    split_dataset = get_dataset_chunk(
        cfg['dataset']['path'], 
        cfg['dataset']['chunk_index'],
        test_size=cfg['dataset']['test_size'],
        seed=cfg['dataset']['seed']
    )
    
    tokenized_dataset = split_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=split_dataset["train"].column_names
    )

    # 4. Chargement du Modèle et LoRA
    model = AutoModelForCausalLM.from_pretrained(
        cfg['model']['base_model'],
        trust_remote_code=True,
        torch_dtype=torch.float16 if cfg['training']['fp16'] else torch.float32,
        device_map="auto",
        cache_dir=cfg['model']['cache_dir']
    )
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=cfg['lora']['r'],
        lora_alpha=cfg['lora']['alpha'],
        lora_dropout=cfg['lora']['dropout'],
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=cfg['lora']['target_modules']
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 5. Configuration et Entraînement
    training_args = SFTConfig(
        output_dir=os.path.join(cfg['training']['output_dir'], f"chunk_{cfg['dataset']['chunk_index']}"),
        per_device_train_batch_size=cfg['training']['per_device_train_batch_size'],
        gradient_accumulation_steps=cfg['training']['gradient_accumulation_steps'],
        learning_rate=float(cfg['training']['learning_rate']),
        num_train_epochs=cfg['training']['num_train_epochs'],
        logging_steps=cfg['training']['logging_steps'],
        save_steps=cfg['training']['save_steps'],
        fp16=cfg['training']['fp16'],
        optim=cfg['training']['optim'],
        lr_scheduler_type=cfg['training']['lr_scheduler_type'],
        warmup_ratio=cfg['training']['warmup_ratio'],
        max_grad_norm=cfg['training']['max_grad_norm'],
        report_to="none",
        group_by_length=cfg['training']['group_by_length'],
        max_seq_length=cfg['training']['max_seq_length']
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        peft_config=lora_config,
        args=training_args
    )

    print("Début de l'entraînement...")
    trainer.train()

    # Sauvegarde finale
    final_save_path = os.path.join(cfg['training']['output_dir'], f"chunk_{cfg['dataset']['chunk_index']}", "final_model")
    model.save_pretrained(final_save_path)
    print(f"Entraînement terminé. Adaptateurs sauvegardés dans : {final_save_path}")

    # Ménage mémoire
    torch.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":
    main()