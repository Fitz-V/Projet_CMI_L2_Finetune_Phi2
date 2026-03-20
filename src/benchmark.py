import os
import json
import argparse
import logging
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="microsoft/phi-2", help="Nom du modèle de base")
    parser.add_argument("--lora_weights", type=str, required=True, help="Chemin vers les poids LoRA")
    parser.add_argument("--output_file", type=str, required=True, help="Fichier .jsonl de sortie")
    parser.add_argument("--batch_size", type=int, default=4, help="Taille du batch pour la génération")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Nombre max de tokens à générer")
    parser.add_argument("--temperature", type=float, default=0.7, help="Température pour l'échantillonnage")
    args = parser.parse_args()

    # Chargement du modèle
    logging.info("Chargement du modèle de base...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, args.lora_weights)
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

    # Chargement du dataset DS-1000
    logging.info("Chargement du dataset DS-1000...")
    dataset = load_dataset("xlangai/DS-1000")["test"]

    # Reprise
    results = {}
    if os.path.exists(args.output_file):
        with open(args.output_file, "r") as f:
            for line in f:
                entry = json.loads(line)
                results[entry["task_id"]] = entry["completion"]
        logging.info(f"Reprise : {len(results)} réponses déjà générées.")

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    with open(args.output_file, "a") as f:
        task_ids = list(range(len(dataset)))
        task_ids = [tid for tid in task_ids if tid not in results]

        for i in tqdm(range(0, len(task_ids), args.batch_size), desc="Génération"):
            batch_ids = task_ids[i:i + args.batch_size]
            batch_items = [dataset[tid] for tid in batch_ids]
            prompts = [item["prompt"] + "\n# solution:\n" for item in batch_items]

            try:
                outputs = generator(prompts, max_new_tokens=args.max_new_tokens, do_sample=True, temperature=args.temperature)
            except Exception as e:
                logging.error(f"Erreur sur batch {batch_ids}: {e}")
                outputs = [{} for _ in prompts]

            for tid, prompt, output in zip(batch_ids, prompts, outputs):
                try:
                    generated = output[0]["generated_text"] if isinstance(output, list) else output["generated_text"]
                    completion = generated.replace(prompt, "").strip()
                except Exception as e:
                    logging.error(f"Erreur parsing task_id {tid}: {e}")
                    completion = ""

                results[tid] = completion
                f.write(json.dumps({"task_id": tid, "completion": completion}) + "\n")
                f.flush()

    logging.info("Génération terminée.")

if __name__ == "__main__":
    main()