import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def main():
    parser = argparse.ArgumentParser(description="Inférence avec Phi-2 fine-tuné (LoRA)")
    parser.add_argument("--base_model", type=str, default="microsoft/phi-2", help="ID du modèle de base")
    parser.add_argument("--lora_weights", type=str, required=True, help="Chemin local ou Hugging Face vers les poids LoRA")
    parser.add_argument("--max_tokens", type=int, default=150, help="Nombre maximum de tokens à générer")
    parser.add_argument("--temperature", type=float, default=0.2, help="Température de génération (basse pour du code)")
    args = parser.parse_args()

    print(f" Chargement du tokenizer et du modèle de base ({args.base_model})...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model, 
        torch_dtype=torch.float16, 
        device_map="auto",
        trust_remote_code=True
    )

    print(f" Application des poids LoRA depuis : {args.lora_weights}")
    model = PeftModel.from_pretrained(base_model, args.lora_weights)

    # Exemple de prompt : On simule l'amorce d'un script (pas une question !)
    default_prompt = (
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n\n"
        "# Load data.csv, clean missing values, and return the dataframe\n"
        "def load_and_clean_data(filepath):\n"
    )

    print("\n" + "="*50)
    print(" PROMPT D'ENTRÉE (Complétion causale) :")
    print("="*50)
    print(default_prompt)
    print("="*50 + "\n")

    inputs = tokenizer(default_prompt, return_tensors="pt").to(model.device)
    
    print(" Génération en cours...")
    outputs = model.generate(
        **inputs,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id
    )

    print("\n" + "="*50)
    print(" RÉSULTAT GÉNÉRÉ :")
    print("="*50)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    print("="*50 + "\n")

if __name__ == "__main__":
    main()