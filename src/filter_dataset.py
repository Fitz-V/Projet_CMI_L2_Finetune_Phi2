import re
import os
from datasets import load_dataset

def main():
    print("Chargement du dataset localement...")
    # Attention : nécessite d'être authentifié sur Hugging Face CLI
    dataset = load_dataset(
        "bigcode/the-stack-dedup", 
        split="train", 
        data_dir="data/python",
        trust_remote_code=True
    )

    print(f"Dataset chargé avec {len(dataset)} exemples.")

    # Définition du regex pour isoler l'usage des librairies cibles
    pattern = re.compile(r"(import\s+pandas|import\s+matplotlib)")

    def has_pandas_or_matplotlib(example):
        content = example.get("content", "")
        return bool(pattern.search(content))

    print("Application du filtre...")
    filtered_dataset = dataset.filter(
        has_pandas_or_matplotlib,
        desc="Filtrage en cours" 
    )

    print(f"Dataset filtré : {len(filtered_dataset)} exemples restants.")

    # Sauvegarde du dataset filtré
    save_path = "./data/filtered"
    os.makedirs(save_path, exist_ok=True)
    filtered_dataset.save_to_disk(save_path)
    
    print(f"Sauvegarde terminée dans : {save_path}")

if __name__ == "__main__":
    main()