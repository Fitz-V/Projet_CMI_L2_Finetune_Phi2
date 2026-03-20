# Poids du Modèle (LoRA Adapters)

Ce dossier est l'emplacement par défaut pour stocker les poids du modèle générés par le script d'entraînement (`../src/train.py`) ou téléchargés pour l'inférence.

## Téléchargement des Poids
Les adaptateurs LoRA finaux (issus du fine-tuning sur le chunk_0) sont hébergés publiquement sur Hugging Face pour garantir la reproductibilité.

**Lien du modèle :** https://huggingface.co/jhondoe789/projet_cmi_L2_phi-2_lora

## Utilisation Locale
Si vous souhaitez exécuter le script d'inférence ou d'évaluation en local, vous avez deux options :
1. **Directement via Hugging Face (Recommandé) :** Passez simplement le lien du dépôt distant à l'argument `--lora_weights` de nos scripts. Les bibliothèques `peft` et `transformers` les téléchargeront et les mettront en cache automatiquement.
2. **Manuellement :** Téléchargez les fichiers depuis Hugging Face et placez-les dans un sous-dossier ici (ex: `models/phi2_chunk0/`), puis indiquez ce chemin relatif lors de l'exécution des scripts.
