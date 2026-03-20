#!/usr/bin/env python3
"""
Évalue les réponses générées sur DS-1000.
Adapté du script test_ds1000.py original.
"""
import argparse
import json
import os
import pandas as pd
from tqdm import tqdm
import concurrent.futures as cfuts
from datasets import load_dataset
from src.utils import execution
from src.utils.ds1000_utils import postprocess

def evaluate(answers_file, output_txt):
    # Charger DS-1000
    ds1000 = list(load_dataset("xlangai/DS-1000")["test"])

    # Charger les réponses
    with open(answers_file, "r") as f:
        generated = [json.loads(l) for l in f.readlines()]
    # Trier par task_id et appliquer post-traitement
    answers = [None] * len(ds1000)
    for g in generated:
        answers[g['task_id']] = postprocess(g['completion'])

    # Évaluation parallèle
    ds1000_results = []
    with cfuts.ProcessPoolExecutor(max_workers=4) as executor:
        futs = []
        for p in ds1000:
            pid = int(p['metadata']['problem_id'])
            test_program = (
                p['code_context'] + '\n'
                + f'code = {repr(answers[pid])}\n'
                + 'test_execution(code)\n'
                + ('test_string(code)\n' if 'test_string(' in p['code_context'] else '\n')
            )
            futs.append(executor.submit(execution.check_correctness, test_program, timeout=120, completion_id=pid))

        for f in tqdm(cfuts.as_completed(futs), total=len(futs)):
            result = f.result()
            cid = result['completion_id']
            result['score'] = 1 if result['passed'] else 0
            result['library'] = ds1000[cid]['metadata']['library']
            result['perturbation_type'] = ds1000[cid]['metadata']['perturbation_type']
            ds1000_results.append(result)

    df = pd.DataFrame(ds1000_results)
    summary = df.agg({'score': ['count', 'mean']}).to_string()
    summary += '\n' + df.groupby('library')['score'].agg(['count', 'mean']).to_string()
    summary += '\n' + df.groupby('perturbation_type')['score'].agg(['count', 'mean']).to_string()

    with open(output_txt, 'w') as f:
        f.write(summary)
    print(summary)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers", type=str, required=True, help="Fichier .jsonl des réponses")
    parser.add_argument("--output", type=str, required=True, help="Fichier .txt de sortie")
    args = parser.parse_args()
    evaluate(args.answers, args.output)