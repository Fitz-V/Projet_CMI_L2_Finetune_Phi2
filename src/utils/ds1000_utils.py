"""
Utilitaires d'exécution isolée et d'évaluation.
Code adapté du dépôt officiel DS-1000 (https://github.com/HKUNLP/DS-1000)
"""
def postprocess(code):
    """
    Nettoie le code généré par les modèles (supprime les balises, les backticks, etc.)
    """
    if isinstance(code, list):
        code = code[0]
    code = code.split('</code>')[0]
    code = code.replace('```python', '').replace('```', '')
    code = code.split('\nEND SOLUTION')[0]
    code = code.replace('<code>', '')
    return code.strip()