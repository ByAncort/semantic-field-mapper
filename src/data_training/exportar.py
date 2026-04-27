# exportar.py
import json
from src.data_training.datos_entrenamiento import fp, fn, datos_base, similares, distintos


def exportar_jsonl(filepath="training_data.jsonl"):
    todas = fp() + fn() + datos_base() + similares() + distintos()

    # Deduplicar
    vistas = set()
    unicas = []
    for a, b, label in todas:
        key = (min(a, b), max(a, b))  # normalizar orden
        if key not in vistas:
            vistas.add(key)
            unicas.append({"field_a": a, "field_b": b, "match": label})

    with open(filepath, "w", encoding="utf-8") as f:
        for row in unicas:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"{len(unicas)} pares guardados en {filepath}")

def cargar_pares(filepath="training_data.jsonl"):
    with open(filepath, encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def buscar_campo_match(campo, match, pares):
    return [p for p in pares
            if campo in (p["field_a"], p["field_b"]) and p["match"] == match]

pares = cargar_pares()
var="email"
si = buscar_campo_match(var, True, pares)   # pares donde rfc SI matchea
no=buscar_campo_match(var, False, pares)  # pares donde rfc NO matchea
filas =si+no
for datos in filas:
    print(datos)


