"""
ensemble_predictor.py
─────────────────────
Predictor de similitud semántica con ensemble de 3 señales:

    score_final = w_nn * score_nn + w_emb * score_emb + w_rules * score_rules

Integrado con cross_validator.py (extraer_caracteristicas, tokenizar,
TOKENS_DIFERENCIADORES) y compatible con el modelo + scaler guardados
por model_validator().
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.training.cross_validator import (
    TOKENS_DIFERENCIADORES,
    _get_st_model,
    extraer_caracteristicas,
    tokenizar,
)
from src.config import MODEL_PATH, SCALER_PATH, TOKENIZER_PATH

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# SEÑALES INDIVIDUALES
# ─────────────────────────────────────────────────────────────────────────────

def _score_semantic(modelo, campo1: str, campo2: str) -> float:
    """
    Inferencia con el semantic_model (LSTM + Attention + features semánticos).
    Prepara los 7 inputs que espera el modelo multi-output.
    Usa el tokenizer guardado durante el entrenamiento si existe.
    """
    from src.training.semantic_features_extractor import (
        extraer_features_semanticos, create_default_context
    )
    from src.training.cross_validator import SimpleTokenizer
    import json

    if not hasattr(_score_semantic, '_tokenizer') or _score_semantic._tokenizer is None:
        # Intentar cargar tokenizer guardado
        if TOKENIZER_PATH.exists():
            with open(TOKENIZER_PATH, 'r', encoding='utf-8') as f:
                state = json.load(f)
            tokenizer = SimpleTokenizer(
                vocab_size=state.get('vocab_size', 10000),
                max_len=state.get('max_len', 20)
            )
            tokenizer.word2idx = {str(k): int(v) for k, v in state['word2idx'].items()}
            tokenizer.idx2word = {int(v): str(k) for k, v in state['word2idx'].items()}
            _score_semantic._tokenizer = tokenizer
            logger.info(f"✅ Tokenizer cargado desde {TOKENIZER_PATH.resolve()}")
        else:
            logger.warning(f"No se encontró tokenizer guardado en {TOKENIZER_PATH}. Usando tokenizer por defecto.")
            _score_semantic._tokenizer = SimpleTokenizer(vocab_size=10000, max_len=20)
    tokenizer = _score_semantic._tokenizer

    src_idx = np.array([tokenizer.tokenize_to_indices(campo1)])
    tgt_idx = np.array([tokenizer.tokenize_to_indices(campo2)])

    src_sem = np.array([extraer_features_semanticos(campo1)], dtype=np.float32)
    tgt_sem = np.array([extraer_features_semanticos(campo2)], dtype=np.float32)

    ctx = create_default_context().to_features()
    ctx_batch = np.array([ctx], dtype=np.float32)

    inputs = {
        'src_name': src_idx,
        'src_value': src_idx,
        'tgt_name': tgt_idx,
        'tgt_value': tgt_idx,
        'src_semantic': src_sem,
        'tgt_semantic': tgt_sem,
        'context': ctx_batch,
    }

    outputs = modelo.predict(inputs, verbose=0)
    if isinstance(outputs, dict):
        return float(outputs['similarity_score'][0][0])
    if isinstance(outputs, (list, tuple)):
        return float(outputs[0][0][0])
    return float(outputs[0][0])

def _score_nn(modelo, scaler, campo1: str, campo2: str):
    """
    Probabilidad sigmoid de la red neuronal.
    Soporta modelo clásico (single-output dense) y semantic_model (multi-output).
    Retorna: (score, source) donde source es 'nn', 'semantic' o 'rules_fallback'.
    """
    if hasattr(modelo, 'outputs') and len(modelo.outputs) > 1:
        if TOKENIZER_PATH.exists():
            try:
                score = _score_semantic(modelo, campo1, campo2)
                return score, "semantic"
            except Exception as e:
                logger.debug(f"semantic_model falló ({e}), usando fallback rules")
        else:
            logger.debug(f"semantic_model sin tokenizer guardado, usando reglas")
        return _score_rules(campo1, campo2), "rules_fallback"

    # Modelo antiguo (single output)
    features = extraer_caracteristicas(campo1, campo2, use_embeddings=True)
    if scaler is not None:
        features = scaler.transform([features])
    return float(modelo.predict(features, verbose=0)[0][0]), "nn"

def _score_emb(campo1: str, campo2: str) -> float:
    """
    Similitud coseno entre embeddings de sentence-transformers.
    Captura equivalencias semánticas cross-idioma no cubiertas por el diccionario:
        rfc ↔ tax_id,  KUNNR ↔ customer_number,  correo ↔ email
    Devuelve 0.0 si sentence-transformers no está instalado.
    """
    model = _get_st_model()
    if model is None:
        return 0.0
    e1 = model.encode(campo1)
    e2 = model.encode(campo2)
    return float(cosine_similarity([e1], [e2])[0][0])


def _score_rules(campo1: str, campo2: str) -> float:
    """
    Score de reglas determinístico basado en las features de mayor poder discriminatorio:
        F1  token_jaccard semántico     (+) similares
        F3  seq_sim                     (+) similares
        F11 match_semantico             (+) similares
        F9  penalización diferenciadora (−) no similares
    """
    f = extraer_caracteristicas(campo1, campo2, use_embeddings=False)
    token_jaccard   = f[0]
    seq_sim         = f[2]
    penalizacion    = f[8]
    match_semantico = f[10]

    raw = (
        0.45 * token_jaccard
        + 0.30 * seq_sim
        + 0.25 * match_semantico
        - 0.40 * penalizacion
    )
    return float(np.clip(raw, 0.0, 1.0))


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EnsembleConfig:
    """
    Pesos y umbral del ensemble.

    Recomendaciones:
        Con sentence-transformers      → (0.45, 0.30, 0.25)  ← por defecto
        Sin sentence-transformers      → (0.65, 0.00, 0.35)
        Dominio muy técnico (SAP, ERP) → (0.40, 0.35, 0.25)  ← emb pesa más
    """
    weight_nn:    float = 0.45
    weight_emb:   float = 0.30
    weight_rules: float = 0.25
    umbral:       float = 0.40

    def __post_init__(self):
        total = self.weight_nn + self.weight_emb + self.weight_rules
        if not (0.999 < total < 1.001):
            raise ValueError(f"Los pesos deben sumar 1.0, suman {total:.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# CLASE PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EnsemblePredictor:
    """
    Uso básico:
        predictor = EnsemblePredictor.from_paths()
        resultado = predictor.predecir("customer_id", "id_cliente")
        print(resultado["score_final"], resultado["son_similares"])
    """
    modelo: object
    scaler: object
    config: EnsembleConfig = field(default_factory=EnsembleConfig)

    # ── Constructor ───────────────────────────────────────────────────────────

    @classmethod
    def from_paths(
        cls,
        model_path:  str = MODEL_PATH,
        scaler_path: str = SCALER_PATH,
        config: Optional[EnsembleConfig] = None,
    ) -> "EnsemblePredictor":
        import joblib
        from tensorflow import keras
        from src.arq.neuron import AbsoluteValue
        import tensorflow as tf

        logger.info(f"Cargando modelo  : {model_path}")
        logger.info(f"Cargando scaler  : {scaler_path}")
        
        # Habilitar unsafe deserialization para modelos con Lambda layers guardados anteriormente
        keras.config.enable_unsafe_deserialization()
        
        custom_objects = {'AbsoluteValue': AbsoluteValue}
        
        try:
            # Intentar cargar el modelo normalmente
            modelo = keras.models.load_model(model_path, custom_objects=custom_objects)
            logger.info("✅ Modelo cargado exitosamente")
        except ValueError as e:
            # Si falla por problemas de deserialización de loss/metrics, cargar sin compilación
            if "deserialize" in str(e).lower() or "metric" in str(e).lower():
                logger.warning(f"⚠️  Problemas de deserialización detectados. Cargando arquitectura sin compilación...")
                modelo = keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
                
                # Recompilar con configuración válida
                modelo.compile(
                    optimizer=keras.optimizers.Adam(learning_rate=0.0005),
                    loss={
                        'similarity_score': 'binary_crossentropy',
                        'match_type': 'sparse_categorical_crossentropy',
                        'confidence': 'mse'
                    },
                    loss_weights={
                        'similarity_score': 1.0,
                        'match_type': 0.7,
                        'confidence': 0.3
                    },
                    metrics={
                        'similarity_score': ['accuracy', 'AUC', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
                        'match_type': ['accuracy'],
                        'confidence': ['mae']
                    }
                )
                logger.info("✅ Modelo recompilado exitosamente")
            else:
                raise
        
        scaler = joblib.load(scaler_path)
        return cls(modelo=modelo, scaler=scaler, config=config or EnsembleConfig())

    # ── Predicción individual ─────────────────────────────────────────────────

    def predecir(self, campo1: str, campo2: str) -> dict:
        """
        Combina las 3 señales y retorna un resultado detallado.

        Salida:
        {
            'campo1':        'customer_id',
            'campo2':        'id_cliente',
            'son_similares': True,
            'score_final':   0.731,
            'porcentaje':    '73.1%',
            'desglose': {
                'score_nn':    0.854,
                'score_emb':   0.712,
                'score_rules': 0.480,
            },
            'pesos':               {'weight_nn': 0.45, ...},
            'tokens_campo1':       [...],
            'tokens_campo2':       [...],
            'tokens_compartidos':  [...],
            'tokens_diferenciadores_1': [...],
            'tokens_diferenciadores_2': [...],
        }
        """
        cfg = self.config

        s_nn, source = _score_nn(self.modelo, self.scaler, campo1, campo2)
        s_emb = _score_emb(campo1, campo2)
        s_rules = _score_rules(campo1, campo2)
        if source == "semantic":
            weight_nn = 0.55
            weight_emb = 0.25
            weight_rules = 0.20
        elif source == "rules_fallback":
            weight_nn = 0.0
            weight_rules = 0.7
            weight_emb = 0.3
        else:
            weight_nn = cfg.weight_nn
            weight_emb = cfg.weight_emb
            weight_rules = cfg.weight_rules

        score_final = np.clip(
            weight_nn * s_nn +
            weight_emb * s_emb +
            weight_rules * s_rules,
            0.0, 1.0
        )
        if s_rules > 0.8:
            score_final = max(score_final, 0.9)

        t1 = tokenizar(campo1)
        t2 = tokenizar(campo2)
        if "id" in t1 and "id" in t2:
            score_final = max(score_final, 0.85)

        if "correo" in t1 and "correo" in t2:
            score_final = max(score_final, 0.85)
        return {
            "campo1":          campo1,
            "campo2":          campo2,
            "son_similares":   score_final >= cfg.umbral,
            "score_final":     round(score_final, 4),
            "porcentaje":      f"{score_final * 100:.1f}%",
            "desglose": {
                "score_nn":    round(s_nn,    4),
                "score_emb":   round(s_emb,   4),
                "score_rules": round(s_rules, 4),
            },
            "pesos": {
                "weight_nn":    cfg.weight_nn,
                "weight_emb":   cfg.weight_emb,
                "weight_rules": cfg.weight_rules,
            },
            "tokens_campo1":            sorted(t1),
            "tokens_campo2":            sorted(t2),
            "tokens_compartidos":       sorted(t1 & t2),
            "tokens_diferenciadores_1": sorted((t1 - t2) & TOKENS_DIFERENCIADORES),
            "tokens_diferenciadores_2": sorted((t2 - t1) & TOKENS_DIFERENCIADORES),
        }

    # ── Predicción multi-campo (comparar dos objetos JSON completos) ──────────

    def predecir_objetos(
        self,
        obj1: dict,
        obj2: dict,
        estrategia: str = "top3",
    ) -> dict:
        """
        Compara todos los pares de campos entre dos objetos de API.

        estrategia:
            'max'   → mejor match (más permisivo, bueno para detección)
            'avg'   → promedio de todos los pares (visión global)
            'top3'  → promedio de los 3 mejores (equilibrio entre max y avg)

        Ejemplo:
            api_a = {"customer_id": 1, "email": "a@b.com"}
            api_b = {"id_cliente":  1, "correo": "a@b.com"}
            resultado = predictor.predecir_objetos(api_a, api_b, estrategia="top3")
        """
        pares = []
        for k1 in obj1:
            for k2 in obj2:
                r = self.predecir(k1, k2)
                r["clave_obj1"] = k1
                r["clave_obj2"] = k2
                pares.append(r)

        scores = [p["score_final"] for p in pares]
        scores_sorted = sorted(scores, reverse=True)

        if estrategia == "max":
            score_global = scores_sorted[0]
        elif estrategia == "avg":
            score_global = float(np.mean(scores))
        elif estrategia == "top3":
            score_global = float(np.mean(scores_sorted[:3]))
        else:
            raise ValueError(f"Estrategia inválida: {estrategia!r}. Usa 'max', 'avg' o 'top3'.")

        score_global = round(score_global, 4)
        pares_sorted = sorted(pares, key=lambda p: p["score_final"], reverse=True)

        return {
            "score_global":    score_global,
            "son_similares":   score_global >= self.config.umbral,
            "estrategia":      estrategia,
            "total_pares":     len(pares),
            "mejor_par": {
                "campo_obj1": pares_sorted[0]["clave_obj1"],
                "campo_obj2": pares_sorted[0]["clave_obj2"],
                "score":      pares_sorted[0]["score_final"],
            },
            "todos_los_pares": pares_sorted,
        }

    # ── Evaluación del ensemble sobre un dataset etiquetado ──────────────────

    def evaluar_dataset(self, casos: list[tuple[str, str, int]]) -> dict:
        """
        Corre el ensemble sobre casos etiquetados y retorna métricas + casos difíciles.

        Útil para:
        - Ajustar pesos sin reentrenar
        - Identificar hard negatives / hard positives para data augmentation

        casos: [(campo1, campo2, etiqueta), ...]  etiqueta ∈ {0, 1}
        """
        tp = fp = tn = fn = 0
        hard_negatives: list[tuple] = []   # FP con score >= 0.55
        hard_positives: list[tuple] = []   # FN con score <= 0.35

        for campo1, campo2, etiqueta in casos:
            r     = self.predecir(campo1, campo2)
            pred  = int(r["son_similares"])
            score = r["score_final"]

            if etiqueta == 1 and pred == 1:
                tp += 1
            elif etiqueta == 0 and pred == 1:
                fp += 1
                if score >= 0.55:
                    hard_negatives.append((campo1, campo2, score))
            elif etiqueta == 1 and pred == 0:
                fn += 1
                if score <= 0.35:
                    hard_positives.append((campo1, campo2, score))
            else:
                tn += 1

        total     = tp + fp + tn + fn
        accuracy  = (tp + tn) / total         if total           else 0.0
        precision = tp / (tp + fp)            if (tp + fp)       else 0.0
        recall    = tp / (tp + fn)            if (tp + fn)       else 0.0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        return {
            "total":     total,
            "accuracy":  round(accuracy,  4),
            "precision": round(precision, 4),
            "recall":    round(recall,    4),
            "f1":        round(f1,        4),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "hard_negatives": hard_negatives,
            "hard_positives": hard_positives,
        }


# ─────────────────────────────────────────────────────────────────────────────
# DEMO  (python ensemble_predictor.py)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    predictor = EnsemblePredictor.from_paths()

    casos_demo = [
        ("rfc",                   "tax_id",                 1),
        ("email",                 "correo_electronico",     1),
        ("KUNNR",                 "customer_number",        1),
        ("custbody_mx_cfdi_uuid", "PFR_UUID_TB120",         1),
        ("cp",                    "codigo_postal",          1),
        ("nombre",                "first_name",             1),
        ("customer_email",        "vendor_email",           0),
        ("shipping_address",      "billing_address",        0),
        ("precio_venta",          "precio_costo",           0),
        ("apellido_paterno",      "apellido_materno",       0),
        ("folio_factura",         "folio_pedido",           0),
    ]

    print(f"\n{'Campo1':30} {'Campo2':28} {'NN':6} {'Emb':6} {'Rules':6} {'Final':7} {'OK?'}")
    print("─" * 98)

    for c1, c2, esp in casos_demo:
        r    = predictor.predecir(c1, c2)
        d    = r["desglose"]
        pred = int(r["son_similares"])
        marca = "✅" if pred == esp else "❌"
        print(
            f"{marca} {c1:30} {c2:28} "
            f"{d['score_nn']:.3f}  {d['score_emb']:.3f}  {d['score_rules']:.3f}  "
            f"{r['score_final']:.3f}   esp={esp}"
        )

    print("\n── Métricas ──")
    met = predictor.evaluar_dataset(casos_demo)
    print(json.dumps(
        {k: v for k, v in met.items() if k not in ("hard_negatives", "hard_positives")},
        indent=2
    ))

    if met["hard_negatives"]:
        print("\n⚠️  Hard negatives (agregar al training como label=0):")
        for c1, c2, s in met["hard_negatives"]:
            print(f"   {c1} ↔ {c2}  score={s:.3f}")

    if met["hard_positives"]:
        print("\n⚠️  Hard positives (agregar al training como label=1):")
        for c1, c2, s in met["hard_positives"]:
            print(f"   {c1} ↔ {c2}  score={s:.3f}")

    print("\n── Multi-campo demo ──")
    api_a = {"customer_id": 1, "email": "x@y.com", "cp": "06600"}
    api_b = {"id_cliente":  1, "correo": "x@y.com", "codigo_postal": "06600"}
    res = predictor.predecir_objetos(api_a, api_b, estrategia="top3")
    print(f"Score global : {res['score_global']}  →  {'MATCH ✅' if res['son_similares'] else 'NO MATCH ❌'}")
    print(f"Mejor par    : {res['mejor_par']}")