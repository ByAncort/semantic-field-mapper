from __future__ import annotations

import logging
import re
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.utils import compute_class_weight
from tensorflow import keras

import src.db.config_mongodb as db_setup
from src.config import MODEL_PATH, SCALER_PATH, TOKENIZER_PATH
from src.arq.neuron import Neuron

logger = logging.getLogger(__name__)

# ── Config MongoDB ─────────────────────────────────────────────────────────────
config = db_setup.SemanticConfigMongoDb()

TOKENS_DIFERENCIADORES = config.get_tokens_diferenciadores()
TOKENS_IDENTIDAD       = config.get_tokens_identidad()
STOPWORDS_SISTEMA      = config.get_stopwords()
GRUPOS_SEMANTICOS      = config.get_grupos_semanticos()
INDICE_SEMANTICO       = config.get_indice_semantico()



# SENTENCE-TRANSFORMER  (singleton — se carga una sola vez por proceso)
ST_MODEL_NAME = "all-MiniLM-L6-v2"   # rápido, 384-dim, muy bueno para nombres cortos


@lru_cache(maxsize=1)
def _get_st_model():
    """
    Carga el modelo sentence-transformers la primera vez que se llama.
    Las llamadas siguientes devuelven la instancia cacheada.

    Si sentence-transformers no está instalado, devuelve None y F12 = 0.0.
    Instalar: pip install sentence-transformers
    """
    try:
        from sentence_transformers import SentenceTransformer
        logger.info(f"[ST] Cargando encoder semántico: {ST_MODEL_NAME}")
        return SentenceTransformer(ST_MODEL_NAME)
    except ImportError:
        logger.warning(
            "[ST] sentence-transformers no disponible — F12 será 0.0.\n"
            "     Instala con: pip install sentence-transformers"
        )
        return None
def embedding_sim(texto1: str, texto2: str) -> float:
    """
    F12: similitud coseno entre los embeddings de los dos nombres de campo.
    Detecta equivalencias semánticas cross-idioma sin diccionario:
        rfc ↔ tax_id, email ↔ correo_electronico, KUNNR ↔ customer_number
    """
    model = _get_st_model()
    if model is None:
        return 0.0
    e1 = model.encode(texto1)
    e2 = model.encode(texto2)
    return float(cosine_similarity([e1], [e2])[0][0])
# TOKENIZACIÓN
def mismo_grupo_semantico(tokens1: set, tokens2: set) -> float:
    grupos1 = {INDICE_SEMANTICO[t] for t in tokens1 if t in INDICE_SEMANTICO}
    grupos2 = {INDICE_SEMANTICO[t] for t in tokens2 if t in INDICE_SEMANTICO}
    return 1.0 if grupos1 & grupos2 else 0.0
def tokenizar(texto: str) -> set:
    """
    Tokeniza preservando tokens diferenciadores.
    NO elimina 'customer', 'vendor', 'billing', 'shipping', etc.
    Solo elimina prefijos de sistema sin semántica.
    """
    texto = texto.lower()
    texto = re.sub(r'[.\-/\\]', '_', texto)
    texto = re.sub(r'([a-z])([A-Z])', r'\1_\2', texto)
    texto = re.sub(r'([a-zA-Z])(\d)', r'\1_\2', texto)
    texto = re.sub(r'(\d)([a-zA-Z])', r'\1_\2', texto)
    tokens = re.split(r'[^a-z0-9]+', texto)
    return {t for t in tokens if t and t not in STOPWORDS_SISTEMA and len(t) >= 2}
def tokenizar_con_normalizacion(texto: str) -> set:
    """
    Normaliza cada token a su representante de grupo semántico si existe.
    tax → rfc (grupo),  correo → email (grupo),  customer → customer (sin grupo)
    """
    return {INDICE_SEMANTICO.get(t, t) for t in tokenizar(texto)}

# FEATURE ENGINEERING  (F1–F12)
def lcs_ratio(a: str, b: str) -> float:
    """Longest Common Substring normalizado por longitud máxima."""
    best = 0
    la, lb = len(a), len(b)
    for i in range(la):
        for j in range(lb):
            k = 0
            while i + k < la and j + k < lb and a[i + k] == b[j + k]:
                k += 1
            best = max(best, k)
    return best / max(la, lb) if max(la, lb) > 0 else 0.0

def extraer_caracteristicas(texto1: str, texto2: str, use_embeddings: bool = True) -> list[float]:
    """
    Extrae el vector de features para un par de nombres de campo.

    F1–F11  features manuales
    F12     similitud coseno de embeddings  (requiere sentence-transformers)

    Args:
        use_embeddings: False deshabilita F12 (útil para compatibilidad con modelos
                        entrenados sin embeddings, o cuando ST no está disponible).
    """
    t1_lower = texto1.lower()
    t2_lower = texto2.lower()

    tokens1      = tokenizar(texto1)
    tokens2      = tokenizar(texto2)
    tokens1_norm = tokenizar_con_normalizacion(texto1)
    tokens2_norm = tokenizar_con_normalizacion(texto2)

    interseccion_norm = tokens1_norm & tokens2_norm
    union_norm        = tokens1_norm | tokens2_norm
    interseccion_raw  = tokens1 & tokens2

    # F1: Token Jaccard semántico
    token_jaccard = len(interseccion_norm) / len(union_norm) if union_norm else 0.0
    # F2: Ratio tokens compartidos semántico
    tokens_compartidos_ratio = min(
        len(interseccion_norm) / max(len(tokens1_norm), len(tokens2_norm), 1), 1.0
    )

    # F3–F7: métricas sobre el string raw
    seq_sim    = SequenceMatcher(None, t1_lower, t2_lower).ratio()
    lcs        = lcs_ratio(t1_lower, t2_lower)
    len1, len2 = len(texto1), len(texto2)
    len_sim    = 1.0 - abs(len1 - len2) / max(len1, len2, 1)
    prefix_sim = SequenceMatcher(None, t1_lower[:3], t2_lower[:3]).ratio()
    suffix_sim = SequenceMatcher(None, t1_lower[-3:], t2_lower[-3:]).ratio()

    # F8–F10: lógica diferenciadora (tokens RAW, no normalizados)
    tiene_token_identidad = 1.0 if (interseccion_raw & TOKENS_IDENTIDAD) else 0.0

    exclusivos1 = tokens1 - tokens2
    exclusivos2 = tokens2 - tokens1
    dif1 = exclusivos1 & TOKENS_DIFERENCIADORES
    dif2 = exclusivos2 & TOKENS_DIFERENCIADORES
    if dif1 and dif2:
        penalizacion = 1.0
    elif dif1 or dif2:
        penalizacion = 0.5
    else:
        penalizacion = 0.0

    total_dif    = len(dif1) + len(dif2)
    total_tokens = max(len(tokens1) + len(tokens2), 1)
    ratio_dif    = min(total_dif / total_tokens, 1.0)

    # F11: match semántico de grupo
    match_semantico = mismo_grupo_semantico(tokens1, tokens2)

    features = [
        token_jaccard,            # F1
        tokens_compartidos_ratio, # F2
        seq_sim,                  # F3
        lcs,                      # F4
        len_sim,                  # F5
        prefix_sim,               # F6
        suffix_sim,               # F7
        tiene_token_identidad,    # F8
        penalizacion,             # F9
        ratio_dif,                # F10
        match_semantico,          # F11
    ]

    # F12: embedding cosine similarity
    if use_embeddings:
        features.append(embedding_sim(texto1, texto2))

    return features

# ─────────────────────────────────────────────────────────────────────────────
# PREPARACIÓN DE DATOS
# ─────────────────────────────────────────────────────────────────────────────

def preparar_datos(training_data, use_embeddings: bool = True):
    """
    Convierte la lista de (texto1, texto2, etiqueta) en arrays numpy.
    use_embeddings controla si se incluye F12.
    """
    X, y = [], []
    total = len(training_data)
    for i, (texto1, texto2, etiqueta) in enumerate(training_data):
        if i % 50 == 0:
            logger.info(f"  Extrayendo features {i}/{total}…")
        X.append(extraer_caracteristicas(texto1, texto2, use_embeddings=use_embeddings))
        y.append(int(etiqueta))
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

# ─────────────────────────────────────────────────────────────────────────────
# ENTRENAMIENTO
# ─────────────────────────────────────────────────────────────────────────────

def crear_directorio_si_no_existe(ruta):
    Path(ruta).parent.mkdir(parents=True, exist_ok=True)


def model_validator(X_train, y_train, X_val, y_val):
    """
    Entrena la red neuronal sobre el vector de features escalado.
    input_dim se infiere automáticamente → funciona con 11 o 12 features.
    """
    input_dim = X_train.shape[1]
    logger.info(f"input_dim = {input_dim}  ({'con' if input_dim == 12 else 'sin'} embeddings)")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled   = scaler.transform(X_val)

    modelo = Neuron.dense_classifier(input_dim=input_dim)

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=50,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=10,
            min_lr=1e-5,
            verbose=1,
        ),
    ]

    crear_directorio_si_no_existe(SCALER_PATH)
    joblib.dump(scaler, SCALER_PATH)
    logger.info(f"Scaler guardado en: {SCALER_PATH.resolve()}")

    class_weights     = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(enumerate(class_weights))
    logger.info(f"⚖Class weights: {class_weight_dict}")

    historia = modelo.fit(
        X_train_scaled, y_train,
        validation_data=(X_val_scaled, y_val),
        class_weight=class_weight_dict,
        epochs=100,
        batch_size=16,
        callbacks=callbacks,
        verbose=1,
    )
    modelo.save(MODEL_PATH)
    logger.info(f"Modelo guardado en: {MODEL_PATH.resolve()}")
    return modelo, historia, scaler


# ─────────────────────────────────────────────────────────────────────────────
# SEMANTIC MODEL SUPPORT: TOKENIZACIÓN PARA SECUENCIAS
# ─────────────────────────────────────────────────────────────────────────────

class SimpleTokenizer:
    """
    Tokenizador simple que convierte texto a índices numéricos.
    Útil para el semantic_model que requiere secuencias de números.
    """

    def __init__(self, vocab_size=10000, max_len=20):
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.word2idx = {}
        self.idx2word = {}
        self._build_vocab_from_config()

    def _build_vocab_from_config(self):
        """Construye vocabulario a partir de tokens conocidos (orden determinístico)."""
        idx = 1  # 0 es para padding

        # Agregar tokens diferenciadores (ordenados para determinismo)
        for token in sorted(TOKENS_DIFERENCIADORES)[:100]:
            if idx < self.vocab_size:
                self.word2idx[token] = idx
                self.idx2word[idx] = token
                idx += 1

        # Agregar tokens de identidad (ordenados para determinismo)
        for token in sorted(TOKENS_IDENTIDAD)[:100]:
            if idx < self.vocab_size:
                self.word2idx[token] = idx
                self.idx2word[idx] = token
                idx += 1

        logger.debug(f"Vocabulario inicializado con {idx - 1} tokens conocidos")

    def _deterministic_hash(self, token: str) -> int:
        """Hash determinístico (no depende de PYTHONHASHSEED)."""
        h = 0
        for c in token:
            h = (h * 31 + ord(c)) & 0xFFFFFFFF
        return h

    def tokenize_to_indices(self, texto: str) -> list[int]:
        """
        Convierte texto a lista de índices (determinístico entre sesiones).
        Tokens desconocidos se asignan a índices basados en hash determinístico.
        """
        tokens = tokenizar(texto)
        indices = []
        vocab_range = self.vocab_size - len(self.word2idx)

        for token in sorted(tokens)[:self.max_len]:
            if token in self.word2idx:
                indices.append(self.word2idx[token])
            else:
                idx = (self._deterministic_hash(token) % vocab_range) + len(self.word2idx)
                indices.append(idx)

        while len(indices) < self.max_len:
            indices.append(0)

        return indices[:self.max_len]


# ─────────────────────────────────────────────────────────────────────────────
# SEMANTIC MODEL SUPPORT: PREPARACIÓN DE DATOS
# ─────────────────────────────────────────────────────────────────────────────

def preparar_datos_semantico(training_data, use_tokenizer=True):
    """
    Prepara datos para el semantic_model.

    Args:
        training_data: lista de tuplas (campo1, campo2, etiqueta)
        use_tokenizer: si True, convierte texto a índices numéricos

    Returns:
        dict con 7 arrays (inputs del semantic_model) y array de targets
    """
    from src.training.semantic_features_extractor import (
        extraer_features_semanticos,
        create_default_context,
        preparar_par_campos
    )

    tokenizer = SimpleTokenizer(vocab_size=10000, max_len=20) if use_tokenizer else None

    # Guardar tokenizer state para inferencia reproducible
    if tokenizer is not None:
        import json
        with open(TOKENIZER_PATH, 'w', encoding='utf-8') as f:
            json.dump({
                'word2idx': tokenizer.word2idx,
                'vocab_size': tokenizer.vocab_size,
                'max_len': tokenizer.max_len,
            }, f, ensure_ascii=False)
        logger.info(f"✅ Tokenizer guardado en: {TOKENIZER_PATH.resolve()}")

    src_names_seq = []
    src_values_seq = []
    tgt_names_seq = []
    tgt_values_seq = []
    src_semantics = []
    tgt_semantics = []
    contexts = []
    targets = []

    total = len(training_data)
    for i, (campo1, campo2, etiqueta) in enumerate(training_data):
        if i % 50 == 0:
            logger.info(f"  Preparando datos semánticos {i}/{total}…")

        # Extraer contextos por defecto
        ctx = create_default_context()

        # Preparar par de campos
        pair_data = preparar_par_campos(
            src_name=campo1,
            src_value=campo1,  # En este caso, nombre y valor son iguales
            tgt_name=campo2,
            tgt_value=campo2,
            src_context=ctx,
            tgt_context=ctx
        )

        # Convertir a índices si se usa tokenizer
        if use_tokenizer:
            src_name_indices = tokenizer.tokenize_to_indices(campo1)
            src_value_indices = tokenizer.tokenize_to_indices(campo1)
            tgt_name_indices = tokenizer.tokenize_to_indices(campo2)
            tgt_value_indices = tokenizer.tokenize_to_indices(campo2)
        else:
            src_name_indices = np.zeros(20, dtype=np.int32)
            src_value_indices = np.zeros(20, dtype=np.int32)
            tgt_name_indices = np.zeros(20, dtype=np.int32)
            tgt_value_indices = np.zeros(20, dtype=np.int32)

        src_names_seq.append(src_name_indices)
        src_values_seq.append(src_value_indices)
        tgt_names_seq.append(tgt_name_indices)
        tgt_values_seq.append(tgt_value_indices)
        src_semantics.append(pair_data['src_semantic'])
        tgt_semantics.append(pair_data['tgt_semantic'])
        contexts.append(pair_data['context'])
        targets.append(int(etiqueta))

    # Convertir a arrays numpy
    X = {
        'src_name': np.array(src_names_seq, dtype=np.int32),
        'src_value': np.array(src_values_seq, dtype=np.int32),
        'tgt_name': np.array(tgt_names_seq, dtype=np.int32),
        'tgt_value': np.array(tgt_values_seq, dtype=np.int32),
        'src_semantic': np.array(src_semantics, dtype=np.float32),
        'tgt_semantic': np.array(tgt_semantics, dtype=np.float32),
        'context': np.array(contexts, dtype=np.float32),
    }

    y = np.array(targets, dtype=np.float32)

    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# SEMANTIC MODEL SUPPORT: ENTRENAMIENTO
# ─────────────────────────────────────────────────────────────────────────────

def model_validator_semantic(X_train, y_train, X_val, y_val):
    """
    Entrena el semantic_model sobre datos estructurados.

    Args:
        X_train: dict con 7 arrays de entrenamiento
        y_train: array de etiquetas (0 o 1)
        X_val: dict con 7 arrays de validación
        y_val: array de etiquetas de validación

    Returns:
        (modelo, history, None)  # sin scaler porque inputs ya están normalizados
    """
    modelo = Neuron.semantic_model(
        vocab_size=10000,
        embed_dim=64,
        max_len=20,
        semantic_features=9
    )

    logger.info("Semantic Model Summary:")
    modelo.summary()

    # Preparar targets para el modelo
    # El semantic_model tiene 3 outputs: similarity_score, match_type, confidence

    # Convertir etiquetas binarias a targets multi-output
    y_similarity = y_train.reshape(-1, 1)  # [0, 1]
    y_match_type = y_train.astype(np.int32)  # Simplificar: 0=no_match, 1=exacto
    # Confianza: 0.85 para matches (incertidumbre moderada), 0.95 para no-matches
    y_confidence = np.where(y_train == 1, 0.85, 0.95).reshape(-1, 1)

    y_train_dict = {
        'similarity_score': y_similarity,
        'match_type': y_match_type,
        'confidence': y_confidence
    }

    # Validación
    y_val_similarity = y_val.reshape(-1, 1)
    y_val_match_type = y_val.astype(np.int32)
    y_val_confidence = np.where(y_val == 1, 0.85, 0.95).reshape(-1, 1)

    y_val_dict = {
        'similarity_score': y_val_similarity,
        'match_type': y_val_match_type,
        'confidence': y_val_confidence
    }

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=15,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-5,
            verbose=1,
        ),
    ]

    # Guardar scaler (aunque no se usa en semantic_model)
    crear_directorio_si_no_existe(SCALER_PATH)
    joblib.dump(None, SCALER_PATH)
    logger.info(f"✅ Scaler path preparado: {SCALER_PATH.resolve()}")

    # Nota: No se puede usar class_weight con multi-output models en Keras
    # El modelo tiene 3 outputs, por lo que el balanceo de clases se maneja en loss functions
    logger.info("⚠️  Multi-output model detectado. class_weight no soportado - usando loss functions predefinidas")

    history = modelo.fit(
        X_train,
        y_train_dict,
        validation_data=(X_val, y_val_dict),
        epochs=50,
        batch_size=16,
        callbacks=callbacks,
        verbose=1,
    )

    modelo.save(MODEL_PATH)
    logger.info(f"✅ Semantic model guardado en: {MODEL_PATH.resolve()}")

    return modelo, history, None


# ─────────────────────────────────────────────────────────────────────────────
# DIAGNÓSTICO  (python cross_validator.py)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    casos = [
        ("customer_email", "vendor_email", 0),
        ("shipping_address", "billing_address", 0),
        ("precio_venta", "precio_costo", 0),
        ("work_phone", "home_phone", 0),
        ("primary_email", "secondary_email", 0),
        ("apellido_paterno", "apellido_materno", 0),
        ("fecha_creacion", "fecha_modificacion", 0),
        ("folio_factura", "folio_pedido", 0),
        ("stock_minimo", "stock_maximo", 0),
        (None, None, None),
        ("custbody_mx_cfdi_uuid", "PFR_UUID_TB120", 1),
        ("rfc", "tax_id", 1),
        ("email", "correo_electronico", 1),
        ("KUNNR", "customer_number", 1),
        ("cp", "codigo_postal", 1),
        ("nombre", "first_name", 1),
    ]

    print(f"\n{'Campo1':30} {'Campo2':28} {'F9':5} {'F10':5} {'F12(emb)':9} {'Esp'}")
    print("─" * 88)

    for c1, c2, esp in casos:
        if esp is None:
            print()
            continue
        f = extraer_caracteristicas(c1, c2, use_embeddings=True)
        f9 = f[8]
        f10 = f[9]
        f12 = f[11] if len(f) > 11 else 0.0

        tokens1 = tokenizar(c1)
        tokens2 = tokenizar(c2)
        dif1 = sorted((tokens1 - tokens2) & TOKENS_DIFERENCIADORES)
        dif2 = sorted((tokens2 - tokens1) & TOKENS_DIFERENCIADORES)

        correcto = (esp == 0 and f9 >= 0.5) or (esp == 1 and f9 < 0.5)
        marca = "✅" if correcto else "❌"
        print(
            f"{marca} {c1:30} {c2:28} {f9:.1f}  {f10:.2f}  {f12:.3f}     {esp}"
            f"  dif:[{dif1}|{dif2}]"
        )