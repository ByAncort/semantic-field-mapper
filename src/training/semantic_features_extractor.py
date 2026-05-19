"""
semantic_features_extractor.py
──────────────────────────────
Extractores de features semánticos y de contexto JSON para el semantic_model.

Proporciona funciones para extraer:
  - 8 features semánticos por campo
  - 5 features de contexto JSON
"""

from __future__ import annotations

import logging
import re
import unicodedata
from typing import Tuple

import numpy as np

import src.db.config_mongodb as db_setup

logger = logging.getLogger(__name__)

# ── Config MongoDB ─────────────────────────────────────────────────────────────
config = db_setup.SemanticConfigMongoDb()

TOKENS_DIFERENCIADORES = config.get_tokens_diferenciadores()
TOKENS_IDENTIDAD       = config.get_tokens_identidad()
STOPWORDS_SISTEMA      = config.get_stopwords()
GRUPOS_SEMANTICOS      = config.get_grupos_semanticos()
INDICE_SEMANTICO       = config.get_indice_semantico()


# ─────────────────────────────────────────────────────────────────────────────
# TOKENIZACIÓN (reutilizado de cross_validator.py)
# ─────────────────────────────────────────────────────────────────────────────

def tokenizar(texto: str) -> set:
    """Tokeniza preservando tokens diferenciadores."""
    texto = texto.lower()
    texto = re.sub(r'[.\-/\\]', '_', texto)
    texto = re.sub(r'([a-z])([A-Z])', r'\1_\2', texto)
    texto = re.sub(r'([a-zA-Z])(\d)', r'\1_\2', texto)
    texto = re.sub(r'(\d)([a-zA-Z])', r'\1_\2', texto)
    tokens = re.split(r'[^a-z0-9]+', texto)
    return {t for t in tokens if t and t not in STOPWORDS_SISTEMA and len(t) >= 2}


def remove_accents(text: str) -> str:
    """Elimina acentos de un texto."""
    nfd = unicodedata.normalize('NFD', text)
    return ''.join(c for c in nfd if unicodedata.category(c) != 'Mn')


def detect_case_pattern(texto: str) -> int:
    """
    Detecta el patrón de mayúsculas:
    0: snake_case o lowercase
    1: camelCase
    2: PascalCase
    3: UPPERCASE
    """
    if '_' in texto:
        return 0
    if len(texto) == 1:
        return 3 if texto.isupper() else 0
    if texto.isupper():
        return 3
    if texto[0].isupper():
        return 2 if any(c.islower() for c in texto) else 3
    if any(c.isupper() for c in texto[1:]):
        return 1
    return 0


def semantic_group_to_id(group_name: str) -> float:
    """
    Convierte un nombre de grupo semántico (string) a un ID numérico.
    Usa hash para generar un ID consistente en el rango [0, 1000].

    Args:
        group_name: Nombre del grupo semántico (ej: 'email', 'date', 'identifier')

    Returns:
        float: ID numérico normalizado en el rango [0, 1000]
    """
    if not group_name:
        return 0.0
    # Usar hash para generar un ID consistente
    hash_value = hash(group_name) % 1000
    return float(hash_value)


# ─────────────────────────────────────────────────────────────────────────────
# EXTRACTION: SEMANTIC FEATURES (8 features)
# ─────────────────────────────────────────────────────────────────────────────

def extraer_features_semanticos(nombre_campo: str) -> np.ndarray:
    """
    Extrae 8 features semánticos de un nombre de campo.

    Returns:
        array de shape (8,) con los siguientes features:
        [0] has_identity_token: ¿Contiene token de identidad? [0, 1]
        [1] has_differentiator: ¿Contiene token diferenciador? [0, 1]
        [2] semantic_group_id: ID del grupo semántico [0, N]
        [3] semantic_group_strength: Confianza del grupo [0, 1]
        [4] text_length: Longitud normalizada [0, 1]
        [5] vowel_ratio: Proporción de vocales [0, 1]
        [6] special_chars_count: Caracteres especiales normalizados [0, 1]
        [7] case_pattern: Patrón de mayúsculas [0, 3]
    """
    tokens = tokenizar(nombre_campo)

    # F0: ¿Tiene token de identidad?
    has_identity = 1.0 if (tokens & TOKENS_IDENTIDAD) else 0.0

    # F1: ¿Tiene token diferenciador?
    has_differentiator = 1.0 if (tokens & TOKENS_DIFERENCIADORES) else 0.0

    # F2 y F3: Grupo semántico
    semantic_groups = {INDICE_SEMANTICO.get(t) for t in tokens if t in INDICE_SEMANTICO}
    if semantic_groups and None not in semantic_groups:
        # Tomar el primer grupo encontrado y convertir a ID numérico
        group_name = next(iter(semantic_groups))  # Tomar el primer grupo
        # group_id = semantic_group_to_id(group_name)
        GROUP_MAP = {
            "email": 1,
            "customer_id": 2,
            "postal_code": 3,
            "name": 4,
            "date": 5
        }

        group_id = float(GROUP_MAP.get(group_name, 0))
        group_strength = 0.8  # Confianza por defecto
        # cuando hay grupo
    else:
        group_id = 0.0
        group_strength = 0.0

    # F4: Longitud normalizada (logarítmica, capped at 50)
    normalized_length = min(len(nombre_campo) / 50.0, 1.0)

    # F5: Proporción de vocales
    vocales = sum(1 for c in nombre_campo.lower() if c in 'aeiouáéíóú')
    vowel_ratio = vocales / max(len(nombre_campo), 1)

    # F6: Caracteres especiales normalizados
    special_chars = sum(1 for c in nombre_campo if not c.isalnum() and c != '_')
    special_ratio = min(special_chars / max(len(nombre_campo), 1), 1.0)

    # F7: Patrón de mayúsculas
    case_pattern = float(detect_case_pattern(nombre_campo))

    shared_tokens = len(tokens & TOKENS_IDENTIDAD)
    shared_ratio = shared_tokens / max(len(tokens), 1)

    return np.array([
        has_identity,
        has_differentiator,
        float(group_id),
        group_strength,
        normalized_length,
        vowel_ratio,
        special_ratio,
        case_pattern,
        shared_ratio
    ], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# EXTRACTION: CONTEXT FEATURES (5 features)
# ─────────────────────────────────────────────────────────────────────────────
class JSONContext:
    """
    Información de contexto de un campo en la estructura JSON.
    """
    def __init__(
        self,
        depth: int = 0,
        is_array: bool = False,
        is_object: bool = False,
        parent_type: str = "root",
        position_in_parent: int = 0,
        total_siblings: int = 1
    ):
        self.depth = depth
        self.is_array = is_array
        self.is_object = is_object
        self.parent_type = parent_type  # root, response, data, nested, otros
        self.position_in_parent = position_in_parent
        self.total_siblings = total_siblings

    def to_features(self) -> np.ndarray:
        """
        Convierte el contexto a array de 5 features.

        Returns:
            array de shape (5,):
            [0] depth: Profundidad normalizada [0, 1] (capped at 10)
            [1] is_array: En un array? [0, 1]
            [2] is_object: En un objeto? [0, 1]
            [3] parent_type: Tipo de padre [0, 4]
            [4] position_ratio: Posición relativa [0, 1]
        """
        parent_type_map = {
            "root": 0,
            "response": 1,
            "data": 2,
            "nested": 3,
        }
        parent_id = float(parent_type_map.get(self.parent_type, 4))

        # Profundidad normalizada
        normalized_depth = min(self.depth / 10.0, 1.0)

        # Posición relativa
        position_ratio = (
            self.position_in_parent / max(self.total_siblings - 1, 1)
            if self.total_siblings > 1
            else 0.5
        )

        return np.array([
            normalized_depth,
            1.0 if self.is_array else 0.0,
            1.0 if self.is_object else 0.0,
            parent_id,
            position_ratio
        ], dtype=np.float32)


def create_default_context() -> JSONContext:
    """Crea un contexto JSON por defecto (sin información adicional)."""
    return JSONContext(
        depth=0,
        is_array=False,
        is_object=True,
        parent_type="data",
        position_in_parent=0,
        total_siblings=1
    )


# ─────────────────────────────────────────────────────────────────────────────
# COMBINED: PREPARAR PAR DE CAMPOS PARA EL MODELO
# ─────────────────────────────────────────────────────────────────────────────

def preparar_par_campos(
    src_name: str,
    src_value: str,
    tgt_name: str,
    tgt_value: str,
    src_context: JSONContext | None = None,
    tgt_context: JSONContext | None = None,
    tokenizer_fn=None
) -> dict:
    """
    Prepara un par de campos para pasar al semantic_model.

    Args:
        src_name: Nombre del campo origen
        src_value: Valor del campo origen (como string)
        tgt_name: Nombre del campo destino
        tgt_value: Valor del campo destino (como string)
        src_context: Contexto JSON del campo origen
        tgt_context: Contexto JSON del campo destino
        tokenizer_fn: Función de tokenización (si None, usa default)

    Returns:
        dict con keys:
        {
            'src_name': array tokenizado,
            'src_value': array tokenizado,
            'tgt_name': array tokenizado,
            'tgt_value': array tokenizado,
            'src_semantic': array de 8 features,
            'tgt_semantic': array de 8 features,
            'context': array de 5 features
        }
    """
    # Usar contextos por defecto si no se proporcionan
    if src_context is None:
        src_context = create_default_context()
    if tgt_context is None:
        tgt_context = create_default_context()

    # Features semánticos
    src_semantic = extraer_features_semanticos(src_name)
    tgt_semantic = extraer_features_semanticos(tgt_name)

    # Features de contexto (usar contexto origen; en futuro podría combinarse)
    context_features = src_context.to_features()

    # Para tokenización de texto, aquí necesitarías una función que convierta
    # texto a índices numéricos. Por ahora, retornamos placeholder.
    # En la integración real, esto usará el tokenizer del modelo.

    return {
        'src_name': src_name,
        'src_value': src_value,
        'tgt_name': tgt_name,
        'tgt_value': tgt_value,
        'src_semantic': src_semantic,
        'tgt_semantic': tgt_semantic,
        'context': context_features,
        'src_context_obj': src_context,
        'tgt_context_obj': tgt_context,
    }


# ─────────────────────────────────────────────────────────────────────────────
# DIAGNÓSTICO
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    casos = [
        "rfc",
        "email",
        "nombre_completo",
        "fecha_nacimiento",
        "shipping_address",
        "customer_id",
        "CURP_MEX",
        "uuid_primary",
    ]

    print("\n📊 FEATURES SEMÁNTICOS EXTRAÍDOS\n")
    print(f"{'Campo':25} {'Identity':10} {'Differ':10} {'Group':8} {'Length':8} {'Vowels':8} {'Special':8} {'Case':6}")
    print("─" * 93)

    for campo in casos:
        features = extraer_features_semanticos(campo)
        print(
            f"{campo:25} {features[0]:.1f}       {features[1]:.1f}       "
            f"{features[2]:.1f}      {features[4]:.2f}     {features[5]:.2f}     "
            f"{features[6]:.2f}     {features[7]:.0f}"
        )

    print("\n📍 CONTEXTO JSON (ejemplo)\n")
    ctx = JSONContext(depth=2, is_object=True, parent_type="data", position_in_parent=3, total_siblings=10)
    ctx_features = ctx.to_features()
    print(f"Contexto: {ctx.__dict__}")
    print(f"Features: {ctx_features}")
