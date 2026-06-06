"""Tema 6 — Mapeo Semántico ML: extracción de features de nombres de campo.

Estas funciones son la base del matching semántico: convierten un nombre de
campo en tokens y en un vector numérico de features que alimenta al modelo.
Se prueban con los datos de respaldo embebidos (sin MongoDB).
"""
import numpy as np

from src.training.semantic_features_extractor import (
    JSONContext,
    create_default_context,
    detect_case_pattern,
    extraer_features_semanticos,
    remove_accents,
    semantic_group_to_id,
    tokenizar,
)


class TestTokenizar:
    def test_separa_snake_case(self):
        assert tokenizar("customer_id") == {"customer", "id"}

    def test_elimina_stopwords_de_sistema(self):
        # 'custbody' es stopword embebida → solo queda 'rfc'.
        assert tokenizar("custbody_rfc") == {"rfc"}

    def test_descarta_tokens_de_un_caracter(self):
        assert tokenizar("a_email") == {"email"}

    def test_normaliza_separadores(self):
        assert tokenizar("user.email") == {"user", "email"}
        assert tokenizar("user/email") == {"user", "email"}


class TestRemoveAccents:
    def test_quita_acentos_y_tildes(self):
        assert remove_accents("máñana áéíóú") == "manana aeiou"

    def test_texto_sin_acentos_intacto(self):
        assert remove_accents("customer_id") == "customer_id"


class TestDetectCasePattern:
    def test_snake_case_es_cero(self):
        assert detect_case_pattern("customer_id") == 0

    def test_camel_case_es_uno(self):
        assert detect_case_pattern("customerId") == 1

    def test_pascal_case_es_dos(self):
        assert detect_case_pattern("CustomerId") == 2

    def test_uppercase_es_tres(self):
        assert detect_case_pattern("RFC") == 3


class TestSemanticGroupToId:
    def test_grupo_vacio_es_cero(self):
        assert semantic_group_to_id("") == 0.0

    def test_mismo_grupo_mismo_id(self):
        assert semantic_group_to_id("email") == semantic_group_to_id("email")

    def test_id_en_rango(self):
        assert 0.0 <= semantic_group_to_id("identifier") <= 1000.0


class TestExtraerFeaturesSemanticos:
    def test_vector_de_nueve_features(self):
        features = extraer_features_semanticos("customer_id")
        assert features.shape == (9,)
        assert features.dtype == np.float32

    def test_token_de_identidad_activa_f0(self):
        # 'rfc' es token de identidad embebido.
        assert extraer_features_semanticos("rfc")[0] == 1.0

    def test_sin_token_identidad_f0_cero(self):
        assert extraer_features_semanticos("color")[0] == 0.0

    def test_case_pattern_coincide_con_detector(self):
        features = extraer_features_semanticos("customerId")
        assert features[7] == float(detect_case_pattern("customerId"))

    def test_longitud_normalizada_se_satura_en_uno(self):
        features = extraer_features_semanticos("x" * 100)
        assert features[4] == 1.0

    def test_vowel_ratio_en_rango(self):
        features = extraer_features_semanticos("aeiou")
        assert 0.0 <= features[5] <= 1.0
        assert features[5] == 1.0


class TestJSONContext:
    def test_to_features_shape_y_valores(self):
        ctx = JSONContext(depth=2, is_object=True, parent_type="data",
                          position_in_parent=3, total_siblings=10)
        features = ctx.to_features()

        assert features.shape == (5,)
        assert features[0] == np.float32(0.2)   # depth 2/10
        assert features[1] == 0.0               # no es array
        assert features[2] == 1.0               # es objeto
        assert features[3] == 2.0               # parent_type 'data'
        assert abs(features[4] - (3 / 9)) < 1e-6  # posición relativa

    def test_parent_type_desconocido_es_cuatro(self):
        ctx = JSONContext(parent_type="loquesea")
        assert ctx.to_features()[3] == 4.0

    def test_un_solo_hermano_posicion_media(self):
        ctx = JSONContext(total_siblings=1)
        assert ctx.to_features()[4] == 0.5

    def test_depth_se_satura_en_uno(self):
        ctx = JSONContext(depth=50)
        assert ctx.to_features()[0] == 1.0

    def test_contexto_por_defecto(self):
        ctx = create_default_context()
        assert ctx.is_object is True
        assert ctx.parent_type == "data"
