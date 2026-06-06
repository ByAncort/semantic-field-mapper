"""Tema 6 — Mapeo Semántico ML: configuración de tokens/grupos en MongoDB.

Cubre las dos rutas de SemanticConfigMongoDb:
  - Sin MongoDB → datos de respaldo embebidos (robustez).
  - Con MongoDB (via mongomock) → lectura, caché y escritura de tokens/grupos.
"""
import mongomock
import pytest

import src.db.config_mongodb as cfg
from src.db.config_mongodb import SemanticConfigMongoDb


class TestFallbackSinMongo:
    """El conftest fuerza MongoClient a fallar → modo respaldo."""

    def test_no_conectado(self):
        config = SemanticConfigMongoDb()
        assert config._connected is False

    def test_tokens_identidad_de_respaldo(self):
        config = SemanticConfigMongoDb()
        assert "rfc" in config.get_tokens_identidad()
        assert "curp" in config.get_tokens_identidad()

    def test_stopwords_de_respaldo(self):
        config = SemanticConfigMongoDb()
        assert "cust" in config.get_stopwords()

    def test_indice_semantico_invierte_grupos(self):
        config = SemanticConfigMongoDb()
        indice = config.get_indice_semantico()
        # 'correo' pertenece al grupo 'email' en los datos de respaldo.
        assert indice.get("correo") == "email"

    def test_guardar_dataset_es_noop_sin_mongo(self):
        config = SemanticConfigMongoDb()
        assert config.guardar_dataset([("a", "b", 1)]) == ""


class TestConMongomock:
    """Ruta conectada usando mongomock en lugar de un MongoDB real."""

    @pytest.fixture
    def config(self, monkeypatch):
        monkeypatch.setattr(cfg, "MongoClient", mongomock.MongoClient)
        c = SemanticConfigMongoDb()
        assert c._connected is True
        return c

    def test_lee_tokens_identidad_desde_la_coleccion(self, config):
        config.tokens_identidad.insert_one({"token": "rut", "activo": True})

        tokens = config.get_tokens_identidad(refresh_cache=True)

        assert "rut" in tokens

    def test_token_inactivo_no_se_incluye(self, config):
        config.tokens_identidad.insert_one({"token": "viejo", "activo": False})

        assert "viejo" not in config.get_tokens_identidad(refresh_cache=True)

    def test_add_to_grupo_semantico_persiste_y_refleja_en_indice(self, config):
        config.add_to_grupo_semantico("email", "correo_corp")

        indice = config.get_indice_semantico(refresh_cache=True)
        assert indice.get("correo_corp") == "email"

    def test_buscar_token_clasifica(self, config):
        config.tokens_identidad.insert_one({"token": "rfc", "activo": True})
        config.add_to_grupo_semantico("rfc", "rfc")

        resultado = config.buscar_token("rfc")

        assert resultado["identidad"] is True
        assert "rfc" in resultado["grupos"]
