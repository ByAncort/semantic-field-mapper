"""Configuración global de tests para semantic-field-mapper.

Por defecto se fuerza el modo "sin MongoDB" para que toda la lógica de
extracción de features use los datos de respaldo embebidos (rápido y
determinista). Los tests que necesiten MongoDB usan mongomock explícitamente.
"""
import pymongo


def _no_mongo(*args, **kwargs):
    raise Exception("MongoDB deshabilitado en tests (usar mongomock si se requiere)")


# Se parchea ANTES de que config_mongodb haga `from pymongo import MongoClient`,
# de modo que SemanticConfigMongoDb() caiga en el fallback embebido.
pymongo.MongoClient = _no_mongo
