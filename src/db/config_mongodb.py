# config_mongodb.py
import hashlib
import json

from pymongo import MongoClient, ASCENDING
from pymongo.errors import DuplicateKeyError
from typing import Set, Dict, Optional, List
from datetime import datetime
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SemanticConfigMongoDb:
    # ── Fallback data when MongoDB is not available ────────────────────────
    _FALLBACK_TOKENS_DIFERENCIADORES = frozenset({
        'customer', 'cliente', 'vendor', 'proveedor', 'employee', 'empleado',
        'supplier', 'suministrador', 'home', 'casa', 'work', 'trabajo',
        'office', 'oficina', 'venta', 'sale', 'costo', 'cost', 'compra', 'purchase',
    })

    _FALLBACK_TOKENS_IDENTIDAD = frozenset({
        'uuid', 'cfdi', 'rfc', 'curp', 'clabe', 'iban', 'swift',
        'sku', 'upc', 'ean', 'isbn', 'sat', 'timbre', 'id',
    })

    _FALLBACK_STOPWORDS = frozenset({
        'cust', 'custbody', 'custentity', 'custrecord', 'custcol',
        'mx', 'tb', 'pfr', 'field', 'sys', 'db', 'tbl',
    })

    _FALLBACK_GRUPOS = {
        "email": frozenset({"correo", "mail", "email", "correo_electronico",
                            "e-mail", "text_correo_250", "email_address", "user_email"}),
        "telefono": frozenset({"phone", "tel", "fono", "telefono", "movil", "mobile", "celular"}),
        "nombre": frozenset({"name", "nombre", "firstname", "given_name", "fullname", "first_name", "last_name"}),
        "apellido": frozenset({"lastname", "surname", "apellido", "apellido_paterno", "apellido_materno"}),
        "rfc": frozenset({"rfc", "tax_id", "taxid", "fiscal_id"}),
        "customer_id": frozenset({"customer_id", "id_cliente", "cliente_id", "customerid",
                                  "idcustomer", "client_id", "id_client", "customer", "cliente"}),
        "postal_code": frozenset({"cp", "codigo_postal", "postal_code", "zip", "zipcode",
                                  "zip_code", "cod_postal", "postal"}),
        "date": frozenset({"date", "fecha", "fecha_nacimiento", "fecha_creacion",
                           "fecha_modificacion", "fecha_alta", "fecha_emision"}),
        "name": frozenset({"name", "nombre", "nombre_completo", "full_name",
                           "first_name", "last_name", "apellido", "nombres"}),
    }

    def __init__(self, connection_string: str = "mongodb://localhost:27017/", db_name: str = "semantic_model"):
        self._connected = False
        self._connection_string = connection_string
        self._db_name = db_name

        # Cache para mejorar rendimiento
        self._cache_diferenciadores = None
        self._cache_identidad = None
        self._cache_stopwords = None
        self._cache_grupos = None
        self._cache_indice_semantico = None

        try:
            self.client = MongoClient(connection_string, serverSelectionTimeoutMS=3000)
            self.client.admin.command('ping')
            self._connected = True
            self.db = self.client[db_name]

            self.tokens_diferenciadores = self.db["tokens_diferenciadores"]
            self.tokens_identidad = self.db["tokens_identidad"]
            self.stopwords = self.db["stopwords"]
            self.grupos_semanticos = self.db["grupos_semanticos"]
            self.cambios_log = self.db["cambios_log"]

            self._crear_indices()

            self.training_datasets = self.db["training_datasets"]
            self.training_pairs = self.db["training_pairs"]
            self._crear_indices_entrenamiento()

            logger.info(f"Conectado a MongoDB: {db_name}")
        except Exception as e:
            self._connected = False
            self.client = None
            logger.warning(f"MongoDB no disponible ({e}). Usando datos de respaldo embebidos.")

    def _crear_indices_entrenamiento(self):
        """Crear índices para las colecciones de entrenamiento"""
        # Índice para identificar datasets únicos por hash
        self.training_datasets.create_index("dataset_hash", unique=True)
        self.training_datasets.create_index([("fecha_creacion", -1)])
        self.training_datasets.create_index("version")

        # Índices para búsqueda de pares
        self.training_pairs.create_index([
            ("dataset_id", 1),
            ("field_a", 1),
            ("field_b", 1)
        ], unique=True)

        # Índice para búsqueda por etiqueta dentro de un dataset
        self.training_pairs.create_index([
            ("dataset_id", 1),
            ("match", 1)
        ])

        logger.info("Índices de entrenamiento creados")
    def _calcular_hash_dataset(self, datos_ent: list) -> str:
        """Calcular hash único para un dataset basado en su contenido"""
        # Convertir a string ordenado para hash consistente
        datos_str = json.dumps(sorted(datos_ent), sort_keys=True, default=str)
        return hashlib.sha256(datos_str.encode()).hexdigest()

    def guardar_dataset(self,
                        datos_ent: list,
                        nombre: str = None,
                        version: str = "1.0",
                        metadata: dict = None,
                        sobrescribir: bool = False) -> str:
        if not self._connected:
            logger.warning("MongoDB no disponible. Dataset no guardado.")
            return ""

        """
        Guardar un dataset de entrenamiento en MongoDB

        Args:
            datos_ent: Lista de tuplas (field_a, field_b, match)
            nombre: Nombre descriptivo del dataset
            version: Versión del dataset
            metadata: Metadatos adicionales (configuración, parámetros, etc.)
            sobrescribir: Si True, sobrescribe dataset existente con mismo hash

        Returns:
            dataset_id: ID del dataset guardado
        """
        # Calcular estadísticas
        total = len(datos_ent)
        positivos = sum(1 for d in datos_ent if d[2] == 1)
        negativos = total - positivos

        # Calcular hash único
        dataset_hash = self._calcular_hash_dataset(datos_ent)

        # Verificar si ya existe
        existente = self.training_datasets.find_one({"dataset_hash": dataset_hash})
        if existente and not sobrescribir:
            logger.info(f"Dataset ya existe con ID: {existente['_id']}")
            return str(existente['_id'])

        # Preparar metadatos
        metadata_default = {
            "total_pares": total,
            "positivos": positivos,
            "negativos": negativos,
            "balance_ratio": positivos / max(negativos, 1),
            "fecha_generacion": datetime.now(),
            "config": metadata or {}
        }

        # Crear documento del dataset
        dataset_doc = {
            "dataset_hash": dataset_hash,
            "nombre": nombre or f"dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "version": version,
            "metadata": metadata_default,
            "fecha_creacion": datetime.now(),
            "fecha_modificacion": datetime.now(),
            "activo": True
        }

        # Insertar o actualizar dataset
        if existente and sobrescribir:
            self.training_datasets.update_one(
                {"_id": existente["_id"]},
                {"$set": dataset_doc}
            )
            dataset_id = existente["_id"]

            # Eliminar pares antiguos
            self.training_pairs.delete_many({"dataset_id": dataset_id})
            logger.info(f"Dataset {dataset_id} actualizado (sobrescrito)")
        else:
            result = self.training_datasets.insert_one(dataset_doc)
            dataset_id = result.inserted_id
            logger.info(f"Nuevo dataset creado con ID: {dataset_id}")

        # Guardar pares individuales
        pares_docs = []
        for i, (field_a, field_b, match) in enumerate(datos_ent):
            par_doc = {
                "dataset_id": dataset_id,
                "indice": i,
                "field_a": field_a,
                "field_b": field_b,
                "match": int(match),
                "tokens_a": list(self.tokenizar_rapido(field_a)) if hasattr(self, 'tokenizar_rapido') else None,
                "tokens_b": list(self.tokenizar_rapido(field_b)) if hasattr(self, 'tokenizar_rapido') else None,
                "fecha_creacion": datetime.now()
            }
            pares_docs.append(par_doc)

            # Insertar en lotes para mejor rendimiento
            if len(pares_docs) >= 1000:
                self.training_pairs.insert_many(pares_docs)
                pares_docs = []

        if pares_docs:
            self.training_pairs.insert_many(pares_docs)

        logger.info(f"Dataset guardado: {total} pares ({positivos} positivos, {negativos} negativos)")

        # Registrar en log
        self._log_cambio(
            "training_datasets",
            "INSERT" if not existente else "UPDATE",
            f"dataset_{nombre}",
            {"total": total, "positivos": positivos, "negativos": negativos}
        )

        return str(dataset_id)

    # En tu clase SemanticConfigMongoDb, modifica cargar_dataset:

    def cargar_dataset(self,
                       dataset_id: str = None,
                       nombre: str = None,
                       version: str = None,
                       balancear: bool = False,
                       max_negativos_ratio: float = 2.0,
                       return_dicts: bool = False) -> list:
        """
        Cargar un dataset de entrenamiento desde MongoDB

        Args:
            return_dicts: Si True, devuelve lista de diccionarios, si False devuelve tuplas
        """
        if not self._connected:
            raise ConnectionError("MongoDB no está disponible")

        from bson.objectid import ObjectId

        filtro = {"activo": True}
        if dataset_id:
            filtro["_id"] = ObjectId(dataset_id)
        elif nombre:
            filtro["nombre"] = nombre
            if version:
                filtro["version"] = version
        else:
            filtro = {}

        dataset = self.training_datasets.find_one(filtro, sort=[("fecha_creacion", -1)])

        if not dataset:
            raise ValueError(f"No se encontró dataset con filtro: {filtro}")

        logger.info(f"Cargando dataset: {dataset['nombre']} v{dataset['version']}")

        cursor = self.training_pairs.find(
            {"dataset_id": dataset["_id"]},
            {"field_a": 1, "field_b": 1, "match": 1, "_id": 0}
        )

        if return_dicts:
            datos = list(cursor)
        else:
            datos = [(doc["field_a"], doc["field_b"], doc["match"]) for doc in cursor]

        if balancear and not return_dicts:
            datos = self._balancear_dataset(datos, max_negativos_ratio)
        elif balancear and return_dicts:
            datos_tuplas = [(d["field_a"], d["field_b"], d["match"]) for d in datos]
            datos_tuplas = self._balancear_dataset(datos_tuplas, max_negativos_ratio)
            datos = [{"field_a": d[0], "field_b": d[1], "match": d[2]} for d in datos_tuplas]

        return datos

    def _balancear_dataset(self, datos: list, max_negativos_ratio: float = 2.0) -> list:
        """Balancear dataset limitando negativos"""
        import random

        positivos = [d for d in datos if d[2] == 1]
        negativos = [d for d in datos if d[2] == 0]

        max_negativos = int(len(positivos) * max_negativos_ratio)

        if len(negativos) > max_negativos:
            random.shuffle(negativos)
            negativos = negativos[:max_negativos]
            logger.info(f"Dataset balanceado: {len(positivos)} positivos, {len(negativos)} negativos")

        return positivos + negativos

    def listar_datasets(self, activos_only: bool = True) -> list:
        """Listar todos los datasets disponibles"""
        filtro = {"activo": True} if activos_only else {}

        cursor = self.training_datasets.find(
            filtro,
            {
                "nombre": 1,
                "version": 1,
                "metadata": 1,
                "fecha_creacion": 1,
                "_id": 1
            }
        ).sort("fecha_creacion", -1)

        datasets = []
        for doc in cursor:
            datasets.append({
                "id": str(doc["_id"]),
                "nombre": doc["nombre"],
                "version": doc["version"],
                "total": doc["metadata"]["total_pares"],
                "positivos": doc["metadata"]["positivos"],
                "negativos": doc["metadata"]["negativos"],
                "fecha": doc["fecha_creacion"].strftime("%Y-%m-%d %H:%M")
            })

        return datasets

    def _crear_indices(self):
        self.tokens_diferenciadores.create_index("token", unique=True)
        self.tokens_identidad.create_index("token", unique=True)
        self.stopwords.create_index("token", unique=True)

        self.grupos_semanticos.create_index(
            [("grupo", ASCENDING), ("token", ASCENDING), ("idioma", ASCENDING)],
            unique=True
        )

        self.grupos_semanticos.create_index("token")
        logger.info("Índices creados correctamente")

    def tokenizar_rapido(self, texto):
        """Versión simple de tokenización para guardar tokens"""
        import re
        texto = texto.lower()
        texto = re.sub(r'[.\-/\\]', '_', texto)
        texto = re.sub(r'([a-z])([A-Z])', r'\1_\2', texto)
        tokens = re.split(r'[^a-z0-9]+', texto)
        return {t for t in tokens if t and len(t) >= 2}

    # ── NUEVOS MÉTODOS PARA ACCEDER A LOS DATOS ──────────────────────────────
    def exportar_dataset_json(self, dataset_id: str, filepath: str):
        """Exportar dataset a archivo JSON"""
        import json
        from bson.objectid import ObjectId

        dataset = self.training_datasets.find_one({"_id": ObjectId(dataset_id)})
        if not dataset:
            raise ValueError(f"Dataset {dataset_id} no encontrado")

        cursor = self.training_pairs.find(
            {"dataset_id": ObjectId(dataset_id)},
            {"field_a": 1, "field_b": 1, "match": 1, "_id": 0}
        )

        datos = list(cursor)

        output = {
            "metadata": {
                "nombre": dataset["nombre"],
                "version": dataset["version"],
                "fecha": dataset["fecha_creacion"].isoformat(),
                "total": len(datos),
                "positivos": sum(1 for d in datos if d["match"] == 1),
                "negativos": sum(1 for d in datos if d["match"] == 0)
            },
            "data": datos
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        logger.info(f"Dataset exportado a {filepath}")

    def get_tokens_diferenciadores(self, refresh_cache: bool = False) -> Set[str]:
        """Obtener todos los tokens diferenciadores activos"""
        if not self._connected:
            return set(self._FALLBACK_TOKENS_DIFERENCIADORES)
        if self._cache_diferenciadores is None or refresh_cache:
            cursor = self.tokens_diferenciadores.find(
                {"activo": True},
                {"token": 1, "_id": 0}
            )
            self._cache_diferenciadores = {doc["token"] for doc in cursor}
            logger.debug(f"Cache de diferenciadores actualizado: {len(self._cache_diferenciadores)} tokens")
        return self._cache_diferenciadores

    def get_tokens_identidad(self, refresh_cache: bool = False) -> Set[str]:
        """Obtener todos los tokens de identidad activos"""
        if not self._connected:
            return set(self._FALLBACK_TOKENS_IDENTIDAD)
        if self._cache_identidad is None or refresh_cache:
            cursor = self.tokens_identidad.find(
                {"activo": True},
                {"token": 1, "_id": 0}
            )
            self._cache_identidad = {doc["token"] for doc in cursor}
            logger.debug(f"Cache de identidad actualizado: {len(self._cache_identidad)} tokens")
        return self._cache_identidad

    def get_stopwords(self, refresh_cache: bool = False) -> Set[str]:
        """Obtener todas las stopwords activas"""
        if not self._connected:
            return set(self._FALLBACK_STOPWORDS)
        if self._cache_stopwords is None or refresh_cache:
            cursor = self.stopwords.find(
                {"activo": True},
                {"token": 1, "_id": 0}
            )
            self._cache_stopwords = {doc["token"] for doc in cursor}
            logger.debug(f"Cache de stopwords actualizado: {len(self._cache_stopwords)} tokens")
        return self._cache_stopwords

    def get_grupos_semanticos(self, refresh_cache: bool = False) -> Dict[str, Set[str]]:
        """Obtener todos los grupos semánticos"""
        if not self._connected:
            return {k: set(v) for k, v in self._FALLBACK_GRUPOS.items()}
        if self._cache_grupos is None or refresh_cache:
            pipeline = [
                {"$match": {"activo": True}},
                {"$group": {
                    "_id": "$grupo",
                    "tokens": {"$addToSet": "$token"}
                }}
            ]
            self._cache_grupos = {}
            for doc in self.grupos_semanticos.aggregate(pipeline):
                self._cache_grupos[doc["_id"]] = set(doc["tokens"])
            logger.debug(f"Cache de grupos actualizado: {len(self._cache_grupos)} grupos")
        return self._cache_grupos

    def get_indice_semantico(self, refresh_cache: bool = False) -> Dict[str, str]:
        """
        Obtener índice inverso: token -> grupo
        Útil para búsquedas rápidas de grupo semántico
        """
        if not self._connected:
            idx = {}
            for grupo, tokens in self._FALLBACK_GRUPOS.items():
                for token in tokens:
                    idx[token] = grupo
            return idx
        if self._cache_indice_semantico is None or refresh_cache:
            self._cache_indice_semantico = {}
            grupos = self.get_grupos_semanticos(refresh_cache)
            for grupo, tokens in grupos.items():
                for token in tokens:
                    self._cache_indice_semantico[token] = grupo
            logger.debug(f"Índice semántico actualizado: {len(self._cache_indice_semantico)} entradas")
        return self._cache_indice_semantico

    # ── MÉTODOS DE CONVENIENCIA (PROPIEDADES) ────────────────────────────────

    @property
    def TOKENS_DIFERENCIADORES(self) -> Set[str]:
        """Propiedad para acceso directo (mantiene compatibilidad)"""
        return self.get_tokens_diferenciadores()

    @property
    def TOKENS_IDENTIDAD(self) -> Set[str]:
        """Propiedad para acceso directo"""
        return self.get_tokens_identidad()

    @property
    def STOPWORDS(self) -> Set[str]:
        """Propiedad para acceso directo"""
        return self.get_stopwords()

    @property
    def GRUPOS_SEMANTICOS(self) -> Dict[str, Set[str]]:
        """Propiedad para acceso directo"""
        return self.get_grupos_semanticos()

    @property
    def INDICE_SEMANTICO(self) -> Dict[str, str]:
        """Propiedad para acceso directo"""
        return self.get_indice_semantico()

    # ── MÉTODOS DE REFRESCO ──────────────────────────────────────────────────

    def refresh_cache(self):
        """Refrescar todo el cache"""
        self._cache_diferenciadores = None
        self._cache_identidad = None
        self._cache_stopwords = None
        self._cache_grupos = None
        self._cache_indice_semantico = None
        if self._connected:
            self.get_tokens_diferenciadores()
            self.get_tokens_identidad()
            self.get_stopwords()
            self.get_grupos_semanticos()
            self.get_indice_semantico()
            logger.info("Cache refrescado completamente")

    # ── MÉTODOS ORIGINALES (sin cambios) ─────────────────────────────────────

    def add_token_diferenciador(self, token: str, categoria: str = None,
                                notas: str = None, activo: bool = True):
        if not self._connected:
            logger.warning(f"MongoDB no disponible. Token '{token}' no persistido.")
            return
        try:
            documento = {
                "token": token.lower(),
                "categoria": categoria,
                "activo": activo,
                "notas": notas,
                "fecha_alta": datetime.now(),
                "fecha_modificacion": datetime.now()
            }

            self.tokens_diferenciadores.insert_one(documento)
            self._log_cambio("tokens_diferenciadores", "INSERT", token)
            logger.info(f"Token '{token}' agregado")
            self._cache_diferenciadores = None

        except DuplicateKeyError:
            self.tokens_diferenciadores.update_one(
                {"token": token.lower()},
                {"$set": {
                    "categoria": categoria,
                    "activo": activo,
                    "notas": notas,
                    "fecha_modificacion": datetime.now()
                }}
            )
            logger.info(f"Token '{token}' actualizado")
            self._cache_diferenciadores = None

    def add_to_grupo_semantico(self, grupo: str, token: str,
                               idioma: str = "es", activo: bool = True):
        if not self._connected:
            logger.warning(f"MongoDB no disponible. Grupo '{grupo}'→'{token}' no persistido.")
            return
        try:
            documento = {
                "grupo": grupo,
                "token": token.lower(),
                "idioma": idioma,
                "activo": activo,
                "fecha_alta": datetime.now()
            }

            self.grupos_semanticos.insert_one(documento)
            self._log_cambio("grupos_semanticos", "INSERT", token)
            self._cache_grupos = None
            self._cache_indice_semantico = None

        except DuplicateKeyError:
            self.grupos_semanticos.update_one(
                {"grupo": grupo, "token": token.lower(), "idioma": idioma},
                {"$set": {"activo": activo}}
            )
            self._cache_grupos = None
            self._cache_indice_semantico = None

    def get_grupo_semantico(self, grupo: str, idioma: str = None) -> Set[str]:
        filtro = {"grupo": grupo, "activo": True}
        if idioma:
            filtro["idioma"] = idioma

        cursor = self.grupos_semanticos.find(filtro, {"token": 1})
        return {doc["token"] for doc in cursor}

    def get_all_grupos(self) -> Dict[str, Set[str]]:
        """Alias para get_grupos_semanticos"""
        return self.get_grupos_semanticos()

    def buscar_token(self, token: str) -> Dict:
        token_lower = token.lower()
        resultado = {
            "diferenciador": False,
            "identidad": False,
            "stopword": False,
            "grupos": []
        }

        if self.tokens_diferenciadores.find_one({"token": token_lower, "activo": True}):
            resultado["diferenciador"] = True

        if self.tokens_identidad.find_one({"token": token_lower, "activo": True}):
            resultado["identidad"] = True

        if self.stopwords.find_one({"token": token_lower, "activo": True}):
            resultado["stopword"] = True

        cursor = self.grupos_semanticos.find(
            {"token": token_lower, "activo": True},
            {"grupo": 1}
        )
        resultado["grupos"] = [doc["grupo"] for doc in cursor]

        return resultado

    def _log_cambio(self, coleccion: str, accion: str, token: str = None,
                    detalles: dict = None):
        if not self._connected:
            return
        self.cambios_log.insert_one({
            "coleccion": coleccion,
            "accion": accion,
            "token": token,
            "detalles": detalles,
            "fecha": datetime.now(),
            "usuario": "system"
        })

    def migrar_datos_iniciales(self):
        """Migrar configuración inicial"""
        TOKENS_DIFERENCIADORES = {
            'customer', 'cliente', 'vendor', 'proveedor',
            'employee', 'empleado', 'supplier', 'suministrador',
            'home', 'casa', 'work', 'trabajo', 'office', 'oficina',
            'venta', 'sale', 'costo', 'cost', 'compra', 'purchase',
        }

        TOKENS_IDENTIDAD = {
            'uuid', 'cfdi', 'rfc', 'curp', 'clabe', 'iban', 'swift',
            'sku', 'upc', 'ean', 'isbn', 'sat', 'timbre',
        }

        STOPWORDS_SISTEMA = {
            'cust', 'custbody', 'custentity', 'custrecord', 'custcol',
            'mx', 'tb', 'pfr', 'field', 'sys', 'db', 'tbl',
        }

        GRUPOS_SEMANTICOS = {
            "email": {"correo", "mail", "email", "correo_electronico"},
            "telefono": {"phone", "tel", "fono", "telefono", "movil", "mobile", "celular"},
            "nombre": {"name", "nombre", "firstname", "given_name", "fullname"},
            "apellido": {"lastname", "surname", "apellido"},
            "rfc": {"rfc", "tax_id", "taxid", "fiscal_id"},
        }

        # Migrar tokens diferenciadores
        for token in TOKENS_DIFERENCIADORES:
            self.add_token_diferenciador(token, "general")

        # Migrar tokens identidad
        for token in TOKENS_IDENTIDAD:
            try:
                self.tokens_identidad.insert_one({
                    "token": token.lower(),
                    "tipo": "identidad",
                    "activo": True,
                    "fecha_alta": datetime.now()
                })
            except DuplicateKeyError:
                pass

        # Migrar stopwords
        for token in STOPWORDS_SISTEMA:
            try:
                self.stopwords.insert_one({
                    "token": token.lower(),
                    "contexto": "sistema",
                    "activo": True,
                    "fecha_alta": datetime.now()
                })
            except DuplicateKeyError:
                pass

        # Migrar grupos semánticos
        for grupo, tokens in GRUPOS_SEMANTICOS.items():
            for token in tokens:
                self.add_to_grupo_semantico(grupo, token)

        logger.info("Migración inicial completada")
        self.refresh_cache()

    def close(self):
        if self._connected and self.client:
            self.client.close()


# config_mongodb.py - Agrega esto al final del archivo (después del if __name__ == "__main__":)

if __name__ == "__main__":
    # Conectar a MongoDB
    db = SemanticConfigMongoDb()

    print("=" * 60)
    print("🚀 CARGANDO GRUPOS SEMÁNTICOS DETECTADOS DEL LOG")
    print("=" * 60)

    # ============================================
    # GRUPOS SEMÁNTICOS COMPLETOS (basados en el log)
    # ============================================

    grupos_semanticos = {
        "customer_id": {
            "tokens": [
                "customer_id",
                "id_cliente",
                "cliente_id",
                "customerid",
                "idcustomer",
                "client_id",
                "id_client",
                "customer",
                "cliente"
            ],
            "idioma": "en"
        },

        "email": {
            "tokens": [
                "email",
                "correo",
                "mail",
                "e-mail",
                "text_correo_250",
                "correo_electronico",
                "email_address",
                "user_email"
            ],
            "idioma": "en"
        },

        "postal_code": {
            "tokens": [
                "cp",
                "codigo_postal",
                "postal_code",
                "zip",
                "zipcode",
                "zip_code",
                "cod_postal",
                "postal"
            ],
            "idioma": "en"
        }
    }

    # ============================================
    # EJECUTAR CARGA
    # ============================================

    total_grupos = 0
    total_tokens = 0

    print("\n📦 Cargando grupos semánticos...\n")

    for grupo, config in grupos_semanticos.items():
        tokens = config["tokens"]
        idioma = config.get("idioma", "es")

        for token in tokens:
            try:
                db.add_to_grupo_semantico(grupo, token, idioma=idioma, activo=True)
                print(f"  ✅ {grupo:20} → {token}")
                total_tokens += 1
            except Exception as e:
                print(f"  ❌ Error con {grupo}→{token}: {e}")

        total_grupos += 1
        print()

    db.close()