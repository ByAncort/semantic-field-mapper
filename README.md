Proyecto: Semantic Field Mapper
=================================

Resumen
-------
Semantic Field Mapper es una librería y pipeline para identificar correspondencias semánticas entre nombres de campos (por ejemplo: customer_id ↔ id_cliente) y para comparar estructuras JSON completas entre APIs. Combina tres señales en un ensemble:

- Una red neuronal entrenada sobre features clásicos (token/jaccard, sequence-sim, penalizaciones, etc.).
- Similitud de embeddings (sentence-transformers) para capturar correspondencia cross-idioma y semántica.
- Reglas determinísticas construidas sobre features semánticos y tokenización.

El proyecto incluye además un "semantic_model" avanzado que incorpora features semánticos extraídos desde nombres de campo y contexto JSON (profundidad, tipo de padre, posición, etc.).

Estructura del repositorio
--------------------------
- src/
  - config.py                    Configuración de rutas y parámetros globales
  - arq/
    - neuron.py                  Componentes de red neuronal personalizados
  - db/
    - config_mongodb.py          Acceso a tokens, stopwords y configuración (MongoDB-like)
    - run_seed.py                Scripts de inicialización / seed
  - data_training/
    - gen_datos.py               Generadores / utilidades para crear datasets de entrenamiento
    - datos_entrenamiento.py     Helpers para manejo de datasets
  - training/
    - cross_validator.py         Pipeline clásico: preparar datos, validar, guardar modelo y scaler
    - ensemble_predictor.py      Implementación del ensemble (NN + Emb + Rules)
    - semantic_features_extractor.py  Extracción de 8 features semánticos y 5 de contexto JSON
    - train_model.py             Scripts de entrenamiento (classic y semantic)
    - models/                    Modelos y artefactos guardados (h5, scaler.pkl, imágenes)

Requisitos
----------
Instala las dependencias mínimas (recomendado usar virtualenv/venv):

```
python -m venv .venv
source .venv/bin/activate    # Linux / macOS
.venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

Nota: En este repo no se encuentra requirements.txt. Dependencias detectadas por lectura del código:
- numpy
- scikit-learn
- tensorflow
- joblib
- sentence-transformers (opcional para mejorar señal de embeddings)
- matplotlib (opcional para graficar entrenamiento)

Instalación rápida (sin requirements.txt):

```
pip install numpy scikit-learn tensorflow joblib
pip install sentence-transformers matplotlib  # opcional
```

Configuración
-------------
- Revisa `src/config.py` para configurar la ruta donde se guardan los modelos (MODELS_DIR, MODEL_PATH, SCALER_PATH).
- `src/db/config_mongodb.py` expone una clase SemanticConfigMongoDb que proporciona tokens diferenciadores, stopwords e índices semánticos. Asegúrate de tener la fuente de datos correcta (puede ser una capa local o una conexión Mongo configurada dentro de ese módulo).

Uso - Inferencia (ensemble)
---------------------------
Ejemplo mínimo para usar el ensemble (usa modelo y scaler guardados en src/training/models/...):

```python
from src.training.ensemble_predictor import EnsemblePredictor

predictor = EnsemblePredictor.from_paths()
res = predictor.predecir('customer_email', 'vendor_email')
print(res)

# Comparar dos objetos JSON completos
api_a = {'customer_id': 1, 'email': 'x@y.com'}
api_b = {'id_cliente': 1, 'correo': 'x@y.com'}
res = predictor.predecir_objetos(api_a, api_b, estrategia='top3')
print(res)
```

Uso - Entrenamiento
--------------------
Hay dos flujos principales:

1) Modelo clásico (features tradicionales + opcional embedding):

```python
from src.training.train_model import train_model
historia, modelo, scaler = train_model(use_embeddings=True)
```

2) Semantic model (recomendado): incorpora features semánticos y contexto JSON.

```python
from src.training.train_model import train_model_semantic
historia, modelo, _ = train_model_semantic()
```

El script `train_model.py` también incluye utilidades para graficar (`graficar_entrenamiento`) y evaluación post-entrenamiento (`evaluar_post_entrenamiento`).

Descripción técnica de módulos clave
-----------------------------------
- src/training/cross_validator.py
  - Responsabilidad: preparar features, crear datasets de entrenamiento, definir y validar la arquitectura de la red neuronal, guardar modelo y scaler.
  - Salidas: modelo (.h5) y scaler (scaler.pkl).

- src/training/semantic_features_extractor.py
  - Extrae 8 features semánticos por nombre de campo:
    0) token identidad (ej. id, rfc, email)
    1) token diferenciador (ej. primary, billing)
    2) id de grupo semántico (hash sobre grupo)
    3) fuerza/confianza del grupo
    4) longitud normalizada
    5) proporción de vocales
    6) ratio de caracteres especiales
    7) patrón de mayúsculas (snake, camel, pascal, upper)
  - Además define JSONContext con 5 features de contexto (profundidad, is_array, is_object, tipo de padre, posición relativa).

- src/training/ensemble_predictor.py
  - Implementa un EnsemblePredictor que combina: score_nn, score_emb, score_rules.
  - score_nn: usa el modelo guardado + scaler (o fallback de reglas si el modelo es la variante semantic_model con multiples outputs).
  - score_emb: similitud coseno usando sentence-transformers (si está disponible).
  - score_rules: calculado desde features (token_jaccard, seq_sim, match_semantico, penalizacion).
  - Funciones adicionales: predecir_objetos (compara dos objetos JSON completos) y evaluar_dataset (retorna métricas y casos difíciles).

Formato de datos y generación
----------------------------
- `src/data_training/gen_datos.py` contiene generadores de datasets con parejas (campo1, campo2, etiqueta). El pipeline de `train_model` llama a `generacion_datos` para obtener train/test.
- Las entradas al modelo clásico son vectores numéricos (features normalizadas). El semantic_model usa inputs más complejos (vectors tokenizados + features semánticas + contexto).

Buenas prácticas y notas
-----------------------
- Evitar commitear artefactos binarios de entrenamientos grandes. Actualmente en `src/training/models/v3.2.1` hay un `.h5` y `scaler.pkl` ya versionados; considera moverlos a un storage externo y añadirlos a .gitignore si crecen demasiado.
- Normalización de finales de línea: en Windows Git muestra advertencias LF → CRLF. Si el equipo es mixto, añadir un `.gitattributes` con `* text=auto` y/o configurar core.autocrlf según la política del equipo.
- Para reproducibilidad almacena los seeds del generador (random_state) y la versión de paquetes en un `requirements.txt` o `environment.yml`.

Contribuyendo
-------------
- Fork & Pull Request: crea ramas descriptivas (`feature/xxx`, `fix/xxx`).
- Añade tests unitarios para funcionalidades críticas (tokenización, extracción de features, predecir_objetos).
- Actualiza README si cambias la API pública.

Licencia y autor
----------------
Incluye la licencia del proyecto aquí (no proporcionada en el repo). Añade también autor y contacto si corresponde.

Contacto
--------
Para preguntas técnicas o integración: abrir un issue en GitHub o contactar al autor del repositorio.
