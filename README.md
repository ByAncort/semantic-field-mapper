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

Flujo técnico y técnicas utilizadas
----------------------------------
La arquitectura general combina generación de datos, extracción de features, entrenamiento de modelos y un ensemble para inferencia. A continuación se describe el flujo y las técnicas principales.

- Fuentes de datos: ejemplos etiquetados (campo_1, campo_2, etiqueta) generados por `src/data_training/gen_datos.py` o datasets reales cargados desde la capa DB (`src/db/config_mongodb.py`).
- Generación de dataset: normalización, particionado (train/val/test), y opcional data augmentation para hard negatives/positives.
- Tokenización: reglas deterministas que preservan tokens diferenciadores (reemplazo de separadores, separación camelCase/ dígitos) y eliminación de stopwords del sistema.
- Features clásicos: token_jaccard, sequence similarity, longitudes, counts, penalizaciones, match_semantico y (opcional) similitud de embeddings.
- Features semánticos: 8 features por nombre de campo (identidad, diferenciadores, grupo semántico, fuerza, longitud normalizada, vowel_ratio, special_chars, case pattern).
- Contexto JSON: 5 features (profundidad, is_array, is_object, tipo de padre, posición relativa).
- Modelos:
  - Classic NN: red entrenada sobre vector de features (scaler + keras model). Guardado como .h5 + scaler.pkl.
  - Semantic model: arquitectura siamesa que consume tokenizaciones + features semánticas + contexto. Puede incluir LSTM/Atención.
  - Embeddings: sentence-transformers para similitud coseno (opcional, mejora robustez cross-idioma).
- Ensemble: combina score_nn (modelo clásico o fallback), score_emb (embeddings) y score_rules (reglas determinísticas) con pesos configurables. Umbral configurable para decidir match/no-match.
- Evaluación: métricas estándar (accuracy, precision, recall, F1). Identificación de hard negatives/positives para mejorar dataset.

Diagrama (Mermaid)
-------------------
Aquí tienes un diagrama en formato Mermaid que representa el flujo de datos y componentes principales. Puedes renderizarlo en cualquier visor compatible con Mermaid (GitHub, VSCode Markdown Preview con extensión, etc.).

```mermaid
flowchart LR
  A[Fuentes de datos] --> B[Generación de datasets]
  B --> C[Extracción de features]
  C --> C1[Tokenización + Features clásicos]
  C --> C2[Features semánticos + Contexto JSON]
  C1 --> D[Entrenamiento: Classic NN (cross_validator)]
  C2 --> E[Entrenamiento: Semantic Model (Siamese LSTM + Attention)]
  D --> F[Modelo clásico (.h5) + scaler.pkl]
  E --> G[Modelo semántico (.h5)]
  C --> I[Sentence-transformers (embeddings)]
  F --> H[EnsemblePredictor]
  G --> H
  I --> H
  H --> J[Inferencia: predecir / predecir_objetos]
  J --> K[Outputs: score_final, desglose, tokens, métricas]
  D --> L[Evaluación: precision/recall/F1]
  E --> L
  L --> M[Identificar hard positives / hard negatives]

  style A fill:#f9f,stroke:#333,stroke-width:1px
  style H fill:#bbf,stroke:#333,stroke-width:1px
  style J fill:#bfb,stroke:#333,stroke-width:1px
```

Leyenda y notas
---------------
- Tokenización: se aplican reglas específicas para conservar tokens relevantes (underscore replacement, separación camelCase, separación letra-número) y se filtran stopwords.
- Scaler: indispensable para que la red neuronal reciba features normalizados; guardado con joblib.
- Fallbacks: cuando el modelo semantic (multisalida) no es compatible con el ensemble, el pipeline usa score_rules como fallback en el componente _score_nn.
- Ajustes de ensemble: los pesos por defecto son (weight_nn=0.45, weight_emb=0.30, weight_rules=0.25) y el umbral por defecto es 0.40; estos valores son configurables en EnsembleConfig.
