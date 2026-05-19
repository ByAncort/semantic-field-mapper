"""
train_semantic.py
─────────────────
Entrena el modelo semántico (LSTM + Attention + features semánticos)
y guarda el tokenizer para inferencia reproducible.

Uso:
    python train_semantic.py
"""

import logging, os, sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Crear directorio del modelo
from src.config import MODELS_DIR, TRAINING_CONFIG
MODELS_DIR.mkdir(parents=True, exist_ok=True)
logger.info(f"Modelo se guardará en: {MODELS_DIR.resolve()}")

# Importar pipeline semántico
from src.training.train_model import train_model_semantic, evaluar_post_entrenamiento, graficar_entrenamiento

logger.info("═" * 70)
logger.info("  ENTRENAMIENTO DEL MODELO SEMÁNTICO v3.2.5")
logger.info("═" * 70)

historia, modelo, _ = train_model_semantic()

evaluar_post_entrenamiento()
graficar_entrenamiento(historia)

logger.info("✅ Entrenamiento completado")
logger.info(f"Modelo: {MODELS_DIR.resolve()}")
