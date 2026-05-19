import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent

MODEL_NAME = "semantic_field_matcher_v1"

MODELS_DIR = BASE_DIR / 'src' / 'training' / 'models' / 'v3.2.5'
MODEL_PATH = MODELS_DIR / f"{MODEL_NAME}.h5"
SCALER_PATH = MODELS_DIR / 'scaler.pkl'
TOKENIZER_PATH = MODELS_DIR / 'tokenizer.json'


MODEL_CONFIG = {
    'version': 'v1_pro',
    'model_name': MODEL_NAME,
    'input_shape': (None, 10),  # ejemplo
    'threshold': 0.5,  # ejemplo para clasificación
}

TRAINING_CONFIG = {
    'epochs': 100,
    'batch_size': 32,
    'validation_split': 0.2,
    'learning_rate': 0.001,
}