"""
train_model.py
──────────────
Pipeline completo de entrenamiento.

Soporta dos modos:
  1. CLASSIC MODEL (default): Features clásicos (11-12 features)
  2. SEMANTIC MODEL (--semantic): Features semánticos + contexto JSON
"""
from __future__ import annotations
import logging

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

import src.config as config
import src.db.config_mongodb as config_mongodb
from src.training.cross_validator import model_validator, preparar_datos
from src.training.ensemble_predictor import EnsembleConfig, EnsemblePredictor
from src.data_training.gen_datos import generacion_datos

logger = logging.getLogger(__name__)
_db = config_mongodb.SemanticConfigMongoDb()

# Casos de referencia para evaluar el ensemble recién entrenado
_CASOS_REFERENCIA = [
    # Positivos
    ("rfc",                   "tax_id",                 1),
    ("email",                 "correo_electronico",     1),
    ("KUNNR",                 "customer_number",        1),
    ("custbody_mx_cfdi_uuid", "PFR_UUID_TB120",         1),
    ("cp",                    "codigo_postal",          1),
    ("nombre",                "first_name",             1),
    # Negativos
    ("customer_email",        "vendor_email",           0),
    ("shipping_address",      "billing_address",        0),
    ("precio_venta",          "precio_costo",           0),
    ("apellido_paterno",      "apellido_materno",       0),
    ("folio_factura",         "folio_pedido",           0),
    ("stock_minimo",          "stock_maximo",           0),
]

def train_model(use_embeddings: bool = True):
    """
    Entrena el modelo clásico de similitud semántica (compatibilidad hacia atrás).

    Args:
        use_embeddings: Si True, incluye F12 (embedding cosine sim) en el vector.
                        El modelo quedará guardado con input_dim=12.
                        Si False, input_dim=11 (comportamiento anterior).
    """
    logger.info(f"Generando datos de entrenamiento  (embeddings={'✅' if use_embeddings else '❌'})…")
    train_data, test_data = generacion_datos(test_size=0.2, random_state=42)
    # training_data = _db.cargar_dataset(nombre="dataset_alpha", version="1.1")

    logger.info(f"Total de ejemplos: {len(train_data)}")
    X, y = preparar_datos(train_data, use_embeddings=use_embeddings)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.5, random_state=42
    )
    logger.info(f"Train={len(X_train)}  Val={len(X_val)}  features={X.shape[1]}")

    modelo, historia, scaler = model_validator(X_train, y_train, X_val, y_val)

    return historia, modelo, scaler


def train_model_semantic():
    """
    Entrena el semantic_model (versión avanzada con features semánticos y contexto).

    Esta es la versión recomendada que incorpora:
    - Features semánticos (identidad, diferenciadores, grupos)
    - Contexto JSON (profundidad, tipo de objeto)
    - Arquitectura Siamesa mejorada con LSTM y atención
    """
    from src.training.cross_validator import preparar_datos_semantico, model_validator_semantic

    logger.info("🚀 Entrenando SEMANTIC MODEL (versión avanzada)…")
    logger.info("Generando datos de entrenamiento con features semánticos…")

    train_data, test_data = generacion_datos(test_size=0.2, random_state=42)
    logger.info(f"Total de ejemplos: {len(train_data)}")

    X, y = preparar_datos_semantico(train_data, use_tokenizer=True)

    # Mostrar shape de inputs
    logger.info(f"📊 Shape de inputs:")
    for key, val in X.items():
        logger.info(f"   {key:20s}: {val.shape}")
    logger.info(f"   targets            : {y.shape}")

    # Split
    from sklearn.model_selection import train_test_split

    # Dividir manualmente cada input
    indices = np.arange(len(y))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)

    X_train = {k: v[train_idx] for k, v in X.items()}
    X_val = {k: v[val_idx] for k, v in X.items()}
    y_train = y[train_idx]
    y_val = y[val_idx]

    logger.info(f"Train size: {len(train_idx)}  |  Val size: {len(val_idx)}")

    # Entrenar
    modelo, historia, _ = model_validator_semantic(X_train, y_train, X_val, y_val)

    return historia, modelo, None


def evaluar_post_entrenamiento():
    """
    Carga el ensemble desde disco y evalúa sobre los casos de referencia.
    Llamar después de train_model() para tener una métrica rápida del sistema completo.
    """
    logger.info("Evaluando ensemble post-entrenamiento…")
    predictor = EnsemblePredictor.from_paths()
    metricas  = predictor.evaluar_dataset(_CASOS_REFERENCIA)

    logger.info(
        f"Ensemble → Accuracy={metricas['accuracy']:.3f}  "
        f"Precision={metricas['precision']:.3f}  "
        f"Recall={metricas['recall']:.3f}  "
        f"F1={metricas['f1']:.3f}"
    )

    if metricas["hard_negatives"]:
        logger.warning(f"Hard negatives detectados ({len(metricas['hard_negatives'])}):")
        for c1, c2, s in metricas["hard_negatives"]:
            logger.warning(f"  {c1} ↔ {c2}  score={s:.3f}  → agregar como label=0 en training")

    if metricas["hard_positives"]:
        logger.warning(f"Hard positives detectados ({len(metricas['hard_positives'])}):")
        for c1, c2, s in metricas["hard_positives"]:
            logger.warning(f"  {c1} ↔ {c2}  score={s:.3f}  → agregar como label=1 en training")

    return metricas

def graficar_entrenamiento(historia, guardar_imagen: bool = True):
    """
    Grafica loss, accuracy y learning rate del entrenamiento.
    Funciona tanto con modelos clásicos como semánticos.
    """
    hist = historia.history
    
    # Determinar qué tipo de métricas estamos manejando
    # Modelo clásico: tiene 'accuracy' y 'val_accuracy'
    # Modelo semántico: tiene métricas específicas de outputs
    tiene_accuracy_clasico = "accuracy" in hist
    tiene_lr = "lr" in hist
    
    # Contar cuántas columnas necesitamos
    n_cols = 1  # Siempre al menos loss
    if tiene_accuracy_clasico or any('accuracy' in k for k in hist.keys()):
        n_cols += 1  # Accuracy
    if tiene_lr:
        n_cols += 1  # Learning rate

    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 4))
    if n_cols == 1:
        axes = [axes]  # Hacer que sea iterable incluso con un solo subplot
    
    col_idx = 0

    # ── Loss ──────────────────────────────────────────────────────────────────
    axes[col_idx].plot(hist["loss"],     label="Entrenamiento", linewidth=2)
    axes[col_idx].plot(hist["val_loss"], label="Validación",    linewidth=2)
    axes[col_idx].set_title("Pérdida (Loss)", fontsize=12, fontweight="bold")
    axes[col_idx].set_xlabel("Épocas")
    axes[col_idx].set_ylabel("Loss")
    axes[col_idx].legend()
    axes[col_idx].grid(True, alpha=0.3)
    col_idx += 1

    # ── Accuracy ──────────────────────────────────────────────────────────────
    # Para modelo clásico
    if tiene_accuracy_clasico:
        axes[col_idx].plot(hist["accuracy"],     label="Entrenamiento", linewidth=2)
        axes[col_idx].plot(hist["val_accuracy"], label="Validación",    linewidth=2)
        axes[col_idx].set_title("Precisión (Accuracy)", fontsize=12, fontweight="bold")
        axes[col_idx].set_xlabel("Épocas")
        axes[col_idx].set_ylabel("Accuracy")
        axes[col_idx].legend()
        axes[col_idx].grid(True, alpha=0.3)
    # Para modelo semántico - buscar métricas de accuracy de similarity_score
    elif any('similarity_score_accuracy' in k for k in hist.keys()):
        # Encontrar la clave correcta para accuracy de similarity_score
        train_key = next((k for k in hist.keys() if 'similarity_score_accuracy' in k and not k.startswith('val_')), None)
        val_key = next((k for k in hist.keys() if 'val_similarity_score_accuracy' in k), None)
        
        if train_key and val_key:
            axes[col_idx].plot(hist[train_key],     label="Entrenamiento", linewidth=2)
            axes[col_idx].plot(hist[val_key], label="Validación",    linewidth=2)
            axes[col_idx].set_title("Precisión (Similarity Score)", fontsize=12, fontweight="bold")
            axes[col_idx].set_xlabel("Épocas")
            axes[col_idx].set_ylabel("Accuracy")
            axes[col_idx].legend()
            axes[col_idx].grid(True, alpha=0.3)
        else:
            # Si no encontramos las claves específicas, mostrar un mensaje
            axes[col_idx].text(0.5, 0.5, 'Accuracy data not available\nin expected format', 
                             horizontalalignment='center', verticalalignment='center',
                             transform=axes[col_idx].transAxes)
            axes[col_idx].set_title("Precisión (Accuracy)", fontsize=12, fontweight="bold")
    else:
        # No hay datos de accuracy disponibles
        axes[col_idx].text(0.5, 0.5, 'No accuracy data available', 
                         horizontalalignment='center', verticalalignment='center',
                         transform=axes[col_idx].transAxes)
        axes[col_idx].set_title("Precisión (Accuracy)", fontsize=12, fontweight="bold")
    
    col_idx += 1

    # ── Learning Rate  (si ReduceLROnPlateau lo registró) ─────────────────────
    if tiene_lr:
        axes[col_idx].plot(hist["lr"], linewidth=2, color="tomato")
        axes[col_idx].set_title("Learning Rate", fontsize=12, fontweight="bold")
        axes[col_idx].set_xlabel("Épocas")
        axes[col_idx].set_ylabel("LR")
        axes[col_idx].set_yscale("log")
        axes[col_idx].grid(True, alpha=0.3)
        col_idx += 1

    plt.tight_layout()

    if guardar_imagen:
        ruta = config.MODELS_DIR / "train_model.png"
        plt.savefig(ruta, dpi=300, bbox_inches="tight")
        logger.info(f"Gráfico guardado en: {ruta}")

    plt.show()
    return fig

if __name__ == "__main__":
    logger.info("═" * 70)
    logger.info("  ENTRENAMIENTO DE MODELOS DE SIMILITUD SEMÁNTICA")
    logger.info("═" * 70)

    # historia, modelo, scaler = train_model(use_embeddings=True)
    logger.info("\nSEMANTIC MODEL")
    logger.info("   Características: LSTM + Atención + Features Semánticos + Contexto JSON")
    historia, modelo, scaler = train_model()

    evaluar_post_entrenamiento()
    graficar_entrenamiento(historia)