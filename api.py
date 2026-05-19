"""
api.py — FastAPI server for semantic field matching.

Endpoints:
  GET  /health              → Health check
  POST /predict             → Score a single field pair
  POST /predict-objects     → Compare two JSON objects
  POST /predict-batch       → Batch prediction

Usage:
  uvicorn api:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ── Silence TF warnings ───────────────────────────────────────────────────
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("api")

# ── Global predictor (lazy-loaded) ────────────────────────────────────────
_predictor = None


def get_predictor():
    global _predictor
    if _predictor is None:
        from src.training.ensemble_predictor import EnsemblePredictor
        logger.info("Cargando modelo...")
        _predictor = EnsemblePredictor.from_paths()
        logger.info("Modelo cargado exitosamente")
    return _predictor


# ── Pydantic models ───────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    field_a: str = Field(..., description="Source field name", examples=["customer_id"])
    field_b: str = Field(..., description="Target field name", examples=["id_cliente"])


class PredictBatchRequest(BaseModel):
    pairs: list[list[str]] = Field(..., description="List of [field_a, field_b] pairs",
                                    examples=[["customer_id", "id_cliente"]])


class PredictObjectsRequest(BaseModel):
    object_a: dict[str, Any] = Field(..., description="Source JSON object")
    object_b: dict[str, Any] = Field(..., description="Target JSON object")
    strategy: str = Field("top3", description="Strategy: max, avg, or top3")


class PredictResponse(BaseModel):
    campo1: str
    campo2: str
    son_similares: bool
    score_final: float
    porcentaje: str
    desglose: dict
    tokens_compartidos: list[str]


class PredictObjectsResponse(BaseModel):
    score_global: float
    son_similares: bool
    estrategia: str
    total_pares: int
    mejor_par: dict
    todos_los_pares: list[dict]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str = "v3.2.5"


# ── Lifespan ──────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-carga el modelo al iniciar."""
    logger.info("Inicializando API...")
    get_predictor()
    yield
    logger.info("API detenida.")

app = FastAPI(
    title="Semantic Field Matcher API",
    description="API para matching semántico de campos entre APIs/Sistemas",
    version="3.2.5",
    lifespan=lifespan,
)


# ── Endpoints ─────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Status"])
def health():
    predictor = get_predictor()
    return HealthResponse(
        status="ok",
        model_loaded=predictor.modelo is not None,
    )


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict(req: PredictRequest):
    """
    Predice si dos nombres de campo se refieren al mismo concepto.

    Ejemplo:
      POST /predict
      {"field_a": "customer_id", "field_b": "id_cliente"}
    """
    predictor = get_predictor()
    result = predictor.predecir(req.field_a, req.field_b)
    return PredictResponse(
        campo1=result["campo1"],
        campo2=result["campo2"],
        son_similares=result["son_similares"],
        score_final=result["score_final"],
        porcentaje=result["porcentaje"],
        desglose=result["desglose"],
        tokens_compartidos=result.get("tokens_compartidos", []),
    )


@app.post("/predict-batch", tags=["Prediction"])
def predict_batch(req: PredictBatchRequest):
    """
    Predice múltiples pares de campos en una sola petición.

    Ejemplo:
      POST /predict-batch
      {"pairs": [["customer_id", "id_cliente"], ["email", "correo"]]}
    """
    predictor = get_predictor()
    resultados = []
    for field_a, field_b in req.pairs:
        result = predictor.predecir(field_a, field_b)
        resultados.append({
            "campo1": field_a,
            "campo2": field_b,
            "son_similares": result["son_similares"],
            "score_final": result["score_final"],
        })
    return {"resultados": resultados, "total": len(resultados)}


@app.post("/predict-objects", response_model=PredictObjectsResponse, tags=["Prediction"])
def predict_objects(req: PredictObjectsRequest):
    """
    Compara dos objetos JSON campo por campo y retorna el matching global.

    Estrategias:
      - max:   mejor match individual (permisivo)
      - avg:   promedio de todos los pares
      - top3:  promedio de los 3 mejores (balanceado)
    """
    if req.strategy not in ("max", "avg", "top3"):
        raise HTTPException(400, f"Estrategia inválida: {req.strategy!r}. Usa max, avg o top3.")

    predictor = get_predictor()
    result = predictor.predecir_objetos(req.object_a, req.object_b, estrategia=req.strategy)
    return PredictObjectsResponse(
        score_global=result["score_global"],
        son_similares=result["son_similares"],
        estrategia=result["estrategia"],
        total_pares=result["total_pares"],
        mejor_par=result["mejor_par"],
        todos_los_pares=result["todos_los_pares"],
    )


# ── Run directly ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, log_level="info")
