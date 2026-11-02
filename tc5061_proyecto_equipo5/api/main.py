from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request
from loguru import logger

from tc5061_proyecto_equipo5.api.model import LoadedModel, load_model


def get_model(request: Request) -> LoadedModel:
    """Give endpoints the model loaded at startup, or answer 503 if it failed to load."""
    loaded_model = request.app.state.loaded_model
    if loaded_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded; check the API logs")
    return loaded_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once at startup; the API still starts if it fails."""
    try:
        app.state.loaded_model = load_model()
    except Exception:  # noqa: BLE001 - any load failure must leave the API up and answering 503
        logger.exception("Could not load the model; /health and /predict will answer 503")
        app.state.loaded_model = None
    yield


app = FastAPI(title="MiBici P2 · Riesgo de estaciones", version="0.1.0", lifespan=lifespan)


@app.get("/health", dependencies=[Depends(get_model)])
def health():
    """Report that the API is up and the model is loaded."""
    return {"status": "ok"}
