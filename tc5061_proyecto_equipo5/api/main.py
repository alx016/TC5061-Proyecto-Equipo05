from fastapi import FastAPI

app = FastAPI(title="MiBici P2 · Riesgo de estaciones", version="0.1.0")


@app.get("/health")
def health():
    """Report that the API is up."""
    return {"status": "ok"}
