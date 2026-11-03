FROM docker.io/library/python:3.13-slim
#LightGBM needs Open MP (libgomp2), which the slim image does not include 
RUN apt-get update \ 
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
    
#uv pinned to the version that wrote uv.lock
COPY --from=ghcr.io/astral-sh/uv:0.13 /uv /uvx /bin/

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy 

#Dependencies first: this layer is reused while uv.lock does not change
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-default-groups --group api --no-install-project

# Project code, then install the package itself
COPY README.md ./
COPY tc5061_proyecto_equipo5/ ./tc5061_proyecto_equipo5/
RUN uv sync --frozen --no-default-groups --group api

# Model exported from the Registry (run make export-model before building)
COPY models/p2-riesgo/ ./models/p2-riesgo/

# Run without root privileges
RUN useradd --create-home --uid 1000 app
USER app

ENV PATH="/app/.venv/bin:$PATH" \
    MODEL_URI=/app/models/p2-riesgo

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"
CMD ["uvicorn", "tc5061_proyecto_equipo5.api.main:app", "--host", "0.0.0.0", "--port", "8000"]