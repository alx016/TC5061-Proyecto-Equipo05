# Guía de Inicio y Reproducibilidad

Esta guía describe el procedimiento para clonar el repositorio, configurar el entorno virtual, reproducir los análisis exploratorios y compilar este portal de documentación técnica.

---

## Prerrequisitos del Sistema

- **Python 3.12+** instalado en el sistema.
- **uv** (gestor moderno y ultrarrápido de paquetes para Python):

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

- **Git 2.40+** con soporte para autenticación HTTPS o SSH.

---

## 1. Clonación y Configuración del Entorno

Clone el repositorio oficial del equipo e inicialice el entorno virtual con todas las dependencias bloqueadas:

```bash
# Clonar el repositorio
git clone https://github.com/alx016/TC5061-Proyecto-Equipo05.git
cd TC5061-Proyecto-Equipo05

# Sincronizar el entorno virtual desde uv.lock
uv sync
```

---

## 2. Ejecución de Pruebas Automatizadas

Verifique la integridad del módulo de ingesta, utilidades y configuración ejecutando la suite de pruebas unitarias:

```bash
uv run pytest tests/ -v
```

---

## 3. Exploración de Datos en JupyterLab

Para abrir y reejecutar el notebook de auditoría longitudinal y análisis exploratorio:

```bash
uv run jupyter lab notebooks/1.0-exploracion-inicial.ipynb
```

El notebook genera automáticamente las 5 figuras vectoriales estáticas en el directorio `reports/figures/` con reproducibilidad determinista.

---

## 4. Servidor de Documentación Técnica

Para levantar en local este portal de documentación con recarga en tiempo real (*live-reload*):

```bash
uv run mkdocs serve -f docs/mkdocs.yml
```

El portal estará disponible de inmediato en [http://localhost:8000](http://localhost:8000).

Para compilar el sitio estático para producción:

```bash
uv run mkdocs build -f docs/mkdocs.yml --strict
```
