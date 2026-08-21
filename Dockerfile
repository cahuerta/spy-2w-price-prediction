# =========================================================
# Dockerfile — APP PREDICTIVA (Trading Suite)
# =========================================================
# Base: Python 3.11 (slim) — mismo runtime que ya usa Render
# con el buildpack automático.
#
# Se agrega:
#   - Node.js 20 LTS (requerido por Claude Code CLI, que no
#     es un paquete Python y no puede instalarse vía pip)
#   - @anthropic-ai/claude-code (CLI del agente, modo headless)
#
# El resto del build (dependencias Python, start command)
# replica exactamente lo que Render ya hacía con el buildpack
# nativo — no cambia el comportamiento de la app.
# =========================================================

FROM python:3.11-slim

# ---------------------------------------------------------
# Dependencias de sistema + Node.js 20 LTS
# ---------------------------------------------------------
# curl y gnupg son necesarios para agregar el repositorio de
# NodeSource de forma segura (firma GPG oficial de Node.js).
# build-essential cubre paquetes Python que compilan C
# (numpy/pandas/pyarrow suelen necesitarlo en imágenes slim).
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        gnupg \
        build-essential \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------
# Claude Code CLI (modo headless, autenticado vía
# CLAUDE_CODE_OAUTH_TOKEN — variable de entorno configurada
# en Render, generada con `claude setup-token`)
# ---------------------------------------------------------
RUN npm install -g @anthropic-ai/claude-code

# ---------------------------------------------------------
# App Python — idéntico a lo que hacía el buildpack de Render
# ---------------------------------------------------------
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Render inyecta la variable PORT en runtime — se mantiene
# el mismo start command que ya usaba el servicio.
ENV PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
