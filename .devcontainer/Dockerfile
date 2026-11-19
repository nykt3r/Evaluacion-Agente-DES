# Python 3.13 (multiarquitectura: amd64 y arm64) y uv fijado para reproducir uv.lock.
FROM mcr.microsoft.com/devcontainers/python:1-3.13-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.13.5 /uv /uvx /usr/local/bin/

ENV UV_PYTHON_DOWNLOADS=never \
    UV_PYTHON=/usr/local/bin/python3.13 \
    UV_LINK_MODE=copy
