"""P1-P14 como pruebas de pytest (sin clave del modelo). Reutiliza verificar_casos.py.

    uv run --locked --extra mcp pytest
"""
import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import verificar_casos as V  # noqa: E402


@pytest.fixture(scope="module")
def servidor():
    """Servidor MCP real compartido por las pruebas del módulo."""
    s = V.ServidorMCP().iniciar()
    yield s
    s.detener()


@pytest.mark.parametrize("funcion", V.CASOS, ids=[f.__name__.upper() for f in V.CASOS])
def test_caso(funcion, servidor):
    """Ejecuta un caso P1-P14 y exige que su resultado sea correcto."""
    r = asyncio.run(funcion(servidor) if funcion is V.p7 else funcion())
    assert r["ok"], f"{r['caso']}: esperado {r['esperado']}; obtenido {r['obtenido']}"
