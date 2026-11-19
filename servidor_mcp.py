"""Servidor MCP de la oficina estudiantil (DES estocástica), transporte streamable-http.

Expone:
- herramienta `evaluar_capacidades_des`: recibe personas, media exponencial de llegadas,
  distribución y parámetros del servicio, réplicas y semilla; simula 1, 2 y 3 puestos con las
  mismas muestras en cada réplica y devuelve configuración, agregados, indicadores por réplica
  y la ruta de los archivos con las muestras;
- recurso `des://supuestos`: supuestos, distribuciones, parámetros, semillas y regla;
- prompt `asistente_des`: instrucciones para el especialista que usa la herramienta.

Iniciar (terminal 1):   uv run --locked --extra mcp python servidor_mcp.py
Detener:                Ctrl+C
"""

import json
import os
from typing import Any, Literal

from mcp.server.fastmcp import FastMCP

from oficina_des.experimento import LIMITE_PERSONAS, REPLICAS_MAX, REPLICAS_MIN, evaluar_capacidades

HOST = os.getenv("DES_MCP_HOST", "127.0.0.1")
PUERTO = int(os.getenv("DES_MCP_PUERTO", "8765"))

mcp = FastMCP(
    "Oficina estudiantil DES", host=HOST, port=PUERTO,
    stateless_http=True, json_response=True,
)

SUPUESTOS = {
    "sistema": "Una fila FIFO, puestos de atención idénticos, sistema inicialmente vacío, "
               "primera persona en t=0, sin abandonos; termina cuando sale la última persona.",
    "unidad": "minutos",
    "capacidades_evaluadas": [1, 2, 3],
    "distribuciones": {
        "llegadas": {"tipo": "exponencial", "parametros": {"media": "minutos entre llegadas (> 0); tasa = 1/media"}},
        "servicio_triangular": {"tipo": "triangular",
                                "parametros": {"minimo": "> 0", "moda": "valor más probable", "maximo": "> moda"},
                                "restriccion": "0 < minimo < moda < maximo"},
        "servicio_uniforme": {"tipo": "uniforme continua",
                              "parametros": {"minimo": "> 0", "maximo": "> minimo"},
                              "restriccion": "0 < minimo < maximo"},
    },
    "escenario_inicial": {"personas": 20, "llegadas": {"tipo": "exponencial", "media": 2.0},
                          "servicio": {"tipo": "triangular", "minimo": 1.0, "moda": 3.0, "maximo": 5.0},
                          "replicas": 30, "semilla_base": 2026, "meta_espera_min": 1.0},
    "rangos": {"personas": [1, LIMITE_PERSONAS], "replicas": [REPLICAS_MIN, REPLICAS_MAX], "semilla_base": ">= 0"},
    "semillas": "Réplica r (desde 0): llegadas con random.Random(base + 2r), servicios con random.Random(base + 2r + 1). "
                "Las muestras de una réplica se comparten entre 1, 2 y 3 puestos.",
    "indicadores": "Por réplica y capacidad: espera promedio en fila (sin servicio) y salida de la última persona. "
                   "Agregados: promedio entre réplicas y desviación estándar muestral (R - 1).",
    "regla": "Menor cantidad de puestos cuyo promedio de esperas promedio sea <= meta; None si ninguna cumple. "
             "Es una estimación por simulación: no garantiza que cada persona o réplica cumpla la meta.",
}


@mcp.tool()
def evaluar_capacidades_des(
    personas: int,
    llegadas_media: float,
    servicio_tipo: Literal["triangular", "uniforme"],
    servicio_minimo: float,
    servicio_maximo: float,
    replicas: int,
    semilla_base: int,
    servicio_moda: float | None = None,
) -> dict[str, Any]:
    """Simula una DES estocástica con 1, 2 y 3 puestos, compartiendo muestras en cada réplica.

    Llegadas exponenciales con media llegadas_media (min). Servicio triangular
    (servicio_minimo < servicio_moda < servicio_maximo) o uniforme (servicio_moda vacío).
    personas: 1 a 100; replicas: 30 a 100; semilla_base: entero >= 0.
    Devuelve configuración, agregados por capacidad, indicadores por réplica y archivos de muestras.
    """
    servicio = {"tipo": servicio_tipo, "minimo": servicio_minimo, "maximo": servicio_maximo}
    if servicio_tipo == "triangular":
        servicio["moda"] = servicio_moda
    elif servicio_moda is not None:
        raise ValueError("La distribución uniforme no tiene moda: deja servicio_moda vacío.")
    return evaluar_capacidades({
        "personas": personas, "llegadas": {"tipo": "exponencial", "media": llegadas_media},
        "servicio": servicio, "replicas": replicas, "semilla_base": semilla_base,
    })


@mcp.resource("des://supuestos")
def supuestos() -> str:
    """Supuestos, distribuciones, parámetros, semillas y regla de la oficina estudiantil."""
    return json.dumps(SUPUESTOS, ensure_ascii=False, indent=2)


@mcp.prompt()
def asistente_des() -> str:
    """Instrucciones para el especialista que usa el simulador MCP."""
    return (
        "Eres el especialista de simulación de una oficina estudiantil. Responde en español. "
        "Usa evaluar_capacidades_des exactamente una vez con los datos recibidos; no inventes cifras "
        "ni cambies parámetros. Informa que se evaluaron 1, 2 y 3 puestos con muestras compartidas."
    )


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
