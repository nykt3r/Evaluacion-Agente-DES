"""Réplicas con muestras compartidas entre capacidades, agregados y regla de decisión.

- Réplica r (desde 0): semilla de llegadas = base + 2r; de servicios = base + 2r + 1,
  cada una en su propio `random.Random`.
- Las muestras se pregeneran y se reutilizan para 1, 2 y 3 puestos dentro de la réplica.
- Entre réplicas cambian las semillas, por lo tanto las muestras.
- La desviación estándar de las esperas promedio entre réplicas usa denominador R - 1.
- `recomendar` aplica la regla: la menor capacidad cuyo promedio de esperas promedio sea <= meta.
"""

import csv
import hashlib
import json
import math
import os
import random
import statistics
from pathlib import Path

from .muestreo import muestrear_exponencial, muestrear_servicios, validar_llegadas, validar_servicio
from .simulacion_des import CAPACIDADES, simular_cola

LIMITE_PERSONAS = 100
REPLICAS_MIN, REPLICAS_MAX = 30, 100
RAIZ = Path(__file__).resolve().parents[1]


def carpeta_evidencias() -> Path:
    """Carpeta donde se guardan muestras, resultados y trazas (DES_EVIDENCIAS o ./evidencias)."""
    return Path(os.getenv("DES_EVIDENCIAS", RAIZ / "evidencias"))


# ---------------------------------------------------------------------------
# Validación del escenario
# ---------------------------------------------------------------------------
def validar_escenario(personas, llegadas, servicio, replicas, semilla_base,
                      limite_personas: int = LIMITE_PERSONAS) -> dict:
    """Devuelve el escenario normalizado o lanza ValueError. Nunca sustituye valores."""
    if type(personas) is not int or not 1 <= personas <= min(limite_personas, LIMITE_PERSONAS):
        raise ValueError(f"personas debe ser un entero entre 1 y {min(limite_personas, LIMITE_PERSONAS)}.")
    if type(replicas) is not int or not REPLICAS_MIN <= replicas <= REPLICAS_MAX:
        raise ValueError(f"replicas debe ser un entero entre {REPLICAS_MIN} y {REPLICAS_MAX}.")
    if type(semilla_base) is not int or semilla_base < 0:
        raise ValueError("semilla_base debe ser un entero no negativo.")
    return {
        "personas": personas,
        "llegadas": validar_llegadas(llegadas),
        "servicio": validar_servicio(servicio),
        "replicas": replicas,
        "semilla_base": semilla_base,
    }


def validar_meta(meta) -> float:
    """Devuelve la meta como float si es finita y >= 0; si no, lanza ValueError."""
    if isinstance(meta, bool) or not isinstance(meta, (int, float)) or not math.isfinite(meta) or meta < 0:
        raise ValueError("La meta de espera debe ser un número finito mayor o igual a cero.")
    return float(meta)


def huella(escenario: dict) -> str:
    """Identificador corto y estable del escenario (hash de su configuración)."""
    return hashlib.sha256(json.dumps(escenario, sort_keys=True).encode()).hexdigest()[:10]


# ---------------------------------------------------------------------------
# Muestras y simulación
# ---------------------------------------------------------------------------
def semillas(semilla_base: int, r: int) -> tuple[int, int]:
    """Semillas de la réplica r: llegadas = base + 2r; servicios = base + 2r + 1."""
    return semilla_base + 2 * r, semilla_base + 2 * r + 1


def generar_muestras(escenario: dict, r: int) -> dict:
    """Pregenera los n-1 intervalos y los n servicios de la réplica r."""
    s_lleg, s_serv = semillas(escenario["semilla_base"], r)
    rng_llegadas, rng_servicios = random.Random(s_lleg), random.Random(s_serv)
    n = escenario["personas"]
    intervalos = muestrear_exponencial(rng_llegadas, escenario["llegadas"]["media"], n - 1)
    servicios = muestrear_servicios(rng_servicios, escenario["servicio"], n)
    return {"replica": r, "semilla_llegadas": s_lleg, "semilla_servicios": s_serv,
            "intervalos": intervalos, "servicios": servicios}


def evaluar_capacidades(escenario: dict, capacidades=CAPACIDADES, exportar: bool = True) -> dict:
    """Simula todas las capacidades en cada réplica con las MISMAS muestras."""
    escenario = validar_escenario(**escenario)
    if tuple(capacidades) != CAPACIDADES:
        raise ValueError(f"Se evalúan exactamente las capacidades {CAPACIDADES}.")
    muestras, por_replica = [], []
    for r in range(escenario["replicas"]):
        m = generar_muestras(escenario, r)
        muestras.append(m)
        fila = {"replica": r, "semilla_llegadas": m["semilla_llegadas"],
                "semilla_servicios": m["semilla_servicios"], "capacidades": {}}
        for c in capacidades:
            res = simular_cola(c, m["intervalos"], m["servicios"])
            fila["capacidades"][str(c)] = {"espera_promedio": res["espera_promedio"],
                                           "salida_ultimo_cliente": res["salida_ultimo_cliente"]}
        por_replica.append(fila)
    resultado = {
        "configuracion": {**escenario, "capacidades": list(capacidades),
                          "esquema_semillas": "llegadas = base + 2r; servicios = base + 2r + 1 (r desde 0)",
                          "generador": "random.Random (Python), uno por flujo y réplica",
                          "unidad": "minutos"},
        "huella": huella(escenario),
        "agregados": agregar(por_replica, capacidades),
        "por_replica": por_replica,
    }
    if exportar:
        resultado["archivos"] = exportar_evidencia(resultado, muestras)
    return resultado


def agregar(por_replica: list[dict], capacidades=CAPACIDADES) -> list[dict]:
    """Promedio y desviación estándar muestral (R - 1) entre réplicas, por capacidad.

    Cada agregado usa las claves `parametros.servidores` y `espera_promedio`, las mismas que
    lee `recomendar`, de modo que la regla funciona igual sobre datos simulados o controlados.
    """
    agregados = []
    for c in capacidades:
        esperas = [f["capacidades"][str(c)]["espera_promedio"] for f in por_replica]
        salidas = [f["capacidades"][str(c)]["salida_ultimo_cliente"] for f in por_replica]
        agregados.append({
            "parametros": {"servidores": c},
            "replicas": len(esperas),
            "espera_promedio": statistics.fmean(esperas),
            "desviacion_estandar_espera": statistics.stdev(esperas) if len(esperas) > 1 else 0.0,
            "salida_ultimo_cliente": statistics.fmean(salidas),
            "desviacion_estandar_salida": statistics.stdev(salidas) if len(salidas) > 1 else 0.0,
        })
    return agregados


# ---------------------------------------------------------------------------
# Regla de decisión
# ---------------------------------------------------------------------------
def recomendar(resultados, meta):
    """Menor capacidad evaluada con espera promedio <= meta; None si ninguna cumple."""
    meta = validar_meta(meta)
    candidatas = [r["parametros"]["servidores"] for r in resultados
                  if r["espera_promedio"] <= meta]
    return min(candidatas, default=None)


# ---------------------------------------------------------------------------
# Evidencia
# ---------------------------------------------------------------------------
def exportar_evidencia(resultado: dict, muestras: list[dict]) -> dict:
    """JSON con configuración, muestras y resultados; CSV con un renglón por réplica y capacidad."""
    carpeta = carpeta_evidencias() / "simulaciones"
    carpeta.mkdir(parents=True, exist_ok=True)
    base = carpeta / f"evaluacion_{resultado['huella']}"
    with open(base.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump({"configuracion": resultado["configuracion"], "huella": resultado["huella"],
                   "agregados": resultado["agregados"], "por_replica": resultado["por_replica"],
                   "muestras": muestras}, f, ensure_ascii=False, indent=1)
    with open(base.with_suffix(".csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["replica", "semilla_llegadas", "semilla_servicios", "servidores",
                    "espera_promedio_min", "salida_ultimo_cliente_min"])
        for fila in resultado["por_replica"]:
            for c, v in fila["capacidades"].items():
                w.writerow([fila["replica"], fila["semilla_llegadas"], fila["semilla_servicios"], c,
                            repr(v["espera_promedio"]), repr(v["salida_ultimo_cliente"])])
    rel = lambda p: os.path.relpath(p, RAIZ) if str(p).startswith(str(RAIZ)) else str(p)
    return {"muestras_y_resultados_json": rel(base.with_suffix(".json")),
            "resultados_por_replica_csv": rel(base.with_suffix(".csv"))}
