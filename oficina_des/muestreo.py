"""Distribuciones de la actividad y funciones de muestreo con `random.Random` local.

| Componente           | Distribución | Parámetros (minutos)                  | Función de Python                    |
| -------------------- | ------------ | ------------------------------------- | ------------------------------------ |
| Tiempo entre llegadas | exponencial  | media > 0 (tasa = 1 / media)          | rng.expovariate(1 / media)           |
| Servicio inicial     | triangular   | 0 < minimo < moda < maximo            | rng.triangular(minimo, maximo, moda) |
| Servicio alternativo | uniforme     | 0 < minimo < maximo                   | rng.uniform(minimo, maximo)          |

Ojo con el orden de `random.triangular(low, high, mode)`: la moda va al final.
"""

import math
import random


def _finito(nombre: str, valor) -> float:
    """Convierte a float un número finito; rechaza booleanos, textos, infinitos y NaN."""
    if isinstance(valor, bool) or not isinstance(valor, (int, float)) or not math.isfinite(valor):
        raise ValueError(f"{nombre} debe ser un número finito.")
    return float(valor)


def validar_llegadas(llegadas: dict) -> dict:
    """Solo se admite la exponencial para los tiempos entre llegadas."""
    if not isinstance(llegadas, dict) or llegadas.get("tipo") != "exponencial":
        raise ValueError("Las llegadas deben ser {'tipo': 'exponencial', 'media': > 0}.")
    media = _finito("La media exponencial", llegadas.get("media"))
    if media <= 0:
        raise ValueError("La media exponencial debe ser positiva.")
    return {"tipo": "exponencial", "media": media}


def validar_servicio(servicio: dict) -> dict:
    """Triangular (minimo < moda < maximo) o uniforme continua (minimo < maximo), todo > 0."""
    if not isinstance(servicio, dict):
        raise ValueError("El servicio debe describirse con un diccionario.")
    tipo = servicio.get("tipo")
    if tipo == "triangular":
        a, c, b = (_finito(f"El {k} triangular", servicio.get(k)) for k in ("minimo", "moda", "maximo"))
        if not 0 < a < c < b:
            raise ValueError("Triangular inválida: se requiere 0 < minimo < moda < maximo.")
        return {"tipo": "triangular", "minimo": a, "moda": c, "maximo": b}
    if tipo == "uniforme":
        a, b = (_finito(f"El {k} uniforme", servicio.get(k)) for k in ("minimo", "maximo"))
        if not 0 < a < b:
            raise ValueError("Uniforme inválida: se requiere 0 < minimo < maximo.")
        return {"tipo": "uniforme", "minimo": a, "maximo": b}
    raise ValueError("El servicio debe ser 'triangular' o 'uniforme'.")


def muestrear_exponencial(rng: random.Random, media: float, n: int) -> list[float]:
    """n tiempos exponenciales con la media indicada (tasa = 1 / media)."""
    return [rng.expovariate(1.0 / media) for _ in range(n)]


def muestrear_triangular(rng: random.Random, minimo: float, moda: float, maximo: float, n: int) -> list[float]:
    """n tiempos triangulares con mínimo, moda y máximo (random.triangular recibe la moda al final)."""
    return [rng.triangular(minimo, maximo, moda) for _ in range(n)]


def muestrear_uniforme(rng: random.Random, minimo: float, maximo: float, n: int) -> list[float]:
    """n tiempos uniformes continuos entre mínimo y máximo."""
    return [rng.uniform(minimo, maximo) for _ in range(n)]


def muestrear_servicios(rng: random.Random, servicio: dict, n: int) -> list[float]:
    """n tiempos de servicio según la distribución validada (triangular o uniforme)."""
    if servicio["tipo"] == "triangular":
        return muestrear_triangular(rng, servicio["minimo"], servicio["moda"], servicio["maximo"], n)
    return muestrear_uniforme(rng, servicio["minimo"], servicio["maximo"], n)
