"""Comprobaciones adicionales de las distribuciones (sin exigir que la media muestral sea exacta)."""
import random
import statistics
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oficina_des.muestreo import (muestrear_exponencial, muestrear_triangular, muestrear_uniforme,  # noqa: E402
                                  validar_llegadas, validar_servicio)


def test_triangular_respeta_limites_y_no_es_constante():
    """La triangular queda entre mínimo y máximo, varía y su media se acerca a (a+c+b)/3."""
    x = muestrear_triangular(random.Random(1), 1.0, 3.0, 5.0, 2000)
    assert min(x) >= 1.0 and max(x) <= 5.0 and statistics.stdev(x) > 0
    assert abs(statistics.fmean(x) - 3.0) < 0.15          # media teórica (1+3+5)/3, con tolerancia


def test_uniforme_respeta_limites():
    """La uniforme queda entre mínimo y máximo y su media se acerca al punto medio."""
    x = muestrear_uniforme(random.Random(2), 1.0, 3.0, 2000)
    assert min(x) >= 1.0 and max(x) <= 3.0 and abs(statistics.fmean(x) - 2.0) < 0.1


def test_exponencial_positiva_y_variable():
    """La exponencial es positiva, variable y su media se acerca a la indicada."""
    x = muestrear_exponencial(random.Random(3), 2.0, 2000)
    assert min(x) > 0 and statistics.stdev(x) > 0 and abs(statistics.fmean(x) - 2.0) < 0.2


@pytest.mark.parametrize("servicio", [
    {"tipo": "triangular", "minimo": 1, "moda": 1, "maximo": 5},
    {"tipo": "triangular", "minimo": 0, "moda": 3, "maximo": 5},
    {"tipo": "uniforme", "minimo": 2, "maximo": 2},
    {"tipo": "normal", "minimo": 1, "maximo": 3},
    {"tipo": "uniforme", "minimo": 1, "maximo": float("inf")},
])
def test_servicios_invalidos(servicio):
    """Parámetros de servicio inválidos se rechazan con ValueError."""
    with pytest.raises(ValueError):
        validar_servicio(servicio)


@pytest.mark.parametrize("llegadas", [{"tipo": "exponencial", "media": 0}, {"tipo": "exponencial", "media": float("nan")},
                                      {"tipo": "constante", "media": 2}, {"tipo": "exponencial", "media": True}])
def test_llegadas_invalidas(llegadas):
    """Parámetros de llegadas inválidos se rechazan con ValueError."""
    with pytest.raises(ValueError):
        validar_llegadas(llegadas)
