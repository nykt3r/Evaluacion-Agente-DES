"""Simulación de eventos discretos (SimPy) de una fila FIFO con puestos idénticos.

`simular_cola(servidores, intervalos, servicios)` recibe muestras ya generadas:
- `intervalos`: n - 1 tiempos entre llegadas (la primera persona llega en t = 0);
- `servicios`: n tiempos de servicio, asignados por orden de llegada.
Cada persona es un proceso `atender` que llega, pide un puesto (`simpy.Resource`), espera
si están ocupados, se atiende y sale. Devuelve los registros por persona (llegada, inicio,
salida, espera), la espera promedio en fila y la salida de la última persona, en minutos.
"""

import math

import simpy

CAPACIDADES = (1, 2, 3)


def validar_muestras(servidores: int, intervalos: list[float], servicios: list[float]) -> dict:
    """Valida capacidad y muestras; devuelve los parámetros en minutos."""
    if type(servidores) is not int or servidores not in CAPACIDADES:
        raise ValueError(f"servidores debe ser uno de {CAPACIDADES}.")
    if not servicios:
        raise ValueError("Se necesita al menos un servicio (una persona).")
    if len(intervalos) != len(servicios) - 1:
        raise ValueError("Debe haber un intervalo menos que servicios: la primera llegada es t=0.")
    if not all(math.isfinite(t) and t >= 0 for t in intervalos):
        raise ValueError("Los intervalos entre llegadas deben ser finitos y no negativos.")
    if not all(math.isfinite(t) and t > 0 for t in servicios):
        raise ValueError("Los servicios deben ser positivos y finitos.")
    return {"servidores": servidores, "clientes": len(servicios)}


def simular_cola(servidores: int, intervalos: list[float], servicios: list[float]) -> dict:
    """Simula una fila FIFO vacía hasta la salida de la última persona."""
    parametros = validar_muestras(servidores, intervalos, servicios)
    llegadas = [0.0]
    for intervalo in intervalos:
        llegadas.append(llegadas[-1] + intervalo)

    env = simpy.Environment()
    recurso = simpy.Resource(env, capacity=servidores)
    registros = []

    def atender(numero):
        """Proceso de una persona: llega, espera un puesto libre, se atiende y registra sus tiempos."""
        yield env.timeout(llegadas[numero - 1])
        llegada = env.now
        with recurso.request() as solicitud:
            yield solicitud
            inicio = env.now
            yield env.timeout(servicios[numero - 1])
            registros.append({
                "cliente": numero,
                "llegada": llegada,
                "inicio": inicio,
                "salida": env.now,
                "espera": inicio - llegada,
            })

    for numero in range(1, len(servicios) + 1):
        env.process(atender(numero))
    env.run()

    return {
        "parametros": parametros,
        "unidad": "minutos",
        "espera_promedio": sum(r["espera"] for r in registros) / len(servicios),
        "salida_ultimo_cliente": env.now,
        "registros": sorted(registros, key=lambda r: r["cliente"]),
    }
