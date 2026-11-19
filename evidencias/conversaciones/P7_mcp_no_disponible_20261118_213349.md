# Conversación `P7_mcp_no_disponible` (modelo de ensayo)

Inicio: 20261118_213349

## Hilo `P7_mcp_no_disponible-hilo`
**Usuario:** Evalúa el escenario inicial y registra la decisión.

**Agente:**

El servidor MCP no está disponible; no hay evaluación ni registro.

## Estado final

```json
{
  "hilo": "P7_mcp_no_disponible-hilo",
  "contexto": {
    "rol": "operador",
    "meta_espera": 1.0
  },
  "escenario": {
    "personas": 20,
    "llegadas": {
      "tipo": "exponencial",
      "media": 2.0
    },
    "servicio": {
      "tipo": "triangular",
      "minimo": 1.0,
      "moda": 3.0,
      "maximo": 5.0
    },
    "replicas": 30,
    "semilla_base": 2026
  },
  "agregados": [],
  "archivos": null,
  "recomendacion": null,
  "meta_recomendacion": null,
  "decision": null
}
```
