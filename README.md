# Agente DES con MCP para dimensionar una oficina de atención estudiantil

Actividad práctica de implementación de un agente DES · Modelos y Simulación

| Integrante | Contribuciones |
| --- | --- |
| **Lina Paola Suárez Díaz** | Simulación y MCP: muestreo (`muestreo.py`), simulador y réplicas (`simulacion_des.py`, `experimento.py`), regla de decisión y validaciones, servidor MCP y recurso (`servidor_mcp.py`), casos P1, P2, P3, P6, P7, P13 y P14. |
| **Juan José Álvarez González** | Agente y revisión humana: estado, contexto y herramientas con `Command` (`agente.py`), especialista (ampliación A), HITL y restricción de rol, terminal y trazas (`main.py`, `trazas.py`), casos P4, P5, P8 a P12 y las conversaciones reales. |
| Ambos | Integración, README y revisión de todo el contenido. |

---

## 1. Problema, distribuciones, regla y ampliación

Una oficina de atención estudiantil elige entre **1, 2 y 3 puestos idénticos**. Las condiciones del sistema son:
- Hay una sola fila FIFO y la oficina empieza vacía.
- La primera persona llega en t = 0 y todas esperan hasta ser atendidas.
- La simulación termina cuando sale la última persona.
- Todos los tiempos están en **minutos**.

| Componente | Distribución | Parámetros y significado | Validación |
| --- | --- | --- | --- |
| Tiempo entre llegadas | Exponencial | `media`: minutos promedio entre llegadas (`tasa = 1/media`) | media finita y > 0 |
| Servicio (inicial) | Triangular | `minimo`, `moda` (valor más probable), `maximo` | 0 < minimo < moda < maximo |
| Servicio (alternativa) | Uniforme continua | `minimo`, `maximo` | 0 < minimo < maximo |

- **Muestreo** (`oficina_des/muestreo.py`): `random.Random` local, con `expovariate(1/media)`, `triangular(minimo, maximo, moda)` y `uniform(minimo, maximo)`.
- **Semillas:** en la réplica *r* (desde 0), las llegadas usan `Random(base + 2r)` y los servicios `Random(base + 2r + 1)`. Se pregeneran *n − 1* intervalos y *n* servicios. El servicio se asigna por orden de llegada.
- **Comparación de alternativas:** las mismas muestras se usan para 1, 2 y 3 puestos dentro de cada réplica. Cada réplica tiene muestras nuevas.
- **Indicadores por réplica y capacidad:** espera promedio en fila (sin el servicio) y salida de la última persona.
- **Agregados:** promedio entre réplicas y **desviación estándar muestral (R − 1)** de las esperas promedio.
- **Regla (en Python, `recomendar`):** la menor cantidad de puestos cuyo promedio de esperas promedio sea **≤ meta**. Si ninguna cumple, el resultado es `None`. Es una estimación por simulación: no garantiza que cada persona ni cada réplica cumpla la meta.
- **Ampliación elegida para calificación: A, delegación en un subagente.** El coordinador delega la simulación en un *especialista* construido con `create_agent`, cuya única herramienta es la herramienta MCP. Ver la sección 7.

## 2. Requisitos e instalación

- Python **3.13** y [uv](https://docs.astral.sh/uv/) 0.13.5, con todas las dependencias fijadas en `uv.lock`.
- **Opción recomendada:** VS Code con la extensión *Dev Containers* y Docker. Abrir la carpeta y elegir **Reopen in Container**. El contenedor ejecuta `uv sync --locked --extra mcp` y queda listo.

  | Sistema | Requisito de Docker | Observación |
  | --- | --- | --- |
  | macOS (Intel o Apple Silicon) | Docker Desktop | La imagen es multiarquitectura (amd64/arm64). |
  | Windows 10/11 | Docker Desktop con backend WSL 2 | Más rápido si el repositorio se clona dentro de WSL (`\\wsl$\...`). `.gitattributes` fija finales de línea LF. |
  | Linux | Docker Engine o Docker Desktop | `updateRemoteUserUID` evita que los archivos queden a nombre de root. |
- **Sin contenedor:**
  ```bash
  uv sync --locked --extra mcp      # dependencias base + extra MCP (mcp, langchain-mcp-adapters)
  ```
  El extra **`mcp`** es necesario para el servidor y para el agente. Versiones: langchain 1.3.14, langchain-openai 1.4.1, langgraph 1.2.10, mcp 1.29.0, langchain-mcp-adapters 0.3.1, simpy 4.1.1 y pytest 9.1.1. En Windows sin contenedor, use `copy` en lugar de `cp` para crear el `.env`.

## 3. Configuración (`.env`)

```bash
cp .env.example .env              # luego escribir la clave
```

| Variable | Uso |
| --- | --- |
| `OPENROUTER_API_KEY` | Clave de OpenRouter. **Solo `main.py` la usa.** |
| `OPENROUTER_MODEL` | Modelo con soporte de herramientas (por defecto `nvidia/nemotron-3-ultra-550b-a55b:free`). |
| `OPENROUTER_MODEL_REVISION` | Opcional: modelo de la etapa de revisión (por defecto, el mismo). |
| `DES_MCP_URL` | Dirección del servidor MCP (`http://127.0.0.1:8765/mcp`). |
| `DES_MCP_HOST`, `DES_MCP_PUERTO` | Opcionales: host y puerto del servidor. |
| `DES_EVIDENCIAS` | Opcional: carpeta de evidencias (por defecto `./evidencias`). |

**Qué consulta una API externa:** solo `main.py`, que llama a OpenRouter. Los modelos `:free` no cobran pero tienen límites de solicitudes; un modelo de pago sí factura. El servidor MCP, `verificar_casos.py` y `pytest` funcionan sin clave y sin internet.

## 4. Ejecución

Se necesitan **dos terminales**, en este orden:

```bash
# Terminal 1: servidor MCP (Ctrl+C para detener)
uv run --locked --extra mcp python servidor_mcp.py

# Terminal 2: agente (escribir /salir o Ctrl+D para terminar)
uv run --locked --extra mcp python main.py --rol operador --meta 1.0
```

**Contexto de la aplicación**, fijado por argumentos y nunca por el mensaje al modelo:
- `--rol observador|operador`
- `--meta MIN`
- `--limite-personas N`, de 1 a 100
- `--formato breve|didactico`

**Otros argumentos:**
- `--hilo NOMBRE`: el `thread_id`.
- `--guion ARCHIVO`: reproduce una conversación desde un archivo.
- `--etiqueta`: nombre de los archivos de evidencia.

**Comandos durante la conversación:**
- `/meta X`: cambia la meta del contexto y recalcula la recomendación sin volver a simular.
- `/hilo X`: cambia de hilo.
- `/estado`: muestra el estado del hilo.
- `/salir`: termina.

### Ejemplo completo

```text
Usted> Evalúa 20 personas, llegadas exponenciales con media 2 min, servicio triangular 1, 3 y 5 min,
       30 réplicas y semilla 2026. Compara 1, 2 y 3 puestos y recomienda la capacidad.
Agente: (tabla de espera promedio ± DE y salida por puesto) Recomendación: 3 puestos...
Usted> Ahora usa un servicio uniforme entre 1 y 3 minutos; conserva los demás datos.
Agente: (nueva tabla) Recomendación: 2 puestos...
Usted> Registra la capacidad recomendada.
[Revisión humana requerida] Acción propuesta:
  registrar_decision({"servidores": 2, "nota": "..."})
  Decisión registrada en este momento: None
[a]probar / [e]ditar nota / [r]echazar: e
Nota corregida: Revisado por el equipo.
Decisión en el estado: {"servidores": 2, "nota": "Revisado por el equipo.", "meta_espera": 1.0, ...}
```

Para reproducir otro escenario basta pedirlo en el mensaje: otras personas, media, tipo y parámetros de servicio, réplicas (30–100) o semilla. Solo cambia lo que se pide. La meta y el rol se cambian con los argumentos o con `/meta`.

## 5. Pruebas sin clave y conversaciones con API

```bash
uv run --locked --extra mcp python verificar_casos.py   # P1-P14 → evidencias/verificacion/tabla_verificacion.md
uv run --locked --extra mcp pytest                       # mismas verificaciones + pruebas de muestreo (26 pruebas)
```

Ambos comandos **arrancan su propio servidor MCP real** en el puerto 8799 y lo detienen al final. P7 lo apaga a propósito para probar el fallo.

**Modo de ensayo:** la orquestación y la revisión humana se prueban con `ModeloEnsayo`, un modelo de chat cuyas respuestas están programadas. Lo que demuestra es que las herramientas, el estado, `Command`, HITL, los hilos, las validaciones y MCP funcionan. Lo que no demuestra es que un modelo real tome buenas decisiones. Esas evidencias llevan `"origen": "modelo de ensayo"`.

**Conversaciones con un modelo real:** con el servidor iniciado y el `.env` configurado.

```bash
uv run --locked --extra mcp python main.py --rol operador --meta 1 --hilo real-aprobacion \
  --etiqueta real_1_aprobacion --guion guiones/conversacion_1_aprobacion.txt        # P1 + P8 (aprobación)
uv run --locked --extra mcp python main.py --rol operador --meta 1 --hilo real-continuacion \
  --etiqueta real_2_continuacion --guion guiones/conversacion_2_continuacion.txt    # P1 + P4 + aclaración
uv run --locked --extra mcp python main.py --rol observador --meta 1 --hilo real-observador \
  --etiqueta real_3_observador --guion guiones/conversacion_3_observador.txt        # opcional: P5 + P11
```

## 6. Evidencias, resultados y limitaciones

| Carpeta | Contenido |
| --- | --- |
| `evidencias/verificacion/` | Tabla de verificación P1–P14 (`tabla_verificacion.md`, `resultados_verificacion.json`) y un JSON por caso. |
| `evidencias/simulaciones/` | Por escenario evaluado: `evaluacion_<huella>.json` (configuración, semillas, **todas las muestras**, indicadores por réplica y agregados) y `.csv` (una fila por réplica y capacidad). |
| `evidencias/conversaciones/` | Trazas JSONL y transcripciones `.md`: mensajes, contexto, llamadas a herramientas (coordinador y especialista), recurso MCP leído, resultados MCP, interrupciones, revisiones y estado final. Cada evento indica `origen`: modelo real o de ensayo. |

**Resultados del escenario inicial** (P1: 20 personas, semilla 2026, 30 réplicas; espera promedio ± DE entre réplicas):

| Puestos | Espera promedio ± DE |
| --- | --- |
| 1 | 11,881 ± 4,546 min |
| 2 | 1,376 ± 1,377 min |
| 3 | 0,251 ± 0,357 min |

Con meta de 1 min se recomiendan **3 puestos**. Con servicio uniforme(1, 3) se recomiendan **2** (P4). La tabla completa de P1 a P14 está en `evidencias/verificacion/tabla_verificacion.md`.

Una sola corrida aleatoria no basta: la espera con 2 puestos varía de una réplica a otra (DE ≈ 1,4 min, del mismo orden que la meta). Con una sola corrida la recomendación dependería del azar de esa muestra.

**Limitaciones conocidas:**
- La memoria (`InMemorySaver`) y las revisiones pendientes se pierden al cerrar `main.py`.
- Los modelos gratuitos pueden fallar por límite de solicitudes (429). El agente lo informa y no inventa resultados.
- Las preguntas de aclaración (P5 y "cambia el servicio a 1 minuto") dependen de las instrucciones y del modelo. Python impide simular sin datos completos, pero no puede obligar al modelo a formular la pregunta.
- El rol viene de un argumento de la aplicación, no de un usuario autenticado.

## 7. Explicación técnica

**Estado, contexto e historial.**
- **El historial** (`messages`) es la conversación.
- **El estado** (`EstadoOperacion`) guarda en campos explícitos el `escenario` (distribuciones, parámetros, personas, réplicas y semilla), los `resultados`, la `recomendacion`, la `meta_recomendacion` y la `decision`. Las herramientas lo actualizan con `Command`, y cada `ToolMessage` lleva el `tool_call_id` de su llamada.
- **El contexto** (`ContextoOperacion`: rol, meta y límite de personas) lo entrega la aplicación en cada `ainvoke`. No forma parte del mensaje, así que el modelo no puede cambiarlo.

**Invalidación.** `guardar_escenario` borra los resultados, la recomendación y la decisión antes de recalcular. Si solo cambia la meta, un middleware `before_agent` (o el comando `/meta`) recalcula la recomendación con los mismos resultados y anula la decisión.

**Qué decide el modelo y qué calcula Python.**
- *El modelo:* interpreta la petición, pide los datos que faltan, elige qué herramienta llamar y redacta la respuesta.
- *Python:* valida los datos, genera las muestras, simula con SimPy, calcula los agregados, aplica la regla y valida el registro.

El especialista (ampliación A) es otro `create_agent` con un solo trabajo: llamar la herramienta MCP con el escenario guardado. El coordinador no confía en él a ciegas. Comprueba que la configuración y la huella del resultado coincidan con el escenario del estado, y solo entonces guarda los datos y calcula la recomendación.

**Validaciones.**
- Personas de 1 a 100 (y no más que el límite del contexto), réplicas de 30 a 100, semilla entera ≥ 0, parámetros finitos y coherentes con su distribución, meta finita ≥ 0.
- Los datos inválidos se rechazan explícitamente y nunca se reemplazan por valores predeterminados. Sin escenario, `guardar_escenario` lista los datos que faltan.
- `registrar_decision` exige cuatro condiciones:
  - rol operador;
  - resultados completos del escenario vigente (misma huella, todas las réplicas y capacidades);
  - una recomendación calculada con la meta vigente;
  - una capacidad igual a esa recomendación.

**Revisión humana.** `HumanInTheLoopMiddleware` interrumpe antes de `registrar_decision`. La terminal muestra la acción y sus argumentos, y la persona elige:
- *aprobar*: se registra la propuesta;
- *editar*: cambia la nota y la herramienta vuelve a validar;
- *rechazar*: la decisión queda vacía y los resultados se conservan.

La reanudación usa `Command(resume=...)` en el mismo `thread_id`. El observador tiene dos defensas: un `wrap_model_call` le oculta la herramienta y, aun si el modelo la pidiera, la validación interna la rechaza.

**Semillas y variabilidad.** Cada réplica tiene dos generadores independientes: `base+2r` para llegadas y `base+2r+1` para servicios. Las tres capacidades se simulan con las mismas muestras de la réplica, así que las diferencias entre ellas se deben a la capacidad y no al azar. La desviación estándar entre réplicas (R − 1) mide la variabilidad de la espera promedio de un lote de 20 personas, no la dispersión entre personas. Con la misma configuración, versión y semilla los resultados se reproducen exactamente (P14).

**Memoria y rol.**
- La memoria dura mientras vive el proceso de `main.py`, porque `InMemorySaver` vive en memoria. Cada `thread_id` es independiente.
- El rol lo fija quien ejecuta la aplicación (`--rol`). Para asociarlo a usuarios autenticados, un servicio debería derivar `ContextoOperacion` de la identidad verificada (token o sesión) en el servidor, sin aceptarlo del cliente.

## 8. Estructura

```text
servidor_mcp.py                servidor MCP: herramienta evaluar_capacidades_des, recurso des://supuestos, prompt
main.py                        entrada de terminal: conversación, hilos, /meta, revisión humana, trazas
verificar_casos.py             verificaciones P1-P14 y tabla de resultados
oficina_des/muestreo.py        distribuciones y validación de sus parámetros
oficina_des/simulacion_des.py  simulador SimPy que recibe muestras
oficina_des/experimento.py     réplicas, semillas, agregados, regla y exportación de evidencias
oficina_des/agente.py          estado, contexto, herramientas, especialista, middlewares y revisión humana
oficina_des/trazas.py          evidencia JSONL y Markdown
guiones/                       conversaciones reales reproducibles
tests/                         pytest
evidencias/                    resultados, muestras, trazas y tabla de verificación
.devcontainer/                 entorno reproducible (macOS, Windows y Linux)
```

### Archivos adaptados, fuentes y uso de IA

Esta sección responde al requisito del enunciado de identificar el código reutilizado.

| Origen | Archivo del proyecto | Cambios realizados |
| --- | --- | --- |
| Simulador de la fila con tiempos constantes | `oficina_des/simulacion_des.py` | Recibe listas de intervalos y servicios; capacidades 1, 2 y 3. |
| Servidor MCP con herramienta, recurso y prompt | `servidor_mcp.py` | Herramienta estocástica que evalúa las tres capacidades en una llamada; recurso con distribuciones y semillas. |
| Agente con estado, contexto, middlewares y revisión humana | `oficina_des/agente.py` | Tres capacidades, escenario estocástico, límite de personas, especialista MCP, invalidación por meta y validaciones adicionales del registro. |
| Modelo de ensayo y manejo de interrupciones | `verificar_casos.py`, `main.py` | Casos P1-P14, guiones y terminal con aprobar, editar y rechazar. |
| Entorno (`.devcontainer`, `pyproject.toml`, `uv.lock`) | Mismos nombres | Mismas versiones de dependencias; se agrega pytest; configuración para cualquier sistema operativo. |

**Fuentes:** material del curso de Modelos y Simulación; documentación de LangChain (middleware, revisión humana, memoria de corto plazo), de LangGraph y del SDK de MCP; documentación de `random` de Python.

**Uso de asistentes de IA:** se usó Claude para apoyar el desarrollo del código y la documentación. Ambos integrantes revisaron, ejecutaron y pueden explicar todo el código entregado.
