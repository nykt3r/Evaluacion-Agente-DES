"""Evidencia de cada conversación: JSONL de eventos y transcripción en Markdown.

Se registran mensajes del usuario, llamadas a herramientas (coordinador y especialista),
argumentos, resultados MCP, recurso leído, interrupciones, revisiones humanas, errores y
estado final. Sin claves ni datos personales.
"""

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler

from .experimento import carpeta_evidencias


class Traza:
    """Archivo JSONL de eventos y transcripción Markdown de una conversación."""
    def __init__(self, etiqueta: str, origen: str, carpeta: Path | None = None):
        """Crea los archivos de la conversación en evidencias/conversaciones con la etiqueta y la fecha."""
        carpeta = carpeta or carpeta_evidencias() / "conversaciones"
        carpeta.mkdir(parents=True, exist_ok=True)
        marca = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.ruta = carpeta / f"{etiqueta}_{marca}.jsonl"
        self.md = self.ruta.with_suffix(".md")
        self.origen = origen          # "modelo real" o "modelo de ensayo"
        self.hilo = None
        self._lock = threading.Lock()
        self._lineas_md = [f"# Conversación `{etiqueta}` ({origen})", "", f"Inicio: {marca}", ""]

    def evento(self, tipo: str, **datos):
        """Agrega una línea JSON con marca de tiempo, hilo, origen y tipo de evento."""
        reg = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "hilo": self.hilo,
               "origen": self.origen, "tipo": tipo, **datos}
        with self._lock, open(self.ruta, "a", encoding="utf-8") as f:
            f.write(json.dumps(reg, ensure_ascii=False, default=str) + "\n")

    def md_linea(self, texto: str = ""):
        """Agrega una línea a la transcripción Markdown y la guarda."""
        self._lineas_md.append(texto)
        self.md.write_text("\n".join(self._lineas_md) + "\n", encoding="utf-8")


class CallbackTraza(BaseCallbackHandler):
    """Captura llamadas a herramientas y respuestas del modelo (incluido el especialista)."""

    def __init__(self, traza: Traza):
        """Callback que escribe en la traza; guarda el último modelo usado."""
        self.traza = traza
        self.modelo = None

    def on_tool_start(self, serialized, input_str, *, run_id: UUID, inputs=None, **kw):
        """Registra el nombre y los argumentos de cada llamada a herramienta."""
        self.traza.evento("llamada_herramienta", herramienta=(serialized or {}).get("name"),
                          argumentos=inputs if inputs is not None else input_str, run_id=str(run_id))

    def on_tool_end(self, output, *, run_id: UUID, **kw):
        """Registra el resultado de la herramienta (o la actualización de estado de un Command)."""
        contenido = getattr(output, "content", output)
        if hasattr(output, "update"):            # Command de LangGraph
            contenido = {k: v for k, v in output.update.items() if k != "messages"} | {
                "mensajes": [getattr(m, "content", str(m)) for m in output.update.get("messages", [])]}
        texto = contenido if isinstance(contenido, str) else json.dumps(contenido, ensure_ascii=False, default=str)
        self.traza.evento("resultado_herramienta", run_id=str(run_id), resultado=texto[:4000])

    def on_tool_error(self, error, *, run_id: UUID, **kw):
        """Registra los errores lanzados por una herramienta."""
        self.traza.evento("error_herramienta", run_id=str(run_id), error=repr(error)[:500])

    def on_llm_end(self, response, **kw):
        """Registra el modelo usado, las llamadas que solicitó y su texto."""
        try:
            msg = response.generations[0][0].message
            meta = getattr(msg, "response_metadata", {}) or {}
            self.modelo = meta.get("model_name") or meta.get("model") or self.modelo
            self.traza.evento("respuesta_modelo", modelo=self.modelo,
                              llamadas_solicitadas=[{"nombre": t["name"], "args": t["args"]}
                                                    for t in getattr(msg, "tool_calls", []) or []],
                              texto=(msg.text or "")[:2000] if hasattr(msg, "text") else None)
        except Exception:  # la traza nunca rompe la conversación
            pass
