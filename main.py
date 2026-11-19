"""Entrada de terminal del agente DES (fuera de Jupyter).

Terminal 1:  uv run --locked --extra mcp python servidor_mcp.py
Terminal 2:  uv run --locked --extra mcp python main.py --rol operador --meta 1.0

Argumentos (contexto que fija la APLICACIÓN, no el mensaje al modelo):
  --rol {observador,operador}   quién usa el agente; solo el operador puede registrar.
  --meta MINUTOS                meta de espera promedio en fila (>= 0).
  --limite-personas N           límite de personas del contexto (1-100, por defecto 100).
  --formato {breve,didactico}   estilo de las respuestas.
  --hilo NOMBRE                 thread_id inicial (por defecto uno nuevo).
  --guion ARCHIVO               lee mensajes, comandos y revisiones desde un archivo.
  --etiqueta TEXTO              nombre de los archivos de evidencia.

Durante la conversación:
  /meta X     cambia la meta del contexto y recalcula la recomendación sin simular.
  /hilo X     cambia (o crea) el hilo de conversación.
  /estado     muestra escenario, resultados, recomendación y decisión del hilo.
  /salir      termina (la memoria del proceso se pierde).
Revisión humana (cuando el agente propone registrar):
  interactivo: responde a, e o r;  en guion: !aprobar | !editar <nota> | !rechazar <motivo>
"""

import argparse
import asyncio
import json
import sys
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv

RAIZ = Path(__file__).resolve().parent
load_dotenv(RAIZ / ".env", override=True)

from langchain.messages import HumanMessage  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from langgraph.types import Command  # noqa: E402

from oficina_des.agente import (  # noqa: E402
    ContextoOperacion, acciones_pendientes, aplicar_meta, crear_agente_des, crear_modelo_openrouter,
)
from oficina_des.experimento import validar_meta  # noqa: E402
from oficina_des.trazas import CallbackTraza, Traza  # noqa: E402

LIMITE_RECURSION = 40


class Conversacion:
    """Ciclo de conversación reutilizable por main.py y por las verificaciones."""

    def __init__(self, agente, contexto: ContextoOperacion, traza: Traza, hilo: str, salida=print):
        """Prepara la conversación: agente, contexto de la aplicación, traza e hilo inicial."""
        self.agente, self.ctx, self.traza, self.salida = agente, contexto, traza, salida
        self.callback = CallbackTraza(traza)
        self.cambiar_hilo(hilo)
        traza.evento("contexto", **{k: getattr(contexto, k) for k in ("rol", "formato", "meta_espera", "limite_personas")})

    def config(self):
        """Configuración de cada invocación: thread_id, límite de pasos y callback de trazas."""
        return {"configurable": {"thread_id": self.hilo}, "recursion_limit": LIMITE_RECURSION,
                "callbacks": [self.callback]}

    def cambiar_hilo(self, hilo: str):
        """Cambia (o crea) el hilo de conversación; otro hilo no comparte estado."""
        self.hilo = hilo
        self.traza.hilo = hilo
        self.pendientes = []
        self.traza.evento("hilo", thread_id=hilo)
        self.traza.md_linea(f"## Hilo `{hilo}`")

    async def estado(self) -> dict:
        """Valores actuales del estado del hilo (escenario, resultados, recomendación, decisión)."""
        return (await self.agente.aget_state({"configurable": {"thread_id": self.hilo}})).values

    async def _ejecutar(self, entrada):
        """Invoca al agente, maneja errores del proveedor y muestra la respuesta o la revisión pendiente."""
        try:
            r = await self.agente.ainvoke(entrada, config=self.config(), context=self.ctx)
        except Exception as e:  # proveedor caído, límite de llamadas, clave inválida...
            texto = describir_error_proveedor(e)
            self.traza.evento("error_proveedor", clase=type(e).__name__, detalle=str(e)[:500])
            self.salida(f"[Error] {texto}")
            self.traza.md_linea(f"**Error:** {texto}\n")
            self.pendientes = []
            return None
        self.pendientes = acciones_pendientes(r)
        if self.pendientes:
            self.traza.evento("interrupcion", acciones=self.pendientes,
                              decision_antes_de_revisar=r.get("decision"))
            self.salida("\n[Revisión humana requerida] Acción propuesta:")
            for a in self.pendientes:
                self.salida(f"  {a['name']}({json.dumps(a['args'], ensure_ascii=False)})")
            self.salida(f"  Decisión registrada en este momento: {r.get('decision')}")
            self.traza.md_linea("**Revisión pendiente:** `" + json.dumps(self.pendientes[0]['args'], ensure_ascii=False) + "`\n")
        else:
            texto = r["messages"][-1].text
            self.salida(f"\nAgente: {texto}\n")
            self.traza.evento("respuesta_final", texto=texto)
            self.traza.md_linea(f"**Agente:**\n\n{texto}\n")
        return r

    async def mensaje(self, texto: str):
        """Envía un mensaje del usuario al agente en el hilo actual."""
        if self.pendientes:
            self.salida("Primero responde la revisión pendiente (a / e / r).")
            return None
        self.traza.evento("usuario", mensaje=texto)
        self.traza.md_linea(f"**Usuario:** {texto}\n")
        return await self._ejecutar({"messages": [HumanMessage(content=texto)]})

    async def revisar(self, eleccion: str, texto: str = ""):
        """eleccion: aprobar | editar | rechazar. La reanudación usa el MISMO hilo."""
        if not self.pendientes:
            self.salida("No hay ninguna revisión pendiente.")
            return None
        accion = self.pendientes[0]
        if eleccion == "aprobar":
            decision = {"type": "approve"}
        elif eleccion == "editar":
            if not texto.strip():
                self.salida("Para editar escribe la nota corregida.")
                return None
            decision = {"type": "edit", "edited_action": {"name": accion["name"], "args": {**accion["args"], "nota": texto.strip()}}}
        elif eleccion == "rechazar":
            decision = {"type": "reject", "message": (texto.strip() or "La persona revisora rechazó el registro.") + " No reintentes."}
        else:
            self.salida("Elige aprobar, editar o rechazar.")
            return None
        self.traza.evento("revision_humana", eleccion=eleccion, respuesta=decision)
        self.traza.md_linea(f"**Revisión humana:** {eleccion} {texto}\n")
        r = await self._ejecutar(Command(resume={"decisions": [decision]}))
        estado = await self.estado()
        self.traza.evento("decision_despues_de_revisar", decision=estado.get("decision"))
        self.salida(f"Decisión en el estado: {json.dumps(estado.get('decision'), ensure_ascii=False)}")
        return r

    async def cambiar_meta(self, meta: float):
        """Cambia la meta del contexto y recalcula la recomendación sin simular; invalida la decisión."""
        if self.pendientes:
            self.salida("Hay una revisión pendiente: respóndela antes de cambiar la meta.")
            return
        self.ctx = replace(self.ctx, meta_espera=validar_meta(meta))
        cambio = aplicar_meta(await self.estado(), self.ctx.meta_espera)
        if cambio:
            await self.agente.aupdate_state({"configurable": {"thread_id": self.hilo}}, cambio)
        self.traza.evento("meta_cambiada", meta=self.ctx.meta_espera, actualizacion=cambio)
        self.traza.md_linea(f"**Contexto:** meta = {self.ctx.meta_espera} min → {cambio}\n")
        self.salida(f"Meta = {self.ctx.meta_espera} min. Recomendación recalculada: "
                    f"{cambio['recomendacion'] if cambio else 'sin resultados o sin cambio'}; decisión invalidada.")

    async def mostrar_estado(self):
        """Imprime y devuelve un resumen del estado del hilo."""
        e = await self.estado()
        res = e.get("resultados") or {}
        vista = {"hilo": self.hilo, "contexto": {"rol": self.ctx.rol, "meta_espera": self.ctx.meta_espera},
                 "escenario": e.get("escenario"),
                 "agregados": [(a["parametros"]["servidores"], round(a["espera_promedio"], 4),
                                round(a["desviacion_estandar_espera"], 4)) for a in res.get("agregados", [])],
                 "archivos": res.get("archivos"), "recomendacion": e.get("recomendacion"),
                 "meta_recomendacion": e.get("meta_recomendacion"), "decision": e.get("decision")}
        self.salida(json.dumps(vista, ensure_ascii=False, indent=2))
        return vista

    async def cerrar(self):
        """Guarda el estado final del hilo en la traza y en la transcripción."""
        vista = await self.mostrar_estado()
        self.traza.evento("estado_final", **vista)
        self.traza.md_linea("## Estado final\n\n```json\n" + json.dumps(vista, ensure_ascii=False, indent=2) + "\n```")


def describir_error_proveedor(e: Exception) -> str:
    """Traduce errores del proveedor del modelo (clave, límites, modelo inexistente, red, pasos) a mensajes claros."""
    nombre, texto = type(e).__name__, str(e)
    if "ModelCallLimit" in nombre or "limit" in texto.lower() and "call" in texto.lower():
        return "Se alcanzó el límite de llamadas al modelo en este turno; reformula la solicitud."
    if "401" in texto or "Authentication" in nombre:
        return "La API rechazó la clave (401). Revisa OPENROUTER_API_KEY en .env."
    if "429" in texto or "RateLimit" in nombre:
        return "Límite de solicitudes del proveedor (429). Espera un minuto o cambia OPENROUTER_MODEL."
    if "404" in texto or "NotFound" in nombre:
        return "El modelo no existe o no admite herramientas (404). Cambia OPENROUTER_MODEL."
    if "Timeout" in nombre or "Connection" in nombre:
        return "No hubo conexión con el proveedor del modelo."
    if "Recursion" in nombre:
        return "El agente superó el límite de pasos del turno."
    return f"{nombre}: {texto[:300]}"


async def ejecutar_lineas(conv: Conversacion, lineas, interactivo: bool):
    """Procesa mensajes, comandos y revisiones (desde teclado o guion)."""
    for linea in lineas:
        linea = linea.strip()
        if not linea or linea.startswith("#"):
            continue
        if not interactivo:
            conv.salida(f"\n> {linea}")
        if linea == "/salir":
            break
        if linea == "/estado":
            await conv.mostrar_estado()
        elif linea.startswith("/meta "):
            try:
                await conv.cambiar_meta(float(linea.split(maxsplit=1)[1]))
            except ValueError as e:
                conv.salida(f"Meta inválida: {e}")
        elif linea.startswith("/hilo "):
            conv.cambiar_hilo(linea.split(maxsplit=1)[1])
            conv.salida(f"Hilo actual: {conv.hilo}")
        elif linea.startswith("!"):
            partes = linea[1:].split(maxsplit=1)
            await conv.revisar(partes[0], partes[1] if len(partes) > 1 else "")
        else:
            await conv.mensaje(linea)
        while interactivo and conv.pendientes:
            op = input("[a]probar / [e]ditar nota / [r]echazar: ").strip().lower()[:1]
            if op == "a":
                await conv.revisar("aprobar")
            elif op == "e":
                await conv.revisar("editar", input("Nota corregida: "))
            elif op == "r":
                await conv.revisar("rechazar", input("Motivo (opcional): "))
    await conv.cerrar()


def entrada_interactiva():
    """Genera las líneas que escribe el usuario hasta Ctrl+D o Ctrl+C."""
    while True:
        try:
            yield input("Usted> ")
        except (EOFError, KeyboardInterrupt):
            return


async def main():
    """Lee los argumentos, crea el contexto, el modelo, el agente y la traza, y ejecuta la conversación."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rol", choices=["observador", "operador"], default="observador")
    ap.add_argument("--meta", type=float, default=1.0)
    ap.add_argument("--limite-personas", type=int, default=100)
    ap.add_argument("--formato", choices=["breve", "didactico"], default="breve")
    ap.add_argument("--hilo", default=None)
    ap.add_argument("--guion", type=Path, default=None)
    ap.add_argument("--etiqueta", default="conversacion")
    a = ap.parse_args()
    try:
        ctx = ContextoOperacion(rol=a.rol, formato=a.formato, meta_espera=a.meta, limite_personas=a.limite_personas)
        base, revision, info = crear_modelo_openrouter()
    except (ValueError, RuntimeError) as e:
        sys.exit(f"Configuración inválida: {e}")
    traza = Traza(a.etiqueta, origen="modelo real")
    traza.evento("configuracion_modelo", **info)
    agente = crear_agente_des(base, revision, modelo_especialista=base, checkpointer=InMemorySaver(),
                              registrar_evento=traza.evento)
    conv = Conversacion(agente, ctx, traza, a.hilo or f"hilo-{uuid4().hex[:8]}")
    print(f"Agente DES · rol={ctx.rol} · meta={ctx.meta_espera} min · hilo={conv.hilo} · modelo={info['modelo']}")
    print(f"Evidencia: {traza.ruta.relative_to(RAIZ)}  (/salir para terminar)")
    if a.guion:
        await ejecutar_lineas(conv, a.guion.read_text(encoding="utf-8").splitlines(), interactivo=False)
    else:
        await ejecutar_lineas(conv, entrada_interactiva(), interactivo=True)
    print(f"\nEvidencia guardada en {traza.ruta.relative_to(RAIZ)} y {traza.md.relative_to(RAIZ)}")


if __name__ == "__main__":
    asyncio.run(main())
