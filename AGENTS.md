# AGENTS.md — instrucciones para agentes de IA

Este proyecto lo trabajan varios agentes (Claude Code, Codex, y futuros). Estas
reglas aplican a **todos**.

## Convenciones del proyecto

Las convenciones generales viven en `CLAUDE.md` y aplican a cualquier agente:
- **No commitear** salvo que el usuario lo pida explícitamente. Nunca commitear
  de forma automática al terminar una tarea.
- Cada commit lleva una entrada en `DEVLOG.md` (hash, título de una línea, IDs de
  tarea, detalle). Prefijar el título con tu tag de agente, p.ej. `[codex]`.
- Al iniciar sesión, leer `TAREAS_PENDIENTES.md` y mencionar las tareas
  pendientes / en progreso.

## Entorno de desarrollo

El entorno es **por-shell, no se hereda** entre sesiones de agentes. Cada agente
debe activarlo en su propio shell antes de correr nada:

- **Web / cliente (Vite)**: requiere Node `>=20.19`. Con nvm: `nvm use` (toma el
  `.nvmrc` de la raíz → Node 20). Sin nvm: asegurar Node 20+ instalado. **Node 16
  rompe Vite.** Dev desde `client/` con `npm run dev` (preferentemente vía
  `scripts/dev-monitored.sh`, que loguea CPU/temp).
- **Motor Python**: conda env `face-sim` (Python 3.11) con `insightface`,
  `onnxruntime`, `cv2`, `numpy`. Activar con `conda activate face-sim`. El
  `python3` del sistema (3.9) NO tiene estas deps.
- Si tu shell no tiene nvm/conda disponibles, anotalo en `AGENTS_HANDOFF.md` para
  resolver juntos la ruta de entorno.

## Canal de coordinación cross-agente

Hay dos capas, con propósitos distintos:

**1. `AGENTS_HANDOFF.md`** (raíz) — log **durable y asincrónico**, fuente de verdad
de la coordinación:
- **Al iniciar**: leé las últimas entradas para saber qué dejó el agente anterior.
- **Al cerrar**: agregá una entrada arriba de todo con el formato del propio
  archivo.

**2. Inbox sincrónico** (`_meta/agents/inbox/<destinatario>/`) — mensajes **vivos**
para ida-y-vuelta en tiempo real cuando dos sesiones están abiertas a la vez
(gitignored, efímero):
- El emisor escribe un `.md` en el inbox del DESTINATARIO, **uno por mensaje**:
  `_meta/agents/inbox/<destinatario>/<TIMESTAMP>-<emisor>-<slug>.md`.
- Los `.md` del nivel superior = NO leídos. Al consumir uno, **moverlo** a
  `<inbox>/read/` (nunca borrar).
- **Al iniciar sesión**, revisá tu inbox una vez, **no bloqueante**:
  `AGENT=<vos> CHECK_ONCE=1 scripts/agent-inbox-watch.sh` lista lo no leído y sale.
  Atendé los mensajes y movélos a `read/`.
- Para escuchar en tiempo real mientras tu sesión está abierta, corré en background
  `scripts/agent-inbox-watch.sh` (con `AGENT=codex` para vigilar el inbox de Codex):
  sale apenas llega un mensaje, sin polling manual. (Nota: Codex no se re-invoca
  solo al terminar un comando en background; del lado de Codex el watcher sirve
  corrido inline durante un tramo de trabajo activo.)
- Lo importante y durable de un intercambio se resume después en `AGENTS_HANDOFF.md`.

## Scratch privado

Tu borrador/scratch va en `_meta/agents/<tu-nombre>/` (gitignored). **No escribas
en el rincón de otro agente** ni dependas de leerlo: lo que el otro necesita saber
se cura en `AGENTS_HANDOFF.md`.
