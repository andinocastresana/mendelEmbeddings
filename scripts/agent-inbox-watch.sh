#!/usr/bin/env bash
# =========================================
# ID: PHYLO_AGENT_INBOX_WATCH
# VERSION: v1.1
# =========================================
# Escuchador del inbox cross-agente. Vigila la carpeta de mensajes NO leídos de
# un agente y SALE apenas aparece uno, para que la harness de Claude Code re-invoque
# al agente automáticamente (un comando en background re-invoca al agente cuando
# termina). Si no llega nada dentro de WATCH_TIMEOUT, sale con marca de timeout.
#
# Dos modos:
#   - watcher (default): bloquea haciendo polling hasta que llega un mensaje o vence
#     el timeout. Pensado para correr en background.
#   - CHECK_ONCE=1: chequea una sola vez y sale (no bloquea). Pensado para correr
#     al INICIAR sesión: lista lo no leído y devuelve el control enseguida.
#
# Dos capas de comunicación cross-agente (ver AGENTS.md):
#   - AGENTS_HANDOFF.md : log durable y asincrónico (fuente de verdad).
#   - inbox             : mensajes vivos, sincrónicos, efímeros (gitignored).
# Este script opera sobre la capa inbox.
#
# Convención de mensajes:
#   - El emisor escribe un .md en el inbox del DESTINATARIO:
#       _meta/agents/inbox/<destinatario>/<TIMESTAMP>-<emisor>-<slug>.md
#   - Los .md del nivel superior del inbox = NO leídos.
#   - Al consumir un mensaje, MOVERLO a <inbox>/read/ (nunca borrar).
#
# Uso (el script resuelve la raíz del repo solo, no importa el cwd):
#   ./scripts/agent-inbox-watch.sh                      # watcher del inbox de claude
#   AGENT=codex ./scripts/agent-inbox-watch.sh          # watcher del inbox de codex
#   AGENT=codex CHECK_ONCE=1 ./scripts/agent-inbox-watch.sh   # chequeo único al iniciar
#   POLL_INTERVAL=2 WATCH_TIMEOUT=600 ./scripts/agent-inbox-watch.sh
#
# Salida (stdout, parseable):
#   NEW_MESSAGE: <ruta>     (una línea por cada mensaje no leído encontrado)
#   WATCH_TIMEOUT           (modo watcher: no llegó nada dentro del timeout)
#   NO_UNREAD               (modo CHECK_ONCE: no había mensajes sin leer)
#
# Requisitos: bash, coreutils (date). No requiere sudo ni inotify.

set -uo pipefail
shopt -s nullglob

AGENT="${AGENT:-claude}"
POLL_INTERVAL="${POLL_INTERVAL:-3}"     # segundos entre chequeos (modo watcher)
WATCH_TIMEOUT="${WATCH_TIMEOUT:-1800}"  # segundos máx esperando (0 = sin límite)
CHECK_ONCE="${CHECK_ONCE:-0}"           # 1 = chequear una vez y salir (no bloquea)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
INBOX="$REPO_ROOT/_meta/agents/inbox/$AGENT"

if [ ! -d "$INBOX" ]; then
  echo "ERROR: inbox no existe: $INBOX" >&2
  exit 2
fi

# Mensajes NO leídos = .md del nivel superior (read/ queda excluido del glob).
list_unread() {
  local f
  for f in "$INBOX"/*.md; do
    [ -e "$f" ] && printf '%s\n' "$f"
  done
}

report_and_exit() {
  local f
  for f in "$@"; do
    echo "NEW_MESSAGE: $f"
  done
  exit 0
}

# Chequeo inmediato: no perder mensajes llegados antes de armar el watcher.
mapfile -t pending < <(list_unread)
if [ "${#pending[@]}" -gt 0 ]; then
  report_and_exit "${pending[@]}"
fi

# Modo no bloqueante: chequear una vez y salir (útil al iniciar sesión).
if [ "$CHECK_ONCE" = "1" ]; then
  echo "NO_UNREAD"
  exit 0
fi

start="$(date +%s)"
while :; do
  sleep "$POLL_INTERVAL"
  mapfile -t pending < <(list_unread)
  if [ "${#pending[@]}" -gt 0 ]; then
    report_and_exit "${pending[@]}"
  fi
  if [ "$WATCH_TIMEOUT" -gt 0 ]; then
    now="$(date +%s)"
    if [ "$((now - start))" -ge "$WATCH_TIMEOUT" ]; then
      echo "WATCH_TIMEOUT"
      exit 0
    fi
  fi
done
