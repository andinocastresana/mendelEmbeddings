#!/usr/bin/env bash
# =========================================
# ID: PHYLO_TEST_MONITORED
# VERSION: v1.0
# =========================================
# Wrapper de monitoreo de recursos para CUALQUIER comando de test/script que
# termina (a diferencia de `dev-monitored.sh`, que envuelve un dev server que
# corre indefinidamente). Samplea CPU%/temp_max cada N segundos durante la
# corrida, loguea cada muestra a `.test-resources.log` (gitignored) y, al
# terminar el comando, imprime un RESUMEN cuantitativo (muestras, cpu/temp
# avg+max) atribuible a esa corrida.
#
# Motivación: regla del usuario (2026-05-23) — toda corrida de test automático
# (browser o Python) va monitoreada + con log. Para browser ya estaba
# `dev-monitored.sh`; este cubre Python (pytest, smoke, spikes de calibración
# #6) y cualquier comando que no sea un server. Hermano de `dev-monitored.sh`:
# reusa la misma lógica de sampleo (vmstat/sensors/ps).
#
# Uso:
#   ./scripts/test-monitored.sh python3 scripts/spike_kinfacew_embeddings.py
#   ./scripts/test-monitored.sh python3 tests/smoke/test_paso_7_viz.py
#   SAMPLE_INTERVAL=2 TEMP_THRESHOLD=85 ./scripts/test-monitored.sh <cmd...>
#
# El exit code del wrapper = exit code del comando envuelto (para CI / &&).
#
# Requisitos: bash, vmstat (procps), sensors (lm-sensors). No requiere sudo.

set -uo pipefail

CPU_THRESHOLD="${CPU_THRESHOLD:-80}"        # % CPU para WARN sostenido
TEMP_THRESHOLD="${TEMP_THRESHOLD:-80}"      # °C core max para WARN sostenido
SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-3}"     # segundos entre muestras (más fino que dev: los tests son cortos)
SUSTAIN_SAMPLES="${SUSTAIN_SAMPLES:-3}"     # muestras seguidas sobre umbral para WARN

LOG_FILE="${TEST_LOG_FILE:-.test-resources.log}"

if [ "$#" -eq 0 ]; then
  echo "uso: $0 <comando> [args...]" >&2
  exit 2
fi

# Colores sólo si stderr es tty.
if [ -t 2 ]; then
  C_RED='\033[1;31m'; C_YELLOW='\033[1;33m'; C_GREEN='\033[1;32m'; C_DIM='\033[2m'; C_RESET='\033[0m'
else
  C_RED=''; C_YELLOW=''; C_GREEN=''; C_DIM=''; C_RESET=''
fi

# Log por-corrida (sólo las muestras de ESTA invocación) para el resumen final;
# el LOG_FILE persistente acumula append-only entre corridas.
RUN_LOG="$(mktemp)"

# -------------------------------------
# Muestreadores (idénticos a dev-monitored.sh).
# -------------------------------------
get_cpu_pct() {
  local idle
  idle=$(vmstat 1 2 2>/dev/null | tail -1 | awk '{print $15}')
  [ -z "$idle" ] && idle=100
  echo $((100 - idle))
}
get_temp_max() {
  local t
  t=$(sensors -A 2>/dev/null \
      | awk '/^Core [0-9]+:/ { v=$3; gsub(/[+°C]/,"",v); print v+0 }' \
      | sort -g | tail -1)
  printf "%.0f" "${t:-0}"
}
get_top_offenders() {
  ps -eo pcpu,comm --sort=-pcpu --no-headers 2>/dev/null \
    | head -3 | awk '{printf "%s(%.0f%%) ", $2, $1}'
}

# -------------------------------------
# Loop de sampling: muestrea YA (t=0) y luego cada intervalo, para que aun
# corridas de pocos segundos tengan ≥1 muestra. Escribe a ambos logs.
# -------------------------------------
sample_loop() {
  local cpu_high=0 temp_high=0 first=1
  while true; do
    if [ "$first" -eq 1 ]; then first=0; else sleep "$SAMPLE_INTERVAL"; fi
    local cpu temp ts line
    cpu=$(get_cpu_pct); temp=$(get_temp_max); ts=$(date '+%Y-%m-%d %H:%M:%S')
    line="$ts cpu=${cpu}% temp=${temp}°C"
    echo "$line" >> "$LOG_FILE"
    echo "$line" >> "$RUN_LOG"
    if [ "$cpu" -gt "$CPU_THRESHOLD" ]; then
      cpu_high=$((cpu_high + 1))
      [ "$cpu_high" -eq "$SUSTAIN_SAMPLES" ] && \
        echo -e "${C_RED}[test-monitored] WARN CPU ${cpu}% sostenido — top: $(get_top_offenders)${C_RESET}" >&2
    else cpu_high=0; fi
    if [ "$temp" -gt "$TEMP_THRESHOLD" ]; then
      temp_high=$((temp_high + 1))
      [ "$temp_high" -eq "$SUSTAIN_SAMPLES" ] && \
        echo -e "${C_YELLOW}[test-monitored] WARN temp core_max ${temp}°C sostenido${C_RESET}" >&2
    else temp_high=0; fi
  done
}

sample_loop &
SAMPLER_PID=$!

cleanup() { kill "$SAMPLER_PID" 2>/dev/null || true; rm -f "$RUN_LOG" 2>/dev/null || true; }
trap cleanup EXIT

echo -e "${C_DIM}[test-monitored] muestreo cada ${SAMPLE_INTERVAL}s → ${LOG_FILE} · comando: $*${C_RESET}" >&2

# -------------------------------------
# Correr el comando envuelto, preservando su exit code.
# -------------------------------------
"$@"
CMD_EXIT=$?

# Una muestra extra al cierre para capturar el pico de cola.
sleep 1
kill "$SAMPLER_PID" 2>/dev/null || true

# -------------------------------------
# Resumen cuantitativo de ESTA corrida (regla: no sólo "pasó").
# -------------------------------------
summary=$(awk '
  match($0, /cpu=([0-9]+)%/, c) && match($0, /temp=([0-9]+)/, t) {
    n++; cpu=c[1]; tmp=t[1];
    sc+=cpu; if(cpu>mc)mc=cpu;
    st+=tmp; if(tmp>mt)mt=tmp;
  }
  END { if(n>0) printf "muestras=%d  cpu_avg=%d%% cpu_max=%d%%  temp_avg=%d°C temp_max=%d°C", n, sc/n, mc, st/n, mt; else printf "sin muestras"; }
' "$RUN_LOG")

echo -e "${C_GREEN}[test-monitored] resumen recursos: ${summary} (exit=${CMD_EXIT})${C_RESET}" >&2

exit "$CMD_EXIT"
