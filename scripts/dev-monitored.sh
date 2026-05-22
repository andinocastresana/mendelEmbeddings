#!/usr/bin/env bash
# =========================================
# ID: PHYLO_DEV_MONITORED
# VERSION: v1.0
# =========================================
# Wrapper de `npm run dev` con muestreo pasivo de recursos cada N segundos.
# Imprime WARN inline cuando CPU% o temp_max sostenidos pasan umbral, y
# logguea cada muestra a .dev-resources.log (gitignored).
#
# Pensado como complemento del dashboard `monitor.sh`:
#   - monitor.sh   : vista visual cuando querés mirar
#   - dev-monitored: avisos pasivos mientras desarrollás
#
# Uso:
#   ./scripts/dev-monitored.sh
#   CPU_THRESHOLD=90 TEMP_THRESHOLD=85 ./scripts/dev-monitored.sh
#   SAMPLE_INTERVAL=10 ./scripts/dev-monitored.sh
#
# Salida: Ctrl+C mata el sampler y el dev server.
#
# Requisitos: bash, vmstat (procps), sensors (lm-sensors), bc. No requiere sudo.

set -eo pipefail

CPU_THRESHOLD="${CPU_THRESHOLD:-80}"        # % CPU
TEMP_THRESHOLD="${TEMP_THRESHOLD:-80}"      # °C core max
SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-5}"     # segundos entre muestras
SUSTAIN_SAMPLES="${SUSTAIN_SAMPLES:-3}"     # muestras seguidas sobre umbral para disparar WARN

LOG_FILE=".dev-resources.log"

# Paths: este script vive en <proyecto>/scripts/; el dev server vive en <proyecto>/client/.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CLIENT_DIR="$(cd "$SCRIPT_DIR/../client" && pwd)"

# Colores ANSI para que los WARN se vean (sólo si stdout es tty).
if [ -t 2 ]; then
  C_RED='\033[1;31m'
  C_YELLOW='\033[1;33m'
  C_DIM='\033[2m'
  C_RESET='\033[0m'
else
  C_RED=''
  C_YELLOW=''
  C_DIM=''
  C_RESET=''
fi

cd "$CLIENT_DIR"

echo -e "${C_DIM}[dev-monitored] umbrales: CPU>${CPU_THRESHOLD}% temp>${TEMP_THRESHOLD}°C sostenido ${SUSTAIN_SAMPLES} muestras (cada ${SAMPLE_INTERVAL}s)${C_RESET}"
echo -e "${C_DIM}[dev-monitored] log:      ${CLIENT_DIR}/${LOG_FILE}${C_RESET}"

# -------------------------------------
# Muestreadores de recursos.
# Diseñados para ser locale-agnostic y robustos a parsing irregular.
# -------------------------------------

get_cpu_pct() {
  # vmstat 1 2 toma 2 samples 1s apart; usamos la segunda (columna 15 = idle).
  local idle
  idle=$(vmstat 1 2 2>/dev/null | tail -1 | awk '{print $15}')
  [ -z "$idle" ] && idle=100
  echo $((100 - idle))
}

get_temp_max() {
  # Max de las temps "Core N: +XX.X°C" reportadas por sensors.
  local t
  t=$(sensors -A 2>/dev/null \
      | awk '/^Core [0-9]+:/ { v=$3; gsub(/[+°C]/,"",v); print v+0 }' \
      | sort -g | tail -1)
  printf "%.0f" "${t:-0}"
}

get_top_offenders() {
  # Top procesos por %CPU, primeros 3, formato "name(pct%)".
  ps -eo pcpu,comm --sort=-pcpu --no-headers 2>/dev/null \
    | head -3 \
    | awk '{printf "%s(%.0f%%) ", $2, $1}'
}

# -------------------------------------
# Loop de sampling en background.
# Cada muestra se appendea al log; los WARN van a stderr para no contaminar el
# stdout del dev server.
# -------------------------------------

sample_loop() {
  local cpu_high_count=0
  local temp_high_count=0
  while true; do
    sleep "$SAMPLE_INTERVAL"

    local cpu_used temp_max ts
    cpu_used=$(get_cpu_pct)
    temp_max=$(get_temp_max)
    ts=$(date '+%Y-%m-%d %H:%M:%S')

    echo "$ts cpu=${cpu_used}% temp=${temp_max}°C" >> "$LOG_FILE"

    # CPU sustain check.
    if [ "$cpu_used" -gt "$CPU_THRESHOLD" ]; then
      cpu_high_count=$((cpu_high_count + 1))
      if [ "$cpu_high_count" -eq "$SUSTAIN_SAMPLES" ]; then
        local top
        top=$(get_top_offenders)
        echo -e "\n${C_RED}[dev-monitored $(date '+%H:%M:%S')] WARN${C_RESET}  CPU ${cpu_used}% sostenido $((SUSTAIN_SAMPLES * SAMPLE_INTERVAL))s — top: ${top}" >&2
      fi
    else
      cpu_high_count=0
    fi

    # Temp sustain check.
    if [ "$temp_max" -gt "$TEMP_THRESHOLD" ]; then
      temp_high_count=$((temp_high_count + 1))
      if [ "$temp_high_count" -eq "$SUSTAIN_SAMPLES" ]; then
        echo -e "\n${C_YELLOW}[dev-monitored $(date '+%H:%M:%S')] WARN${C_RESET}  Temp core_max ${temp_max}°C sostenido $((SUSTAIN_SAMPLES * SAMPLE_INTERVAL))s" >&2
      fi
    else
      temp_high_count=0
    fi
  done
}

sample_loop &
SAMPLER_PID=$!

# Cleanup: matar sampler al salir (Ctrl+C, npm dev cerrado, etc.).
cleanup() {
  kill "$SAMPLER_PID" 2>/dev/null || true
}
trap cleanup INT TERM EXIT

echo -e "${C_DIM}[dev-monitored] arrancando npm run dev...${C_RESET}"
npm run dev
