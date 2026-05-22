#!/usr/bin/env bash
# =========================================
# ID: PHYLO_HEAT_EXPERIMENT_BASH
# VERSION: v1.0
# =========================================
# Orquestador del experimento de temperatura del Track 2b (Tarea #26 paso 2,
# 2026-05-22). Mide CPU% y temp_max del sistema durante 7 fases controladas:
#
#   0-baseline-no-browser       : dev server arriba, sin browser     (30s)
#   1-comparator-default-idle   : tab Comparador (default), sin click (30s)
#   2-genealogy-empty           : tab Árbol genealógico vacío         (30s)
#   3-genealogy-with-photo      : árbol con 1 persona + 1 foto        (30s)
#   4-comparator-after-compare  : Comparador con MediaPipe+ONNX init  (30s)
#   5-genealogy-after-gpu-init  : tab Árbol con GPU ya cargado        (30s)
#   6-tab-closed                : tab del SPA cerrado                  (30s)
#
# La hipótesis a falsar: GenealogyTree por sí solo NO calienta. Si fase 2/3
# ≈ fase 0, queda confirmado. Si fase 5 > fase 2/3, también queda confirmado
# que el calor de fase 5 viene del leak del Comparator (Tarea #27), no del
# árbol.
#
# Uso:
#   ./scripts/heat-experiment.sh
#
# Salida:
#   client/.heat-experiment.log  : CSV con timestamp,phase,cpu,temp por muestra
#   stdout                       : tabla resumen por fase al final

set -uo pipefail

PROJ_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CLIENT_DIR="$PROJ_DIR/client"
LOG_FILE="$CLIENT_DIR/.heat-experiment.log"
PHASE_FILE="$CLIENT_DIR/.heat-experiment-current-phase.txt"
VITE_LOG="/tmp/heat-exp-vite.log"

# Cleanup state previo
rm -f "$LOG_FILE" "$PHASE_FILE" "$VITE_LOG"
echo "0-baseline-no-browser" > "$PHASE_FILE"
echo "ts,phase,cpu,temp" > "$LOG_FILE"

# Mata vite previo si existe
pkill -f 'vite' 2>/dev/null && sleep 1 || true

echo "[heat-exp] arrancando vite dev en background..."
(cd "$CLIENT_DIR" && npm run dev > "$VITE_LOG" 2>&1) &
VITE_PID=$!

# Espera Vite ready
echo "[heat-exp] esperando que vite levante..."
until grep -q 'Local:' "$VITE_LOG" 2>/dev/null; do
  sleep 0.5
  # Cancelar si vite muere
  if ! kill -0 $VITE_PID 2>/dev/null; then
    echo "[heat-exp] ERROR: vite murió antes de levantar. Log:" >&2
    cat "$VITE_LOG" >&2
    exit 1
  fi
done
echo "[heat-exp] vite ready"

# Sampler en background — escribe CSV con la fase actual leída del file.
sample_loop() {
  while true; do
    sleep 5
    local idle cpu temp ts phase
    idle=$(vmstat 1 2 2>/dev/null | tail -1 | awk '{print $15}')
    cpu=$((100 - idle))
    temp=$(sensors -A 2>/dev/null | awk '/^Core/ {v=$3; gsub(/[+°C]/,"",v); print v+0}' | sort -g | tail -1)
    temp=${temp:-0}
    ts=$(date '+%H:%M:%S')
    phase=$(cat "$PHASE_FILE" 2>/dev/null || echo "unknown")
    echo "$ts,$phase,${cpu},${temp}" >> "$LOG_FILE"
  done
}

sample_loop &
SAMPLER_PID=$!

cleanup() {
  echo "[heat-exp] limpiando procesos..."
  kill $SAMPLER_PID 2>/dev/null || true
  kill $VITE_PID 2>/dev/null || true
  pkill -f 'vite' 2>/dev/null || true
  rm -f "$PHASE_FILE"
}
trap cleanup INT TERM EXIT

echo "[heat-exp] === Phase 0: baseline (sin browser, 30s) ==="
sleep 30

echo "[heat-exp] === Phases 1-6: lanzando Playwright (modo headed) ==="
echo "[heat-exp] Playwright actualiza $PHASE_FILE entre fases; el sampler asocia cada muestra a la fase actual."
cd "$CLIENT_DIR" && node scripts/heat-experiment.mjs
PLAYWRIGHT_EXIT=$?
cd "$PROJ_DIR"

if [ $PLAYWRIGHT_EXIT -ne 0 ]; then
  echo "[heat-exp] WARN: Playwright salió con $PLAYWRIGHT_EXIT (igual analizamos lo que se haya capturado)" >&2
fi

echo "[heat-exp] experimento terminado. Analizando $LOG_FILE..."
echo ""

# Análisis: agg por fase. awk porque está garantizado en cualquier sistema.
awk -F',' '
NR==1 { next }  # skip header
{
  phase = $2
  cpu = $3 + 0
  temp = $4 + 0
  count[phase]++
  cpu_sum[phase] += cpu
  if (cpu > cpu_max[phase]) cpu_max[phase] = cpu
  temp_sum[phase] += temp
  if (temp > temp_max[phase]) temp_max[phase] = temp
}
END {
  printf "%-32s %5s %10s %10s %12s %12s\n", "Phase", "n", "cpu_avg", "cpu_max", "temp_avg", "temp_max"
  printf "%-32s %5s %10s %10s %12s %12s\n", "-----", "-", "-------", "-------", "--------", "--------"
  n = asorti(count, sorted)
  for (i = 1; i <= n; i++) {
    p = sorted[i]
    printf "%-32s %5d %9.1f%% %9d%% %10.1f°C %10d°C\n",
      p, count[p], cpu_sum[p]/count[p], cpu_max[p], temp_sum[p]/count[p], temp_max[p]
  }
}' "$LOG_FILE"

echo ""
echo "[heat-exp] log completo (CSV) en: $LOG_FILE"
