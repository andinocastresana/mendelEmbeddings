// =========================================
// ID: PHYLOFACE_CALIBRATION_CHART
// VERSION: v1.0
// =========================================
// Histograma SVG reutilizable de las distribuciones calibradas (Tarea #6,
// Fase B). Dibuja kin (con parentesco) vs non-kin (sin parentesco) como barras
// semitransparentes superpuestas, normalizadas cada una a su propia masa
// (fracción), para comparar las FORMAS aunque n_pos ≠ n_neg.
//
// Opcionalmente marca:
//   - `marker`: una línea vertical sólida en el valor del usuario (su cosine).
//   - `threshold`: una línea vertical punteada en el umbral de Youden.
//
// Sin dependencias de graficado: SVG plano. Lo consumen `CalibrationModal`
// (un histograma con el cosine propio) y `CalibrationTab` (grilla de todos).

import type { Histogram } from './lib/calibration';

interface HistogramChartProps {
  histogram: Histogram;
  marker?: number | null;
  threshold?: number | null;
  width?: number;
  height?: number;
  posLabel?: string;
  negLabel?: string;
  /** Oculta la leyenda (útil en grillas densas). */
  compact?: boolean;
}

const POS_COLOR = '#1a73e8'; // kin
const NEG_COLOR = '#d23f3f'; // non-kin

export default function HistogramChart({
  histogram,
  marker = null,
  threshold = null,
  width = 460,
  height = 220,
  posLabel = 'Con parentesco',
  negLabel = 'Sin parentesco',
  compact = false,
}: HistogramChartProps) {
  const { bin_edges, pos_counts, neg_counts } = histogram;
  const nBins = pos_counts.length;
  const xmin = bin_edges[0];
  const xmax = bin_edges[bin_edges.length - 1];

  const m = { top: 12, right: 12, bottom: 34, left: 38 };
  const plotW = width - m.left - m.right;
  const plotH = height - m.top - m.bottom - (compact ? 0 : 18); // espacio leyenda

  const totPos = pos_counts.reduce((a, b) => a + b, 0) || 1;
  const totNeg = neg_counts.reduce((a, b) => a + b, 0) || 1;

  // Escala vertical: fracción de cada clase. maxFrac fija el techo común.
  let maxFrac = 0;
  for (let i = 0; i < nBins; i++) {
    maxFrac = Math.max(maxFrac, pos_counts[i] / totPos, neg_counts[i] / totNeg);
  }
  if (maxFrac === 0) maxFrac = 1;

  const xScale = (v: number) => m.left + ((v - xmin) / (xmax - xmin)) * plotW;
  const yBase = m.top + plotH;
  const barH = (frac: number) => (frac / maxFrac) * plotH;

  // Ticks X: 5 valores equiespaciados sobre el rango.
  const xTicks = Array.from({ length: 5 }, (_, k) => xmin + ((xmax - xmin) * k) / 4);

  const bars = (counts: number[], total: number, color: string) =>
    counts.map((c, i) => {
      if (c === 0) return null;
      const x0 = xScale(bin_edges[i]);
      const x1 = xScale(bin_edges[i + 1]);
      const h = barH(c / total);
      return (
        <rect
          key={i}
          x={x0}
          y={yBase - h}
          width={Math.max(0.5, x1 - x0 - 0.4)}
          height={h}
          fill={color}
          fillOpacity={0.45}
        />
      );
    });

  return (
    <svg width={width} height={height} style={{ display: 'block', fontFamily: 'monospace' }}>
      {/* eje Y (línea base) */}
      <line x1={m.left} y1={yBase} x2={m.left + plotW} y2={yBase} stroke="#999" strokeWidth={1} />

      {/* barras: non-kin primero (atrás), kin encima */}
      {bars(neg_counts, totNeg, NEG_COLOR)}
      {bars(pos_counts, totPos, POS_COLOR)}

      {/* línea de umbral (punteada) */}
      {threshold != null && threshold >= xmin && threshold <= xmax && (
        <g>
          <line
            x1={xScale(threshold)} y1={m.top} x2={xScale(threshold)} y2={yBase}
            stroke="#888" strokeWidth={1.2} strokeDasharray="4 3"
          />
          <text x={xScale(threshold)} y={m.top + 8} fontSize={9} fill="#666" textAnchor="middle">
            umbral
          </text>
        </g>
      )}

      {/* marcador del valor propio (sólido, destacado) */}
      {marker != null && marker >= xmin && marker <= xmax && (
        <g>
          <line
            x1={xScale(marker)} y1={m.top} x2={xScale(marker)} y2={yBase}
            stroke="#111" strokeWidth={2}
          />
          <polygon
            points={`${xScale(marker) - 5},${m.top} ${xScale(marker) + 5},${m.top} ${xScale(marker)},${m.top + 7}`}
            fill="#111"
          />
        </g>
      )}

      {/* ticks X */}
      {xTicks.map((t, k) => (
        <g key={k}>
          <line x1={xScale(t)} y1={yBase} x2={xScale(t)} y2={yBase + 4} stroke="#999" />
          <text x={xScale(t)} y={yBase + 15} fontSize={9} fill="#555" textAnchor="middle">
            {t.toFixed(2)}
          </text>
        </g>
      ))}

      {/* leyenda */}
      {!compact && (
        <g transform={`translate(${m.left}, ${height - 8})`}>
          <rect x={0} y={-8} width={10} height={10} fill={POS_COLOR} fillOpacity={0.45} />
          <text x={14} y={1} fontSize={10} fill="#333">{posLabel} (n={totPos})</text>
          <rect x={150} y={-8} width={10} height={10} fill={NEG_COLOR} fillOpacity={0.45} />
          <text x={164} y={1} fontSize={10} fill="#333">{negLabel} (n={totNeg})</text>
        </g>
      )}
    </svg>
  );
}
