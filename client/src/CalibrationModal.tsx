// =========================================
// ID: PHYLOFACE_CALIBRATION_MODAL
// VERSION: v1.0
// =========================================
// Popup que ubica un cosine concreto sobre la distribución calibrada de la
// Tarea #6 (Fase B). Se abre al clickear el valor de un cosine en el
// Comparador (`CosineCard`) o en el árbol (`TripletModal`).
//
// Muestra, para la relación elegida (selector; default best-effort inferido
// del contexto que lo abre):
//   - el histograma kin vs non-kin con el cosine propio marcado y el umbral;
//   - MÉTRICA NUEVA (basada en las distribuciones): probabilidad calibrada de
//     parentesco P(kin|cos) + percentiles dentro de cada distribución + LR;
//   - MÉTRICAS PREVIAS (Fase A): el cosine crudo, el veredicto duro vs umbral
//     de Youden, y la calidad agregada del clasificador (accuracy ± std, AUC).
//   Todas conviven en pantalla a propósito: al re-calibrar (ajustes sucesivos)
//   se ve cómo se mueven unas respecto de otras.
//
// Overlay manual (no <dialog>) siguiendo el patrón de `TripletModal` (gotchas
// de focus/escape en Chromium headless). zIndex alto para anidar sobre el
// TripletModal cuando se abre desde el árbol.

import { useEffect, useState } from 'react';
import {
  loadCalibration,
  scoreValue,
  higherIsKin,
  RELATIONS,
  RELATION_LABEL,
  type CalibrationArtifact,
  type Relation,
  type ValueScore,
} from './lib/calibration';
import HistogramChart from './CalibrationChart';

interface CalibrationModalProps {
  /** El cosine a ubicar. */
  value: number;
  /** Etiqueta del par que se está comparando (ej. "Hijo/a ↔ Padre"). */
  pairLabel?: string;
  /** Relación inicial (best-effort del contexto). Default 'ALL'. */
  defaultRelation?: Relation;
  /** Dataset de calibración a cargar. */
  dataset?: string;
  onClose: () => void;
}

const METRIC = 'cosine' as const; // el comparador/árbol sólo computan cosine

function pct(x: number): string {
  return `${(x * 100).toFixed(1)}%`;
}

// El LR puede dispararse en las colas (posterior→1). Lo capamos en el display
// para no mostrar un número astronómico sin información útil.
function fmtLR(lr: number): string {
  if (!Number.isFinite(lr) || lr > 999) return '>999×';
  if (lr < 0.001) return '<0.001×';
  return `${lr.toFixed(2)}×`;
}

export default function CalibrationModal({
  value,
  pairLabel,
  defaultRelation = 'ALL',
  dataset = 'KinFaceW-I',
  onClose,
}: CalibrationModalProps) {
  const [cal, setCal] = useState<CalibrationArtifact | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [relation, setRelation] = useState<Relation>(defaultRelation);

  useEffect(() => {
    let cancelled = false;
    loadCalibration(dataset)
      .then((c) => { if (!cancelled) setCal(c); })
      .catch((e) => { if (!cancelled) setError((e as Error).message); });
    return () => { cancelled = true; };
  }, [dataset]);

  // ESC cierra.
  useEffect(() => {
    const handler = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose(); };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [onClose]);

  const score: ValueScore | null = cal ? scoreValue(cal, METRIC, relation, value) : null;

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-label="Calibración del parecido"
      data-testid="calibration-modal"
      onClick={(e) => { if (e.target === e.currentTarget) onClose(); }}
      style={{
        position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.55)',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        zIndex: 1100, fontFamily: 'monospace',
      }}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        style={{
          background: '#fff', borderRadius: 8, padding: 24,
          minWidth: 520, maxWidth: 640, maxHeight: '90vh', overflowY: 'auto',
          boxShadow: '0 10px 40px rgba(0,0,0,0.3)',
        }}
      >
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
          <h3 style={{ margin: 0 }}>¿Dónde cae este parecido?</h3>
          <button onClick={onClose} aria-label="Cerrar" style={{ fontSize: 14 }}>✕ cerrar</button>
        </div>

        <div style={{ fontSize: 13, color: '#555', marginBottom: 12 }}>
          {pairLabel && <span style={{ marginRight: 8 }}>{pairLabel}</span>}
          cosine = <strong style={{ color: '#1a73e8' }}>{value.toFixed(4)}</strong>
        </div>

        {error && (
          <div style={{ background: '#fee', color: '#900', padding: 10, borderRadius: 4, fontSize: 12 }}>
            No se pudo cargar la calibración: {error}
          </div>
        )}

        {!cal && !error && <div style={{ color: '#888', fontSize: 13 }}>cargando calibración…</div>}

        {cal && score && (
          <>
            {/* Selector de relación */}
            <div style={{ marginBottom: 12, fontSize: 13 }}>
              <label>
                relación:&nbsp;
                <select
                  value={relation}
                  onChange={(e) => setRelation(e.target.value as Relation)}
                  data-testid="calibration-relation"
                  style={{ fontFamily: 'monospace', fontSize: 13, padding: '2px 6px' }}
                >
                  {RELATIONS.map((r) => (
                    <option key={r} value={r}>{r} — {RELATION_LABEL[r]}</option>
                  ))}
                </select>
              </label>
              {defaultRelation === 'ALL' && (
                <div style={{ fontSize: 11, color: '#999', marginTop: 4 }}>
                  No se conoce el sexo del hijo/a → no se puede fijar Hijo vs Hija.
                  Default agregada; elegí la relación específica para refinar.
                </div>
              )}
            </div>

            {/* Histograma con el cosine propio marcado */}
            <div style={{ border: '1px solid #eee', borderRadius: 4, padding: 8, marginBottom: 14 }}>
              <HistogramChart
                histogram={score.histogram}
                marker={value}
                threshold={score.threshold}
                width={560}
                height={230}
              />
            </div>

            {/* MÉTRICA NUEVA — probabilidad calibrada */}
            <div style={{
              background: '#f0f7ff', border: '1px solid #b8d4ff', borderRadius: 6,
              padding: 14, marginBottom: 12,
            }}>
              <div style={{ fontSize: 12, color: '#555', marginBottom: 2 }}>
                Métrica nueva — probabilidad de parentesco
              </div>
              <div style={{ fontSize: 34, fontWeight: 700, color: '#1a73e8' }}
                   data-testid="calibration-posterior">
                {pct(score.posterior)}
              </div>
              <div style={{ fontSize: 11, color: '#777', marginBottom: 8 }}>
                P(parentesco | cosine) por densidad-ratio de las distribuciones,
                priors iguales (50/50). No es una probabilidad poblacional real.
              </div>
              <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
                <tbody>
                  <MetricRow
                    label="Percentil entre parientes reales"
                    value={pct(score.percentileKin)}
                    hint="fracción de pares CON parentesco cuyo cosine es ≤ al tuyo"
                  />
                  <MetricRow
                    label="Percentil entre no-parientes"
                    value={pct(score.percentileNon)}
                    hint="fracción de pares SIN parentesco cuyo cosine es ≤ al tuyo"
                  />
                  <MetricRow
                    label="Likelihood ratio (kin / non-kin)"
                    value={fmtLR(score.likelihoodRatio)}
                    hint=">1 favorece parentesco; <1 lo desfavorece"
                  />
                </tbody>
              </table>
            </div>

            {/* MÉTRICAS PREVIAS (Fase A) */}
            <div style={{
              background: '#fafafa', border: '1px solid #eee', borderRadius: 6, padding: 14,
            }}>
              <div style={{ fontSize: 12, color: '#555', marginBottom: 8 }}>
                Métricas previas (Fase A — baseline de cosine crudo, KinFaceW-I)
              </div>
              <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
                <tbody>
                  <MetricRow
                    label="cosine crudo"
                    value={value.toFixed(4)}
                  />
                  <MetricRow
                    label={`veredicto vs umbral (${score.threshold.toFixed(3)}, Youden)`}
                    value={score.isKin ? '✓ compatible con parentesco' : '✗ no compatible'}
                    valueColor={score.isKin ? '#080' : '#900'}
                    hint={higherIsKin(METRIC) ? 'kin si cosine ≥ umbral' : 'kin si distancia ≤ umbral'}
                  />
                  <MetricRow
                    label="accuracy 5-CV (esta relación)"
                    value={`${pct(score.accuracyMean)} ± ${pct(score.accuracyStd)}`}
                  />
                  <MetricRow
                    label="AUC (esta relación)"
                    value={score.auc.toFixed(3)}
                  />
                </tbody>
              </table>
            </div>

            {/* Metadata de la corrida cargada */}
            <div style={{ fontSize: 10, color: '#aaa', marginTop: 12, lineHeight: 1.5 }}>
              {cal.dataset} · modelo {cal.modelVersion} · protocolo {cal.protocol} ·
              calibrado {new Date(cal.computedAt).toLocaleString()}
              <br />
              Baseline honesto del cosine crudo (sin cabeza de parentesco aprendida);
              señal débil, en especial cross-género (Padre–Hija). Interpretar como
              orientativo, no como veredicto.
            </div>
          </>
        )}
      </div>
    </div>
  );
}

interface MetricRowProps {
  label: string;
  value: string;
  hint?: string;
  valueColor?: string;
}

function MetricRow({ label, value, hint, valueColor }: MetricRowProps) {
  return (
    <tr style={{ borderBottom: '1px dashed #e4e4e4' }}>
      <td style={{ padding: '4px 0', verticalAlign: 'top' }}>
        {label}
        {hint && <div style={{ fontSize: 10, color: '#aaa' }}>{hint}</div>}
      </td>
      <td style={{ padding: '4px 0', textAlign: 'right', fontWeight: 700, color: valueColor ?? '#333', whiteSpace: 'nowrap' }}>
        {value}
      </td>
    </tr>
  );
}
