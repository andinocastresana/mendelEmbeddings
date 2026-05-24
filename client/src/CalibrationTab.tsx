// =========================================
// ID: PHYLOFACE_CALIBRATION_TAB
// VERSION: v1.0
// =========================================
// Solapa "Calibración" (Tarea #6, Fase B). Vista de toda la calibración:
//   - metadata de la corrida cargada (dataset, modelo, protocolo, fecha) — así
//     se distingue qué ajuste se está mirando;
//   - tabla de métricas agregadas por relación (n, accuracy 5-CV, umbral, AUC)
//     para la métrica elegida (cosine/euclidean);
//   - grilla de histogramas kin vs non-kin, uno por relación.
//
// Es la contraparte "panorámica" del popup `CalibrationModal` (que ubica UN
// cosine concreto). Acá no hay un valor propio que marcar; sólo se muestra el
// umbral de Youden de cada relación. Pensada para ver cómo se mueven las
// distribuciones y las métricas tras sucesivos ajustes del pipeline.

import { useEffect, useState } from 'react';
import {
  loadCalibration,
  RELATIONS,
  RELATION_LABEL,
  type CalibrationArtifact,
  type Metric,
} from './lib/calibration';
import HistogramChart from './CalibrationChart';

export default function CalibrationTab() {
  const [cal, setCal] = useState<CalibrationArtifact | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [metric, setMetric] = useState<Metric>('cosine');

  useEffect(() => {
    let cancelled = false;
    loadCalibration('KinFaceW-I')
      .then((c) => { if (!cancelled) setCal(c); })
      .catch((e) => { if (!cancelled) setError((e as Error).message); });
    return () => { cancelled = true; };
  }, []);

  return (
    <div style={{ fontFamily: 'monospace', padding: '20px', maxWidth: 1200, margin: '0 auto' }}>
      <h1 style={{ borderBottom: '2px solid #333', paddingBottom: 8 }}>
        Calibración de parentesco
      </h1>
      <p style={{ color: '#666', fontSize: 13, lineHeight: 1.5 }}>
        Distribuciones de similitud sobre pares etiquetados de KinFaceW-I, por
        relación. Convierten el cosine crudo del comparador/árbol en un veredicto
        con respaldo cuantitativo: umbral data-driven (Youden) + AUC, y la nueva
        probabilidad calibrada que se ve al clickear cualquier cosine. Baseline
        honesto del cosine crudo (sin cabeza de parentesco aprendida).
      </p>

      {error && (
        <div style={{ background: '#fee', color: '#900', padding: 12, borderRadius: 4 }}>
          No se pudo cargar la calibración: {error}. ¿Existe
          <code> client/public/calibration/KinFaceW-I_calibration.json</code>?
        </div>
      )}

      {!cal && !error && <div style={{ color: '#888' }}>cargando calibración…</div>}

      {cal && (
        <>
          {/* Metadata + toggle de métrica */}
          <div style={{
            display: 'flex', justifyContent: 'space-between', alignItems: 'center',
            flexWrap: 'wrap', gap: 12, margin: '12px 0 8px',
          }}>
            <div style={{ fontSize: 11, color: '#888' }}>
              {cal.dataset} · modelo {cal.modelVersion} · {cal.protocol} ·
              calibrado {new Date(cal.computedAt).toLocaleString()}
            </div>
            <div style={{ fontSize: 13 }}>
              métrica:&nbsp;
              <select
                value={metric}
                onChange={(e) => setMetric(e.target.value as Metric)}
                data-testid="calibration-tab-metric"
                style={{ fontFamily: 'monospace', fontSize: 13, padding: '2px 6px' }}
              >
                <option value="cosine">cosine</option>
                <option value="euclidean">euclidean</option>
              </select>
            </div>
          </div>

          {/* Tabla de métricas agregadas por relación */}
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12, marginBottom: 20 }}>
            <thead>
              <tr style={{ background: '#f4f4f4' }}>
                <th style={th}>relación</th>
                <th style={th}>n_pos</th>
                <th style={th}>n_neg</th>
                <th style={th}>accuracy 5-CV</th>
                <th style={th}>umbral (Youden)</th>
                <th style={th}>AUC</th>
              </tr>
            </thead>
            <tbody>
              {RELATIONS.map((r) => {
                const rc = cal.metrics[metric][r];
                return (
                  <tr key={r}>
                    <td style={td}>{r} — {RELATION_LABEL[r]}</td>
                    <td style={td}>{rc.n_pos}</td>
                    <td style={td}>{rc.n_neg}</td>
                    <td style={td}>
                      {(rc.accuracy_mean * 100).toFixed(1)}% ± {(rc.accuracy_std * 100).toFixed(1)}%
                    </td>
                    <td style={td}>{rc.threshold_mean.toFixed(3)}</td>
                    <td style={td}>{rc.auc.toFixed(3)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>

          {/* Grilla de histogramas */}
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 16 }}>
            {RELATIONS.map((r) => {
              const rc = cal.metrics[metric][r];
              return (
                <div key={r} style={{
                  border: '1px solid #e4e4e4', borderRadius: 6, padding: 12,
                  background: '#fff', minWidth: 360,
                }}>
                  <div style={{ fontWeight: 700, fontSize: 13, marginBottom: 2 }}>
                    {r} — {RELATION_LABEL[r]}
                  </div>
                  <div style={{ fontSize: 11, color: '#888', marginBottom: 6 }}>
                    AUC {rc.auc.toFixed(3)} · acc {(rc.accuracy_mean * 100).toFixed(1)}% ·
                    umbral {rc.threshold_mean.toFixed(3)}
                  </div>
                  <HistogramChart
                    histogram={rc.histogram}
                    threshold={rc.threshold_mean}
                    width={420}
                    height={200}
                  />
                </div>
              );
            })}
          </div>
        </>
      )}
    </div>
  );
}

const td: React.CSSProperties = { border: '1px solid #ccc', padding: '5px 8px' };
const th: React.CSSProperties = { border: '1px solid #ccc', padding: '5px 8px', textAlign: 'left' };
