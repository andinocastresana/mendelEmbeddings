// =========================================
// ID: PHYLOFACE_SPIKE_004
// VERSION: v2.1
// =========================================
// Componente del SPIKE Track 2a — paridad del pipeline e2e JS vs Python,
// versión MULTI-IMAGEN.
//
// Cambio v2.0 -> v2.1 (refactor Tarea #25c):
// - El pipeline e2e (detect → align → embed) se extrajo a `lib/pipeline.ts`
//   (`computeEmbedding`) para que lo reuse la UI MVP del comparador. Acá
//   solo queda la capa de comparación contra el embedding de referencia del
//   fixture multi-caso, las métricas de paridad (kps distance, max_abs_diff,
//   cosine) y la UI de tabla/overlay/download.
//
// Cambio v1 -> v2 (2026-05-21, sesión multi-imagen):
// - Carga un fixture multi-caso (`cases.json`) en lugar de los 3 archivos
//   sueltos de v1.
// - Itera el pipeline e2e sobre TODOS los casos, acumula métricas, muestra
//   tabla por caso + agregadas (mean/median/min/max cosine, # PASS/FAIL).
// - Selector para visualizar el overlay de un caso concreto (no todos a la vez).
// - Botón "Descargar JSON" para persistir el snapshot de la corrida si querés
//   acumularlo manualmente (el script Python ya escribe el "set state" a
//   `_meta/spike_004_runs.md`, pero NO las métricas JS — eso se exporta acá).
//
// =========================================
// DECISIÓN ABIERTA — detector intercambiable
// =========================================
// Este spike usa MediaPipe Face Mesh con `refine_landmarks=true`. BlazeFace
// queda como alternativa más liviana para devices low-end, pero le faltan
// las comisuras de boca — requeriría workaround.
// Memoria: project-browser-detector-adapter.
//
// =========================================
// DECISIÓN DIFERIDA — pipeline upload+dedup+métricas en Track 2b
// =========================================
// La carpeta `data/input/img/spike_e2e_set/` se puebla manualmente hoy. En
// Track 2b vamos a tener drag-and-drop browser que persista uploads + dedup
// por SHA-256 + DB para métricas acumuladas, lo cual va a alimentar el
// dataset de calibración del pipeline.
// Memoria: project-track2b-dataset-pipeline.

import { useEffect, useMemo, useRef, useState } from 'react';
import {
  computeEmbedding,
  cosineSimilarity,
  initFaceLandmarker,
  initOnnxSession,
  loadImage,
} from './lib/pipeline';

// -----------------------------------------
// Tipos del fixture multi-caso
// -----------------------------------------
interface CaseDoc {
  hash: string;
  source_filename: string;
  public_filename: string;
  image_size_wh: number[];
  bbox_global: number[];
  det_score: number;
  kps_global: {
    values: number[][];
    order: string;
  };
  reference_embedding: {
    shape: number[];
    norm_l2: number;
    values: number[];
  };
}

interface CasesFixture {
  version: number;
  set_hash: string;
  generated_at_utc: string;
  cases: CaseDoc[];
}

interface FixtureMetadata {
  generated_at_utc: string;
  version: number;
  image_dir: string;
  n_cases_total: number;
  n_cases_new: number;
  n_cases_reused: number;
  n_cases_no_face: number;
  set_hash: string;
  python_pipeline: { detector: string };
  success_criteria_for_js_client: {
    cosine_similarity_min: number;
    global_pass_rule: string;
  };
}

interface CaseResult {
  hash: string;
  sourceFilename: string;
  publicFilename: string;
  imageSizeWh: number[];
  cosine: number;
  maxAbsDiff: number;
  meanKpsDistance: number;
  maxKpsDistance: number;
  perKpDistance: number[];
  detectMs: number;
  alignMs: number;
  preprocessMs: number;
  inferMs: number;
  passed: boolean;
  // Para overlay visual on-demand.
  kpsJs: number[][];
  kpsRef: number[][];
}

// -----------------------------------------
// Métricas de comparación específicas del spike (no pertenecen al pipeline
// productivo; solo se usan acá contra refs Python).
// -----------------------------------------
function maxAbsDiff(a: Float32Array, b: Float32Array): number {
  let m = 0;
  for (let i = 0; i < a.length; i++) {
    const d = Math.abs(a[i] - b[i]);
    if (d > m) m = d;
  }
  return m;
}

function kpsDistances(jsKps: number[][], refKps: number[][]): { mean: number; max: number; perKp: number[] } {
  const n = Math.min(jsKps.length, refKps.length);
  let sum = 0, max = 0;
  const per: number[] = [];
  for (let i = 0; i < n; i++) {
    const dx = jsKps[i][0] - refKps[i][0];
    const dy = jsKps[i][1] - refKps[i][1];
    const d = Math.sqrt(dx * dx + dy * dy);
    per.push(d);
    sum += d;
    if (d > max) max = d;
  }
  return { mean: sum / n, max, perKp: per };
}

function median(values: number[]): number {
  if (values.length === 0) return NaN;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid];
}

// -----------------------------------------
// Procesar un caso del set: corre el pipeline e2e (lib/pipeline) y lo compara
// contra el embedding+kps de referencia del fixture Python.
// -----------------------------------------
async function runOnePipeline(
  caseDoc: CaseDoc,
  faceLandmarker: Parameters<typeof computeEmbedding>[2],
  session: Parameters<typeof computeEmbedding>[3],
  imagesBaseUrl: string,
): Promise<CaseResult> {
  const refEmb = new Float32Array(caseDoc.reference_embedding.values);
  const refKps = caseDoc.kps_global.values;

  const { img, imageData } = await loadImage(`${imagesBaseUrl}/${caseDoc.public_filename}`);
  const { embedding: embJs, kps: kpsJs, timings } = await computeEmbedding(
    img, imageData, faceLandmarker, session,
  );

  const kpsDist = kpsDistances(kpsJs, refKps);
  const cosine = cosineSimilarity(embJs, refEmb);
  const maxDiff = maxAbsDiff(embJs, refEmb);

  return {
    hash: caseDoc.hash,
    sourceFilename: caseDoc.source_filename,
    publicFilename: caseDoc.public_filename,
    imageSizeWh: caseDoc.image_size_wh,
    cosine,
    maxAbsDiff: maxDiff,
    meanKpsDistance: kpsDist.mean,
    maxKpsDistance: kpsDist.max,
    perKpDistance: kpsDist.perKp,
    detectMs: timings.detectMs,
    alignMs: timings.alignMs,
    preprocessMs: timings.preprocessMs,
    inferMs: timings.inferMs,
    passed: false, // se setea afuera con el threshold del metadata
    kpsJs,
    kpsRef: refKps,
  };
}

// -----------------------------------------
// Componente
// -----------------------------------------
function SpikeDetection() {
  const [steps, setSteps] = useState<string[]>([]);
  const [metadata, setMetadata] = useState<FixtureMetadata | null>(null);
  const [casesFixture, setCasesFixture] = useState<CasesFixture | null>(null);
  const [results, setResults] = useState<CaseResult[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [done, setDone] = useState(false);

  // Selector de caso para el overlay visual.
  const [selectedIdx, setSelectedIdx] = useState<number>(0);
  const canvasSourceRef = useRef<HTMLCanvasElement | null>(null);

  const log = (msg: string) => {
    console.log(msg);
    setSteps((prev) => [...prev, msg]);
  };

  useEffect(() => {
    let cancelled = false;

    (async () => {
      try {
        log('[1] Cargando fixture multi-caso...');
        const [meta, casesFixt] = await Promise.all([
          fetch('/spike_fixtures_detection/metadata.json').then(r => r.json() as Promise<FixtureMetadata>),
          fetch('/spike_fixtures_detection/cases.json').then(r => r.json() as Promise<CasesFixture>),
        ]);
        if (cancelled) return;
        setMetadata(meta);
        setCasesFixture(casesFixt);
        log(`    set_hash=${meta.set_hash.slice(0, 16)}...`);
        log(`    n_cases_total=${meta.n_cases_total} (nuevos=${meta.n_cases_new}, reusados=${meta.n_cases_reused}, no_face=${meta.n_cases_no_face})`);
        log(`    criterio: cosine > ${meta.success_criteria_for_js_client.cosine_similarity_min} por caso`);
        log(`    regla global: ${meta.success_criteria_for_js_client.global_pass_rule}`);

        log('[2] Inicializando MediaPipe FaceLandmarker (1x para todos los casos)...');
        const faceLandmarker = await initFaceLandmarker();
        if (cancelled) return;

        log('[3] Inicializando ONNX session (1x para todos los casos)...');
        const session = await initOnnxSession();
        if (cancelled) return;

        log(`[4] Iterando pipeline e2e sobre ${casesFixt.cases.length} caso(s)...`);
        const threshold = meta.success_criteria_for_js_client.cosine_similarity_min;
        const all: CaseResult[] = [];
        for (let i = 0; i < casesFixt.cases.length; i++) {
          const caseDoc = casesFixt.cases[i];
          if (cancelled) return;
          log(`  [${i + 1}/${casesFixt.cases.length}] ${caseDoc.source_filename}  hash=${caseDoc.hash.slice(0, 12)}`);
          try {
            const result = await runOnePipeline(
              caseDoc, faceLandmarker, session, '/spike_fixtures_detection/images',
            );
            result.passed = result.cosine > threshold;
            all.push(result);
            log(
              `      cosine=${result.cosine.toFixed(6)} ${result.passed ? '✓' : '✗'} | ` +
              `mean_kps=${result.meanKpsDistance.toFixed(2)}px | ` +
              `infer=${result.inferMs.toFixed(0)}ms`
            );
          } catch (e: any) {
            log(`      ERROR: ${e?.message || String(e)}`);
          }
        }

        if (cancelled) return;
        setResults(all);

        const nPass = all.filter(r => r.passed).length;
        const nFail = all.length - nPass;
        const globalPass = nFail === 0 && all.length > 0;
        log('');
        log(`[5] Resumen: ${nPass}/${all.length} casos PASS (${nFail} FAIL).`);
        if (globalPass) {
          log('[OK] Spike GLOBAL PASS. Todos los casos del set pasaron threshold.');
        } else if (all.length === 0) {
          log('[!!] Sin casos válidos (¿set vacío?, ¿fixture corrupto?).');
        } else {
          log(`[!!] Spike GLOBAL FAIL: ${nFail} caso(s) bajo threshold ${threshold}.`);
        }
        setDone(true);
      } catch (e: any) {
        const errMsg = `[ERROR] ${e?.message || String(e)}`;
        console.error(errMsg, e);
        if (!cancelled) {
          setError(errMsg);
          log(errMsg);
        }
      }
    })();

    return () => { cancelled = true; };
  }, []);

  // Métricas agregadas memoizadas.
  const summary = useMemo(() => {
    if (results.length === 0) return null;
    const cosines = results.map(r => r.cosine);
    const meanKps = results.map(r => r.meanKpsDistance);
    return {
      n: results.length,
      nPass: results.filter(r => r.passed).length,
      nFail: results.filter(r => !r.passed).length,
      cosineMean: cosines.reduce((a, b) => a + b, 0) / cosines.length,
      cosineMedian: median(cosines),
      cosineMin: Math.min(...cosines),
      cosineMax: Math.max(...cosines),
      meanKpsMean: meanKps.reduce((a, b) => a + b, 0) / meanKps.length,
      meanKpsMax: Math.max(...meanKps),
    };
  }, [results]);

  // Overlay del caso seleccionado.
  useEffect(() => {
    if (results.length === 0 || !canvasSourceRef.current) return;
    const idx = Math.min(selectedIdx, results.length - 1);
    const r = results[idx];
    const canvas = canvasSourceRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => {
      const renderW = 400;
      const scale = renderW / img.width;
      canvas.width = renderW;
      canvas.height = Math.round(img.height * scale);
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      // ref (rojo) + js (verde).
      ctx.font = '11px monospace';
      ctx.fillStyle = 'rgba(255, 0, 0, 0.85)';
      r.kpsRef.forEach(([x, y], i) => {
        ctx.beginPath(); ctx.arc(x * scale, y * scale, 4, 0, 2 * Math.PI); ctx.fill();
        ctx.fillText(`R${i}`, x * scale + 6, y * scale - 4);
      });
      ctx.fillStyle = 'rgba(0, 200, 0, 0.85)';
      r.kpsJs.forEach(([x, y], i) => {
        ctx.beginPath(); ctx.arc(x * scale, y * scale, 3, 0, 2 * Math.PI); ctx.fill();
        ctx.fillText(`J${i}`, x * scale + 6, y * scale + 12);
      });
    };
    img.src = `/spike_fixtures_detection/images/${r.publicFilename}`;
  }, [results, selectedIdx]);

  const downloadResultsJson = () => {
    if (results.length === 0) return;
    const blob = new Blob([JSON.stringify({
      exported_at_utc: new Date().toISOString(),
      set_hash: metadata?.set_hash,
      threshold: metadata?.success_criteria_for_js_client.cosine_similarity_min,
      summary,
      cases: results.map(r => ({
        hash: r.hash,
        source_filename: r.sourceFilename,
        image_size_wh: r.imageSizeWh,
        cosine: r.cosine,
        max_abs_diff: r.maxAbsDiff,
        mean_kps_distance: r.meanKpsDistance,
        max_kps_distance: r.maxKpsDistance,
        per_kp_distance: r.perKpDistance,
        timings_ms: {
          detect: r.detectMs, align: r.alignMs,
          preprocess: r.preprocessMs, infer: r.inferMs,
        },
        passed: r.passed,
      })),
    }, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    const ts = new Date().toISOString().replace(/[:.]/g, '-');
    a.download = `spike_004_run_${ts}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <div style={{ fontFamily: 'monospace', padding: '20px', maxWidth: 1100, margin: '0 auto' }}>
      <h1 style={{ borderBottom: '2px solid #333', paddingBottom: 8 }}>
        Spike detección e2e — multi-imagen
      </h1>
      <p style={{ color: '#666', fontSize: 13 }}>
        Itera el pipeline browser (Face Mesh → 5 kps → align → ONNX → cosine)
        sobre todas las imágenes del set <code>data/input/img/spike_e2e_set/</code>
        y compara contra el embedding de referencia que el motor Python produjo
        para cada una. Agregá imágenes a esa carpeta y re-corré el script
        <code style={{ marginLeft: 4 }}>scripts/verify_detection_web_parity.py</code>
        para extender el set.
      </p>

      <h2 style={{ marginTop: 20 }}>Log</h2>
      <pre style={{
        background: '#f4f4f4', padding: 12, borderRadius: 4, fontSize: 12,
        lineHeight: 1.5, whiteSpace: 'pre-wrap', maxHeight: 400, overflow: 'auto',
      }}>
        {steps.join('\n')}
        {!done && !error && <span style={{ color: '#888' }}>{'\n…corriendo…'}</span>}
      </pre>

      {error && (
        <div style={{ background: '#fee', color: '#900', padding: 12, borderRadius: 4, marginTop: 12 }}>
          <strong>Error:</strong> {error}
        </div>
      )}

      {summary && metadata && (
        <>
          <h2 style={{ marginTop: 20 }}>Resumen agregado</h2>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <tbody>
              <tr><td style={td}>casos</td><td style={td}>{summary.n} ({summary.nPass} PASS / {summary.nFail} FAIL)</td></tr>
              <tr><td style={td}>cosine</td><td style={td}>mean {summary.cosineMean.toFixed(4)} · median {summary.cosineMedian.toFixed(4)} · min {summary.cosineMin.toFixed(4)} · max {summary.cosineMax.toFixed(4)}</td></tr>
              <tr><td style={td}>mean kps distance</td><td style={td}>{summary.meanKpsMean.toFixed(2)} px (peor caso: {summary.meanKpsMax.toFixed(2)} px)</td></tr>
              <tr style={{ background: summary.nFail === 0 ? '#efffef' : '#ffeeee', fontWeight: 700 }}>
                <td style={td}>GLOBAL</td>
                <td style={td}>{summary.nFail === 0 ? '✓ PASS (todos)' : `✗ FAIL (${summary.nFail} caso(s) bajo cosine ${metadata.success_criteria_for_js_client.cosine_similarity_min})`}</td>
              </tr>
            </tbody>
          </table>

          <h2 style={{ marginTop: 20 }}>Por caso</h2>
          <button onClick={downloadResultsJson} style={{
            padding: '8px 14px', fontFamily: 'monospace', fontSize: 12,
            cursor: 'pointer', marginBottom: 10,
            background: '#e8f0fe', border: '1px solid #4a8',
          }}>
            Descargar JSON de esta corrida ↓
          </button>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ background: '#f4f4f4' }}>
                <th style={th}>#</th>
                <th style={th}>filename</th>
                <th style={th}>hash</th>
                <th style={th}>size</th>
                <th style={th}>cosine</th>
                <th style={th}>kps mean</th>
                <th style={th}>kps max</th>
                <th style={th}>detect</th>
                <th style={th}>align</th>
                <th style={th}>infer</th>
                <th style={th}>status</th>
              </tr>
            </thead>
            <tbody>
              {results.map((r, i) => (
                <tr key={r.hash}
                    onClick={() => setSelectedIdx(i)}
                    style={{
                      background: i === selectedIdx ? '#eef' : (r.passed ? 'transparent' : '#ffe9e9'),
                      cursor: 'pointer',
                    }}>
                  <td style={td}>{i + 1}</td>
                  <td style={td}>{r.sourceFilename}</td>
                  <td style={td} title={r.hash}>{r.hash.slice(0, 10)}</td>
                  <td style={td}>{r.imageSizeWh[0]}×{r.imageSizeWh[1]}</td>
                  <td style={{ ...td, fontWeight: 600 }}>{r.cosine.toFixed(4)}</td>
                  <td style={td}>{r.meanKpsDistance.toFixed(1)}</td>
                  <td style={td}>{r.maxKpsDistance.toFixed(1)}</td>
                  <td style={td}>{r.detectMs.toFixed(0)}</td>
                  <td style={td}>{r.alignMs.toFixed(1)}</td>
                  <td style={td}>{r.inferMs.toFixed(0)}</td>
                  <td style={td}>{r.passed ? '✓' : '✗'}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p style={{ fontSize: 11, color: '#888', marginTop: 6 }}>
            Click en una fila para ver overlay visual de ese caso ↓
          </p>

          <h2 style={{ marginTop: 20 }}>
            Overlay del caso #{selectedIdx + 1} — {results[selectedIdx]?.sourceFilename}
          </h2>
          <p style={{ fontSize: 12, color: '#666' }}>
            <span style={{ color: '#c00' }}>● rojo = InsightFace ref</span>,
            <span style={{ color: '#0a0', marginLeft: 8 }}>● verde = JS (Face Mesh)</span>
          </p>
          <canvas ref={canvasSourceRef} style={{ border: '1px solid #ccc', maxWidth: '100%' }} />
        </>
      )}

      {metadata && (
        <details style={{ marginTop: 20, fontSize: 12 }}>
          <summary style={{ cursor: 'pointer', fontWeight: 600 }}>Fixture metadata</summary>
          <pre style={{ background: '#f4f4f4', padding: 10, borderRadius: 4, overflow: 'auto' }}>
            {JSON.stringify(metadata, null, 2)}
          </pre>
        </details>
      )}
      {casesFixture && (
        <details style={{ marginTop: 8, fontSize: 12 }}>
          <summary style={{ cursor: 'pointer', fontWeight: 600 }}>
            cases.json ({casesFixture.cases.length} casos, set_hash {casesFixture.set_hash.slice(0, 16)}...)
          </summary>
          <pre style={{ background: '#f4f4f4', padding: 10, borderRadius: 4, overflow: 'auto', maxHeight: 300 }}>
            {casesFixture.cases.map(c => `${c.hash.slice(0, 12)}  ${c.source_filename}  det=${c.det_score.toFixed(3)}`).join('\n')}
          </pre>
        </details>
      )}
    </div>
  );
}

const td: React.CSSProperties = { border: '1px solid #ccc', padding: '6px 8px' };
const th: React.CSSProperties = { border: '1px solid #ccc', padding: '6px 8px', textAlign: 'left' };

export default SpikeDetection;
