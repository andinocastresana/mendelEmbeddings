// =========================================
// ID: PHYLOFACE_SPIKE_003
// VERSION: v1.0
// =========================================
// Componente del SPIKE — paridad de la alineación canónica JS vs Python.
//
// Qué hace:
// - Carga el fixture generado por `scripts/verify_alignment_web_parity.py`:
//     * crop_rgb.png            -> imagen de entrada al algoritmo de alineación.
//     * aligned_face_112.png    -> referencia: salida esperada (la que produce Python).
//     * landmarks.json          -> 5 kps_local + template arcface_dst + matrices M / M_adj.
//     * metadata.json           -> criterio de éxito + parámetros.
// - Implementa en TS los dos pasos del algoritmo:
//     1) `estimateNormSimilarity` — Umeyama 2D cerrado (rotación + escala
//        uniforme + traslación, sin reflexión) que mapea los 5 kps al
//        template ArcFace y devuelve la matriz afín 2x3.
//     2) `warpAffineBilinearReplicate` — emula `cv2.warpAffine` con
//        `BORDER_REPLICATE` + interpolación bilineal.
// - Corre DOS paths de verificación (igual estructura que SPIKE_001):
//     * Path "easy"     : usa M_adj del fixture -> solo testea warpAffine.
//     * Path "completo" : estima M en JS con Umeyama + warpAffine.
//   Si easy pasa y completo falla -> bug está en Umeyama JS.
//   Si easy falla -> bug en warpAffine JS (interpolación o borderMode).
// - Compara la imagen JS contra `aligned_face_112.png`:
//     mean_abs_pixel_diff y max_abs_pixel_diff sobre los 112*112*3 píxeles uint8.
// - Muestra PASS/FAIL según el criterio del fixture (mean<1.0, max<=5).
// - Renderiza overlay: [referencia | JS | diff amplificado x10] para inspección.
//
// Razón del threshold "loose" (no exact match):
// cv2 implementa la interpolación bilineal con aritmética entera de punto fijo
// por velocidad; una implementación TS en flotante pura puede tener diferencias
// de 1-2 sobre uint8 en píxeles de borde por redondeo. NO permite desvíos de
// algoritmo (Umeyama o transformación de coords), solo ruido numérico.
//
// Refactor (post spike #003 PASS): los 4 helpers de alineación
// (`estimateNormSimilarity`, `adjustMatrixForMargin`, `invertAffine`,
// `warpAffineBilinearReplicate`) viven en `./lib/alignment.ts` para ser
// reusados por el spike #004 (pipeline e2e) y la UI MVP futura. Este spike
// los importa desde ahí. Sigue construyendo `dstScaled` manualmente a partir
// del template que viene EN EL FIXTURE (no del lib), a propósito: así valida
// Umeyama JS independientemente del template hardcoded en el cliente.

import { useEffect, useRef, useState } from 'react';
import {
  estimateNormSimilarity,
  adjustMatrixForMargin,
  warpAffineBilinearReplicate,
} from './lib/alignment';

// -----------------------------------------
// Tipos del fixture
// -----------------------------------------
interface LandmarksFixture {
  kps_local: {
    shape: number[];
    crop_size_wh: number[];
    values: number[][]; // 5 x [x, y]
    order: string;
  };
  arcface_dst_template: {
    image_size_reference: number;
    values: number[][]; // 5 x [x, y] en sistema 112x112
  };
  reference_matrix_M: { values: number[][] };       // 2x3
  reference_matrix_M_adj: { values: number[][] };   // 2x3
}

interface FixtureMetadata {
  alignment: {
    image_size: number;
    margin_ratio: number;
    method: string;
    interpolation: string;
    border_mode: string;
  };
  success_criteria_for_js_client: {
    mean_abs_pixel_diff_max: number;
    max_abs_pixel_diff_max: number;
    shape_must_match: number[];
    dtype_must_match: string;
  };
}

interface PathResult {
  label: string;
  meanAbsDiff: number;
  maxAbsDiff: number;
  passed: boolean;
  // Matriz usada (para mostrar en el log).
  M: number[][];
  // Para path completo: diferencia entre M_js y M_ref.
  matrixMaxAbsDiff?: number;
  // ImageData resultante para renderizar.
  imageData: ImageData;
}

// -----------------------------------------
// Utilidad: cargar PNG como ImageData
// -----------------------------------------
async function loadPngAsImageData(url: string): Promise<ImageData> {
  const img = new Image();
  img.crossOrigin = 'anonymous';
  await new Promise<void>((resolve, reject) => {
    img.onload = () => resolve();
    img.onerror = (e) => reject(new Error(`Failed to load image at ${url}: ${e}`));
    img.src = url;
  });
  const canvas = document.createElement('canvas');
  canvas.width = img.width;
  canvas.height = img.height;
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('Could not get 2D context');
  ctx.drawImage(img, 0, 0);
  return ctx.getImageData(0, 0, img.width, img.height);
}

// -----------------------------------------
// Utilidad: comparar dos ImageData por canales RGB (ignora alpha)
// -----------------------------------------
function compareImageData(a: ImageData, b: ImageData): { meanAbs: number; maxAbs: number } {
  if (a.width !== b.width || a.height !== b.height) {
    throw new Error(`Shape mismatch: ${a.width}x${a.height} vs ${b.width}x${b.height}`);
  }
  const n = a.width * a.height;
  let sum = 0;
  let max = 0;
  for (let i = 0; i < n; i++) {
    const off = i * 4;
    for (let ch = 0; ch < 3; ch++) {
      const d = Math.abs(a.data[off + ch] - b.data[off + ch]);
      sum += d;
      if (d > max) max = d;
    }
  }
  return { meanAbs: sum / (n * 3), maxAbs: max };
}

// -----------------------------------------
// Diff map amplificado (para inspección visual)
// -----------------------------------------
function buildDiffMap(a: ImageData, b: ImageData, amplify: number = 10): ImageData {
  const out = new ImageData(a.width, a.height);
  const n = a.width * a.height;
  for (let i = 0; i < n; i++) {
    const off = i * 4;
    for (let ch = 0; ch < 3; ch++) {
      const d = Math.abs(a.data[off + ch] - b.data[off + ch]) * amplify;
      out.data[off + ch] = d > 255 ? 255 : d;
    }
    out.data[off + 3] = 255;
  }
  return out;
}

function maxAbsMatrixDiff(A: number[][], B: number[][]): number {
  let m = 0;
  for (let i = 0; i < A.length; i++) {
    for (let j = 0; j < A[i].length; j++) {
      const d = Math.abs(A[i][j] - B[i][j]);
      if (d > m) m = d;
    }
  }
  return m;
}

// -----------------------------------------
// Componente
// -----------------------------------------
function SpikeAlignment() {
  const [steps, setSteps] = useState<string[]>([]);
  const [metadata, setMetadata] = useState<FixtureMetadata | null>(null);
  const [pathEasy, setPathEasy] = useState<PathResult | null>(null);
  const [pathFull, setPathFull] = useState<PathResult | null>(null);
  const [reference, setReference] = useState<ImageData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [done, setDone] = useState(false);

  const canvasRefRef = useRef<HTMLCanvasElement | null>(null);
  const canvasJsRef = useRef<HTMLCanvasElement | null>(null);
  const canvasDiffRef = useRef<HTMLCanvasElement | null>(null);

  const log = (msg: string) => {
    console.log(msg);
    setSteps((prev) => [...prev, msg]);
  };

  useEffect(() => {
    let cancelled = false;

    (async () => {
      try {
        log('[1] Cargando metadata del fixture alignment...');
        const metaResp = await fetch('/spike_fixtures_alignment/metadata.json');
        const meta: FixtureMetadata = await metaResp.json();
        if (cancelled) return;
        setMetadata(meta);
        log(`    image_size=${meta.alignment.image_size}, margin_ratio=${meta.alignment.margin_ratio}`);
        log(`    método: ${meta.alignment.method}`);
        log(`    criterio: mean<${meta.success_criteria_for_js_client.mean_abs_pixel_diff_max}, max<=${meta.success_criteria_for_js_client.max_abs_pixel_diff_max}`);

        log('[2] Cargando landmarks + matrices de referencia...');
        const lmkResp = await fetch('/spike_fixtures_alignment/landmarks.json');
        const lmk: LandmarksFixture = await lmkResp.json();
        const kpsLocal = lmk.kps_local.values;
        const arcfaceDst = lmk.arcface_dst_template.values;
        const Mref = lmk.reference_matrix_M.values;
        const MadjRef = lmk.reference_matrix_M_adj.values;
        log(`    kps_local (5x2): ${kpsLocal.map(p => `(${p[0].toFixed(1)},${p[1].toFixed(1)})`).join(' ')}`);
        log(`    M_ref     row0:  [${Mref[0].map(v => v.toFixed(6)).join(', ')}]`);
        log(`    M_ref     row1:  [${Mref[1].map(v => v.toFixed(6)).join(', ')}]`);

        log('[3] Cargando crop_rgb.png (entrada al algoritmo)...');
        const cropImg = await loadPngAsImageData('/spike_fixtures_alignment/crop_rgb.png');
        log(`    crop: ${cropImg.width}x${cropImg.height}`);

        log('[4] Cargando aligned_face_112.png (referencia Python)...');
        const refImg = await loadPngAsImageData('/spike_fixtures_alignment/aligned_face_112.png');
        if (cancelled) return;
        setReference(refImg);
        log(`    ref:  ${refImg.width}x${refImg.height}`);

        const imageSize = meta.alignment.image_size;
        const marginRatio = meta.alignment.margin_ratio;
        const crit = meta.success_criteria_for_js_client;

        // ---------------------------------------
        // Path "easy": usar M_adj del fixture, solo testear warpAffine.
        // ---------------------------------------
        log('[5/A] Path EASY: usando M_adj del fixture -> solo testea warpAffine JS...');
        const t0 = performance.now();
        const jsEasy = warpAffineBilinearReplicate(cropImg, MadjRef, imageSize, imageSize);
        const tEasy = performance.now() - t0;
        const cmpEasy = compareImageData(jsEasy, refImg);
        const passEasy = cmpEasy.meanAbs < crit.mean_abs_pixel_diff_max
                       && cmpEasy.maxAbs <= crit.max_abs_pixel_diff_max;
        log(`      warp time: ${tEasy.toFixed(1)} ms`);
        log(`      mean_abs_diff = ${cmpEasy.meanAbs.toFixed(4)}   (criterio < ${crit.mean_abs_pixel_diff_max})  ${cmpEasy.meanAbs < crit.mean_abs_pixel_diff_max ? '✓' : '✗'}`);
        log(`      max_abs_diff  = ${cmpEasy.maxAbs}   (criterio <= ${crit.max_abs_pixel_diff_max})  ${cmpEasy.maxAbs <= crit.max_abs_pixel_diff_max ? '✓' : '✗'}`);
        if (cancelled) return;
        setPathEasy({
          label: 'easy (M del fixture)',
          meanAbsDiff: cmpEasy.meanAbs,
          maxAbsDiff: cmpEasy.maxAbs,
          passed: passEasy,
          M: MadjRef,
          imageData: jsEasy,
        });

        // ---------------------------------------
        // Path "completo": estimar M en JS con Umeyama, luego warpAffine.
        // ---------------------------------------
        // Para image_size=112, ratio=1, diff_x=0 -> dst = arcface_dst tal cual.
        // Para image_size múltiplo de 128, habría que recalcular. Acotamos al
        // caso del spike (112) y dejamos un assert explícito.
        log('[5/B] Path COMPLETO: Umeyama JS sobre kps_local + arcface_dst -> warpAffine JS...');
        if (imageSize % 112 !== 0) {
          throw new Error(`Este spike asume image_size múltiplo de 112; recibí ${imageSize}.`);
        }
        const ratio = imageSize / 112;
        const dstScaled = arcfaceDst.map(([x, y]) => [x * ratio, y * ratio]);

        const Mjs = estimateNormSimilarity(kpsLocal, dstScaled);
        const MadjJs = adjustMatrixForMargin(Mjs, imageSize, marginRatio);

        const matrixDiff = maxAbsMatrixDiff(MadjJs, MadjRef);
        log(`      M_js     row0: [${Mjs[0].map(v => v.toFixed(6)).join(', ')}]`);
        log(`      M_js     row1: [${Mjs[1].map(v => v.toFixed(6)).join(', ')}]`);
        log(`      max |M_js - M_ref| (entry-wise) = ${matrixDiff.toExponential(3)}`);

        const t1 = performance.now();
        const jsFull = warpAffineBilinearReplicate(cropImg, MadjJs, imageSize, imageSize);
        const tFull = performance.now() - t1;
        const cmpFull = compareImageData(jsFull, refImg);
        const passFull = cmpFull.meanAbs < crit.mean_abs_pixel_diff_max
                       && cmpFull.maxAbs <= crit.max_abs_pixel_diff_max;
        log(`      warp time: ${tFull.toFixed(1)} ms`);
        log(`      mean_abs_diff = ${cmpFull.meanAbs.toFixed(4)}   (criterio < ${crit.mean_abs_pixel_diff_max})  ${cmpFull.meanAbs < crit.mean_abs_pixel_diff_max ? '✓' : '✗'}`);
        log(`      max_abs_diff  = ${cmpFull.maxAbs}   (criterio <= ${crit.max_abs_pixel_diff_max})  ${cmpFull.maxAbs <= crit.max_abs_pixel_diff_max ? '✓' : '✗'}`);
        if (cancelled) return;
        setPathFull({
          label: 'completo (Umeyama JS + warpAffine JS)',
          meanAbsDiff: cmpFull.meanAbs,
          maxAbsDiff: cmpFull.maxAbs,
          passed: passFull,
          M: MadjJs,
          matrixMaxAbsDiff: matrixDiff,
          imageData: jsFull,
        });

        log('');
        if (passEasy && passFull) {
          log('[OK] Spike alignment EXITOSO en ambos paths.');
          log('     -> warpAffine JS y Umeyama JS reproducen `align_face_from_record` dentro del threshold.');
          log('     -> El comparador browser puede armarse: detect (futuro) -> alineación JS -> ONNX -> embedding.');
        } else if (passEasy && !passFull) {
          log('[!!] FAIL completo / PASS easy: el warpAffine está bien, pero Umeyama JS difiere de Python.');
        } else if (!passEasy && !passFull) {
          log('[!!] FAIL en ambos: revisar warpAffine (interpolación, borderMode, redondeo).');
        } else {
          log('[!!] Resultado raro: pasa completo pero no easy. Revisar la implementación.');
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

  // Render de los 3 canvases (ref | js | diff) cuando hay datos.
  useEffect(() => {
    if (!reference) return;
    const ctx = canvasRefRef.current?.getContext('2d');
    if (ctx && canvasRefRef.current) {
      canvasRefRef.current.width = reference.width;
      canvasRefRef.current.height = reference.height;
      ctx.putImageData(reference, 0, 0);
    }
  }, [reference]);

  useEffect(() => {
    if (!pathFull) return;
    const ctx = canvasJsRef.current?.getContext('2d');
    if (ctx && canvasJsRef.current) {
      canvasJsRef.current.width = pathFull.imageData.width;
      canvasJsRef.current.height = pathFull.imageData.height;
      ctx.putImageData(pathFull.imageData, 0, 0);
    }
  }, [pathFull]);

  useEffect(() => {
    if (!pathFull || !reference) return;
    const diff = buildDiffMap(pathFull.imageData, reference, 10);
    const ctx = canvasDiffRef.current?.getContext('2d');
    if (ctx && canvasDiffRef.current) {
      canvasDiffRef.current.width = diff.width;
      canvasDiffRef.current.height = diff.height;
      ctx.putImageData(diff, 0, 0);
    }
  }, [pathFull, reference]);

  // -----------------------------------------
  // Render
  // -----------------------------------------
  const renderResult = (r: PathResult | null, criterion: FixtureMetadata['success_criteria_for_js_client'] | null) => {
    if (!r || !criterion) return null;
    const rowStyle = { border: '1px solid #ccc', padding: 6 };
    return (
      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13, marginBottom: 8 }}>
        <tbody>
          <tr><td style={rowStyle}><b>Path</b></td><td style={rowStyle}>{r.label}</td></tr>
          <tr><td style={rowStyle}>mean_abs_pixel_diff</td><td style={rowStyle}>{r.meanAbsDiff.toFixed(4)} <span style={{ color: '#666' }}>(criterio &lt; {criterion.mean_abs_pixel_diff_max})</span></td></tr>
          <tr><td style={rowStyle}>max_abs_pixel_diff</td><td style={rowStyle}>{r.maxAbsDiff} <span style={{ color: '#666' }}>(criterio ≤ {criterion.max_abs_pixel_diff_max})</span></td></tr>
          {r.matrixMaxAbsDiff !== undefined && (
            <tr><td style={rowStyle}>max |M_js − M_ref| (entry-wise)</td><td style={rowStyle}>{r.matrixMaxAbsDiff.toExponential(3)}</td></tr>
          )}
          <tr style={{ background: r.passed ? '#efffef' : '#ffeeee', fontWeight: 700 }}>
            <td style={rowStyle}>Status</td>
            <td style={rowStyle}>{r.passed ? '✓ PASS' : '✗ FAIL'}</td>
          </tr>
        </tbody>
      </table>
    );
  };

  return (
    <div style={{ fontFamily: 'monospace', padding: '20px', maxWidth: 1000, margin: '0 auto' }}>
      <h1 style={{ borderBottom: '2px solid #333', paddingBottom: 8 }}>
        Spike alignment — paridad de la alineación canónica JS vs Python
      </h1>
      <p style={{ color: '#666', fontSize: 13 }}>
        Verifica que el algoritmo de <code>align_face_from_record</code>
        (<code>estimate_norm</code> Umeyama + <code>warpAffine</code> bilineal con
        <code>BORDER_REPLICATE</code>) reimplementado en TS produce la misma
        imagen 112×112 pixel-a-pixel que la versión Python. Es la pieza
        intermedia del pipeline browser entre los landmarks (spike #002) y el
        embedding ONNX (spike #001).
      </p>

      <h2 style={{ marginTop: 20 }}>Log</h2>
      <pre style={{
        background: '#f4f4f4', padding: 12, borderRadius: 4, fontSize: 12,
        lineHeight: 1.5, whiteSpace: 'pre-wrap', maxHeight: 500, overflow: 'auto',
      }}>
        {steps.join('\n')}
        {!done && !error && <span style={{ color: '#888' }}>{'\n…corriendo…'}</span>}
      </pre>

      {error && (
        <div style={{ background: '#fee', color: '#900', padding: 12, borderRadius: 4, marginTop: 12 }}>
          <strong>Error:</strong> {error}
        </div>
      )}

      <h2 style={{ marginTop: 20 }}>Resultados</h2>
      {renderResult(pathEasy, metadata?.success_criteria_for_js_client ?? null)}
      {renderResult(pathFull, metadata?.success_criteria_for_js_client ?? null)}

      <h2 style={{ marginTop: 20 }}>Comparación visual</h2>
      <p style={{ fontSize: 12, color: '#666' }}>
        Izq: <b>referencia Python</b>. Centro: <b>resultado JS</b> (path completo).
        Der: <b>diff amplificado ×10</b> (negro = idéntico; manchas claras = píxeles divergentes).
      </p>
      <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
        <div>
          <div style={{ fontSize: 11, color: '#666' }}>aligned_face_112.png (Python)</div>
          <canvas ref={canvasRefRef} style={{ border: '1px solid #ccc', width: 224, height: 224, imageRendering: 'pixelated' }} />
        </div>
        <div>
          <div style={{ fontSize: 11, color: '#666' }}>JS (Umeyama + warpAffine)</div>
          <canvas ref={canvasJsRef} style={{ border: '1px solid #ccc', width: 224, height: 224, imageRendering: 'pixelated' }} />
        </div>
        <div>
          <div style={{ fontSize: 11, color: '#666' }}>|JS − Python| × 10</div>
          <canvas ref={canvasDiffRef} style={{ border: '1px solid #ccc', width: 224, height: 224, imageRendering: 'pixelated' }} />
        </div>
      </div>

      {metadata && (
        <details style={{ marginTop: 20, fontSize: 12 }}>
          <summary style={{ cursor: 'pointer', fontWeight: 600 }}>Fixture metadata</summary>
          <pre style={{ background: '#f4f4f4', padding: 10, borderRadius: 4, overflow: 'auto' }}>
            {JSON.stringify(metadata, null, 2)}
          </pre>
        </details>
      )}
    </div>
  );
}

export default SpikeAlignment;
