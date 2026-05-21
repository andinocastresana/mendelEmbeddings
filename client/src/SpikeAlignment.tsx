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

import { useEffect, useRef, useState } from 'react';

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
// Math: Umeyama 2D (similitud sin reflexión)
// -----------------------------------------
// Fuente: Umeyama 1991 reducido a 2D. Para puntos en el plano sin permitir
// reflexión, la similitud (rotación + escala uniforme + traslación) se
// puede resolver de forma cerrada SIN SVD:
//
//   c = Σ (src_dx * dst_dx + src_dy * dst_dy)   "alineación"
//   s = Σ (src_dx * dst_dy - src_dy * dst_dx)   "rotación signed"
//   var_src = Σ (src_dx² + src_dy²)
//   norm = sqrt(c² + s²)
//   scale = norm / var_src
//   cos(θ) = c / norm,   sin(θ) = s / norm
//   sR = scale * [[cos, -sin], [sin, cos]]
//   t = mean_dst - sR @ mean_src
//
// Esto es exactamente lo que `skimage.transform.SimilarityTransform.estimate`
// devuelve en 2D (y por ende lo que `face_align.estimate_norm` usa).
function estimateNormSimilarity(
  src: number[][], // n x 2
  dst: number[][], // n x 2
): number[][] {
  if (src.length !== dst.length || src.length === 0) {
    throw new Error(`Mismatched or empty src/dst (src=${src.length}, dst=${dst.length})`);
  }
  const n = src.length;

  // Medias.
  let mxSrc = 0, mySrc = 0, mxDst = 0, myDst = 0;
  for (let i = 0; i < n; i++) {
    mxSrc += src[i][0]; mySrc += src[i][1];
    mxDst += dst[i][0]; myDst += dst[i][1];
  }
  mxSrc /= n; mySrc /= n; mxDst /= n; myDst /= n;

  // Sumas.
  let c = 0, s = 0, varSrc = 0;
  for (let i = 0; i < n; i++) {
    const sx = src[i][0] - mxSrc;
    const sy = src[i][1] - mySrc;
    const dx = dst[i][0] - mxDst;
    const dy = dst[i][1] - myDst;
    c += sx * dx + sy * dy;
    s += sx * dy - sy * dx;
    varSrc += sx * sx + sy * sy;
  }

  if (varSrc === 0) {
    throw new Error('Degenerate src points (zero variance)');
  }

  const norm = Math.sqrt(c * c + s * s);
  if (norm === 0) {
    throw new Error('Degenerate dst alignment (zero rotation magnitude)');
  }

  const scale = norm / varSrc;
  const cosT = c / norm;
  const sinT = s / norm;

  // sR (escala * rotación).
  const a = scale * cosT;
  const b = -scale * sinT;
  const d = scale * sinT;
  const e = scale * cosT;

  // Traslación: t = mean_dst - sR @ mean_src.
  const tx = mxDst - (a * mxSrc + b * mySrc);
  const ty = myDst - (d * mxSrc + e * mySrc);

  return [
    [a, b, tx],
    [d, e, ty],
  ];
}

// -----------------------------------------
// Ajuste de margen (lo que `align_face_from_record` hace post estimate_norm)
// -----------------------------------------
function adjustMatrixForMargin(M: number[][], imageSize: number, marginRatio: number): number[][] {
  if (marginRatio === 0) return M.map(row => [...row]);
  const scale = 1.0 - 2.0 * marginRatio;
  const shift = (imageSize * (1.0 - scale)) / 2.0;
  return [
    [M[0][0] * scale, M[0][1] * scale, M[0][2] * scale + shift],
    [M[1][0] * scale, M[1][1] * scale, M[1][2] * scale + shift],
  ];
}

// -----------------------------------------
// Inversa de matriz afín 2x3 (para warpAffine)
// -----------------------------------------
// cv2.warpAffine recibe M como transformación src → dst, pero internamente
// invierte M y muestrea: para cada (xo, yo) de dst, calcula M^(-1)·(xo,yo,1)
// y samplea src en esas coords (fraccionales) con bilinear.
function invertAffine(M: number[][]): number[][] {
  const [a, b, c] = M[0];
  const [d, e, f] = M[1];
  const det = a * e - b * d;
  if (Math.abs(det) < 1e-12) {
    throw new Error(`Singular affine matrix (det=${det})`);
  }
  const inv = 1.0 / det;
  // Inversa de la parte lineal 2x2.
  const ia = e * inv;
  const ib = -b * inv;
  const id = -d * inv;
  const ie = a * inv;
  // Traslación de la inversa: -inv(A) @ t.
  const itx = -(ia * c + ib * f);
  const ity = -(id * c + ie * f);
  return [
    [ia, ib, itx],
    [id, ie, ity],
  ];
}

// -----------------------------------------
// warpAffine bilineal con BORDER_REPLICATE
// -----------------------------------------
// Emula `cv2.warpAffine(src, M, (Wout, Hout), borderMode=BORDER_REPLICATE)`
// con la interpolación bilineal default de OpenCV (INTER_LINEAR).
//
// Diferencias esperadas vs cv2:
// - cv2 hace interpolación con aritmética fixed-point (más rápida); acá vamos
//   en float64 directo. Puede haber ±1 en uint8 por redondeo.
// - cv2 usa `cvRound` (que es "banker's rounding" en algunas plataformas);
//   acá usamos Math.round() (round half away from zero). Otra fuente de ±1.
function warpAffineBilinearReplicate(
  src: ImageData,
  M: number[][],
  Wout: number,
  Hout: number,
): ImageData {
  const Wsrc = src.width;
  const Hsrc = src.height;
  const srcData = src.data; // Uint8ClampedArray RGBA
  const Minv = invertAffine(M);
  const [ia, ib, itx] = Minv[0];
  const [id, ie, ity] = Minv[1];

  const out = new ImageData(Wout, Hout);
  const outData = out.data;

  for (let yo = 0; yo < Hout; yo++) {
    for (let xo = 0; xo < Wout; xo++) {
      // Coordenadas en src.
      const xs = ia * xo + ib * yo + itx;
      const ys = id * xo + ie * yo + ity;

      // Bilineal: 4 vecinos enteros + pesos fraccionales.
      const x0 = Math.floor(xs);
      const y0 = Math.floor(ys);
      const x1 = x0 + 1;
      const y1 = y0 + 1;
      const fx = xs - x0;
      const fy = ys - y0;

      // BORDER_REPLICATE: clamp a [0, Wsrc-1] x [0, Hsrc-1].
      const cx0 = x0 < 0 ? 0 : x0 >= Wsrc ? Wsrc - 1 : x0;
      const cx1 = x1 < 0 ? 0 : x1 >= Wsrc ? Wsrc - 1 : x1;
      const cy0 = y0 < 0 ? 0 : y0 >= Hsrc ? Hsrc - 1 : y0;
      const cy1 = y1 < 0 ? 0 : y1 >= Hsrc ? Hsrc - 1 : y1;

      // Offsets RGBA (stride = 4).
      const o00 = (cy0 * Wsrc + cx0) * 4;
      const o01 = (cy0 * Wsrc + cx1) * 4;
      const o10 = (cy1 * Wsrc + cx0) * 4;
      const o11 = (cy1 * Wsrc + cx1) * 4;
      const oOut = (yo * Wout + xo) * 4;

      // Interpola por canal (R, G, B). Alpha lo dejamos en 255 (cv2 trabaja en 3 canales).
      const w00 = (1 - fx) * (1 - fy);
      const w01 = fx * (1 - fy);
      const w10 = (1 - fx) * fy;
      const w11 = fx * fy;

      for (let ch = 0; ch < 3; ch++) {
        const v = srcData[o00 + ch] * w00
                + srcData[o01 + ch] * w01
                + srcData[o10 + ch] * w10
                + srcData[o11 + ch] * w11;
        // Redondeo + clamp a uint8.
        const r = Math.round(v);
        outData[oOut + ch] = r < 0 ? 0 : r > 255 ? 255 : r;
      }
      outData[oOut + 3] = 255;
    }
  }

  return out;
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
