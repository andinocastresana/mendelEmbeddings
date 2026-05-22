// =========================================
// ID: PHYLOFACE_SPIKE_001
// VERSION: v1.1
// =========================================
// Cambio v1.0 → v1.1 (Tarea #27, bugfix calentamiento sostenido):
// - Cleanup de la `InferenceSession` ONNX al desmontar. Sin esto, el contexto
//   WebGPU/WASM persiste en el proceso GPU compartido del browser hasta
//   refresh de página.
//
// Componente del SPIKE Track 2a — paridad JS/Python.
//
// Qué hace:
// - Carga `w600k_r50.onnx` (modelo de embedding facial ArcFace) en el browser
//   vía onnxruntime-web (backends: webgpu > wasm).
// - Carga el fixture generado por `scripts/verify_onnx_web_parity.py`:
//     * reference_embedding.json  -> embedding de referencia (512-D float32)
//     * aligned_face_tensor.json  -> tensor pre-procesado (1,3,112,112)
//     * aligned_face.png          -> imagen alineada 112x112 RGB
//     * metadata.json             -> normalization params + criterio de éxito
// - Corre dos paths de inferencia:
//     Path A "easy": tensor pre-procesado del JSON -> modelo.
//                   Valida que el MODELO en JS produce los mismos embeddings.
//     Path B "completo": cargar PNG -> preprocesar en JS -> modelo.
//                   Valida también que el PREPROCESSING JS es correcto.
// - Compara cada embedding JS con el de referencia (cosine + max |diff|).
// - Muestra PASS/FAIL según criterios del fixture metadata.
//
// Criterio de éxito (del v0.1 §10):
//     cosine_similarity > 0.9999  AND  max |diff| < 1e-3

import { useEffect, useState } from 'react';
import * as ort from 'onnxruntime-web';

// -----------------------------------------
// Tipos del fixture
// -----------------------------------------
interface ReferenceEmbedding {
  shape: number[];
  dtype: string;
  norm_l2: number;
  values: number[];
}

interface AlignedFaceTensor {
  shape: number[];
  dtype: string;
  layout: string;
  normalization: string;
  input_mean: number;
  input_std: number;
  data_flat: number[];
}

interface FixtureMetadata {
  generated_at_utc: string;
  source_image_path: string;
  model: { library: string; bundle: string; recognition_submodel: string };
  alignment: { image_size: number; margin_ratio: number; method: string };
  preprocessing: {
    color: string;
    normalize: string;
    input_mean: number;
    input_std: number;
    layout: string;
    output_shape: number[];
    note: string;
  };
  sanity_check_manual_vs_get_feat: {
    cosine_similarity: number;
    max_abs_diff: number;
    passed: boolean;
  };
  success_criteria_for_js_client: {
    cosine_similarity_min: number;
    max_abs_diff_max: number;
  };
}

interface ComparisonResult {
  path: string;
  cosine: number;
  maxAbsDiff: number;
  passed: boolean;
  inferMs: number;
}

// -----------------------------------------
// Helpers de math
// -----------------------------------------
function cosineSimilarity(a: Float32Array, b: Float32Array): number {
  // Coseno = (a·b) / (||a|| * ||b||). Definimos sin l2_normalize separado
  // para evitar el costo de crear arrays intermedios.
  if (a.length !== b.length) throw new Error('Length mismatch');
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB) + 1e-8);
}

function maxAbsDiff(a: Float32Array, b: Float32Array): number {
  let maxDiff = 0;
  for (let i = 0; i < a.length; i++) {
    const d = Math.abs(a[i] - b[i]);
    if (d > maxDiff) maxDiff = d;
  }
  return maxDiff;
}

// -----------------------------------------
// Cargar PNG y convertir a tensor CHW float32 normalizado
// -----------------------------------------
// Replica EXACTAMENTE el preprocessing del script Python (que ya validamos
// que matchea get_feat de InsightFace post-bugfix RGB->BGR).
async function pngToPreprocessedTensor(
  pngUrl: string,
  mean: number,
  std: number,
): Promise<Float32Array> {
  // Cargar imagen.
  const img = new Image();
  img.crossOrigin = 'anonymous';
  await new Promise<void>((resolve, reject) => {
    img.onload = () => resolve();
    img.onerror = (e) => reject(new Error(`Failed to load image: ${e}`));
    img.src = pngUrl;
  });
  if (img.width !== 112 || img.height !== 112) {
    throw new Error(`Expected 112x112, got ${img.width}x${img.height}`);
  }

  // Dibujar al canvas y extraer píxeles RGBA (8-bit per channel).
  const canvas = document.createElement('canvas');
  canvas.width = 112;
  canvas.height = 112;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  if (!ctx) throw new Error('Cannot get 2D context');
  ctx.drawImage(img, 0, 0);
  const imageData = ctx.getImageData(0, 0, 112, 112);
  const rgba = imageData.data; // Uint8ClampedArray length 112*112*4

  // Convertir RGBA -> CHW float32 normalizado, orden de canales RGB.
  //
  // Razonamiento (versión correcta, descubierta por error en el spike):
  // - El modelo w600k_r50.onnx fue entrenado viendo RGB (porque InsightFace
  //   en training usa cv2.dnn.blobFromImage(BGR_input, swapRB=True), que
  //   convierte el BGR de cv2.imread a RGB antes del modelo).
  // - El Path A del spike (PASS perfecto: cosine=1.0) usa el tensor JSON que
  //   Python genera con `preprocess_for_recognition(aligned_rgb)`, el cual
  //   pone los canales en orden R, G, B (sin swap). Eso confirma que el
  //   modelo espera RGB en su input.
  // - En JS, cuando hacemos session.run() directo, NO hay `swapRB` interno.
  //   El modelo recibe exactamente los bytes que le damos. Por lo tanto,
  //   debemos entregarle RGB directo (sin invertir).
  // - El canvas devuelve RGBA en orden R, G, B, A → los usamos tal cual.
  //
  // Anécdota del spike: la primera versión de este código invertía a BGR
  // por mala simetría con el bugfix Python (en Python se invierte porque
  // get_feat hace swap interno; en JS NO hay swap). Resultado: cosine =
  // 0.953915 (exactamente la firma del bug "canales invertidos" que vimos
  // en Python pre-bugfix). Con la inversión corregida, cosine = 1.0.
  const out = new Float32Array(1 * 3 * 112 * 112);
  const planeSize = 112 * 112;
  for (let y = 0; y < 112; y++) {
    for (let x = 0; x < 112; x++) {
      const pxIdx = (y * 112 + x) * 4;
      const r = rgba[pxIdx + 0];
      const g = rgba[pxIdx + 1];
      const b = rgba[pxIdx + 2];
      // CHW layout, sin swap: canal 0 = R, canal 1 = G, canal 2 = B.
      // Normalización: (px - mean) / std.
      const outIdx = y * 112 + x;
      out[0 * planeSize + outIdx] = (r - mean) / std;
      out[1 * planeSize + outIdx] = (g - mean) / std;
      out[2 * planeSize + outIdx] = (b - mean) / std;
    }
  }
  return out;
}

// -----------------------------------------
// Componente principal
// -----------------------------------------
function SpikeOnnx() {
  const [steps, setSteps] = useState<string[]>([]);
  const [metadata, setMetadata] = useState<FixtureMetadata | null>(null);
  const [results, setResults] = useState<ComparisonResult[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [done, setDone] = useState(false);

  const log = (msg: string) => {
    console.log(msg);
    setSteps((prev) => [...prev, msg]);
  };

  useEffect(() => {
    let cancelled = false;
    // Capturado para liberar en cleanup (Tarea #27).
    let sessionInstance: ort.InferenceSession | null = null;

    (async () => {
      try {
        log('[1] Cargando metadata del fixture...');
        const metaResp = await fetch('/spike_fixtures/metadata.json');
        const meta: FixtureMetadata = await metaResp.json();
        if (cancelled) return;
        setMetadata(meta);
        log(`    Modelo: ${meta.model.library}/${meta.model.bundle}/${meta.model.recognition_submodel}`);
        log(`    Preprocessing: ${meta.preprocessing.normalize}`);
        log(`    Sanity (Py manual vs get_feat): cos=${meta.sanity_check_manual_vs_get_feat.cosine_similarity.toFixed(6)}`);
        log(`    Criterio JS: cos > ${meta.success_criteria_for_js_client.cosine_similarity_min}, max|diff| < ${meta.success_criteria_for_js_client.max_abs_diff_max}`);

        log('[2] Cargando reference embedding...');
        const refResp = await fetch('/spike_fixtures/reference_embedding.json');
        const ref: ReferenceEmbedding = await refResp.json();
        const refEmb = new Float32Array(ref.values);
        // Array.from convierte Float32Array -> number[] (Float32Array.map requiere callback que devuelva number).
        log(`    shape=${ref.shape}, norm=${ref.norm_l2.toFixed(4)}, first5=[${Array.from(refEmb.slice(0,5)).map(v=>v.toFixed(4)).join(', ')}]`);

        log('[3] Cargando tensor pre-procesado (Path A)...');
        const tensorResp = await fetch('/spike_fixtures/aligned_face_tensor.json');
        const tensorJson: AlignedFaceTensor = await tensorResp.json();
        const tensorData = new Float32Array(tensorJson.data_flat);
        log(`    shape=${tensorJson.shape}, ${tensorJson.normalization}`);

        log('[4] Cargando modelo ONNX (174 MB, primera vez tarda; siguientes se cachean)...');
        const t0 = performance.now();
        const session = await ort.InferenceSession.create('/models/w600k_r50.onnx', {
          // Preferimos WebGPU > WASM. WebGPU es más rápido cuando está disponible;
          // WASM funciona en todos los browsers como fallback.
          executionProviders: ['webgpu', 'wasm'],
          graphOptimizationLevel: 'all',
        });
        sessionInstance = session;
        if (cancelled) return;
        const loadMs = performance.now() - t0;
        log(`    Modelo cargado en ${loadMs.toFixed(0)} ms.`);
        log(`    Input: ${session.inputNames.join(',')} | Output: ${session.outputNames.join(',')}`);

        // -----------------------------------------
        // Path A: tensor pre-procesado del JSON
        // -----------------------------------------
        log('[5] Path A (easy): tensor pre-procesado JSON -> modelo...');
        const inputName = session.inputNames[0];
        const outputName = session.outputNames[0];

        const tensorA = new ort.Tensor('float32', tensorData, [1, 3, 112, 112]);
        const t1 = performance.now();
        const outputsA = await session.run({ [inputName]: tensorA });
        const inferMsA = performance.now() - t1;
        const embAOut = outputsA[outputName].data as Float32Array;
        const embA = new Float32Array(embAOut);
        const cosA = cosineSimilarity(embA, refEmb);
        const diffA = maxAbsDiff(embA, refEmb);
        const passedA = cosA > meta.success_criteria_for_js_client.cosine_similarity_min &&
                        diffA < meta.success_criteria_for_js_client.max_abs_diff_max;
        log(`    cosine=${cosA.toFixed(6)} | max|diff|=${diffA.toExponential(3)} | infer=${inferMsA.toFixed(0)}ms | ${passedA ? 'PASS ✓' : 'FAIL ✗'}`);

        // -----------------------------------------
        // Path B: PNG -> preprocessing JS -> modelo
        // -----------------------------------------
        log('[6] Path B (completo): PNG -> preprocessing JS -> modelo...');
        const t2 = performance.now();
        const tensorBData = await pngToPreprocessedTensor(
          '/spike_fixtures/aligned_face.png',
          meta.preprocessing.input_mean,
          meta.preprocessing.input_std,
        );
        const preprocMs = performance.now() - t2;
        log(`    PNG cargado y preprocesado en ${preprocMs.toFixed(0)} ms.`);

        const tensorB = new ort.Tensor('float32', tensorBData, [1, 3, 112, 112]);
        const t3 = performance.now();
        const outputsB = await session.run({ [inputName]: tensorB });
        const inferMsB = performance.now() - t3;
        const embBOut = outputsB[outputName].data as Float32Array;
        const embB = new Float32Array(embBOut);
        const cosB = cosineSimilarity(embB, refEmb);
        const diffB = maxAbsDiff(embB, refEmb);
        const passedB = cosB > meta.success_criteria_for_js_client.cosine_similarity_min &&
                        diffB < meta.success_criteria_for_js_client.max_abs_diff_max;
        log(`    cosine=${cosB.toFixed(6)} | max|diff|=${diffB.toExponential(3)} | infer=${inferMsB.toFixed(0)}ms | ${passedB ? 'PASS ✓' : 'FAIL ✗'}`);

        // -----------------------------------------
        // Path A vs Path B (debe ser cuasi idénticos)
        // -----------------------------------------
        const cosAB = cosineSimilarity(embA, embB);
        const diffAB = maxAbsDiff(embA, embB);
        log(`[7] Path A vs Path B: cosine=${cosAB.toFixed(6)} | max|diff|=${diffAB.toExponential(3)}`);

        if (cancelled) return;
        setResults([
          { path: 'A: tensor JSON -> modelo', cosine: cosA, maxAbsDiff: diffA, passed: passedA, inferMs: inferMsA },
          { path: 'B: PNG -> preprocess JS -> modelo', cosine: cosB, maxAbsDiff: diffB, passed: passedB, inferMs: inferMsB },
        ]);
        setDone(true);

        log('');
        if (passedA && passedB) {
          log('[OK] Spike Track 2a EXITOSO. ONNX Runtime Web reproduce embeddings de Python en el browser.');
          log('     -> Plan híbrido v0.1 confirmado viable. Avanzar con Track 2a completo.');
        } else if (passedA && !passedB) {
          log('[!!] Path A PASS, Path B FAIL -> problema en el preprocessing JS (no en el modelo).');
        } else if (!passedA && passedB) {
          log('[??] Path A FAIL pero Path B PASS -> raro. Investigar.');
        } else {
          log('[!!] Ambos paths FAIL -> investigar backend (WebGPU vs WASM), versión de onnxruntime-web, modelo.');
        }
      } catch (e: any) {
        const errMsg = `[ERROR] ${e?.message || String(e)}`;
        console.error(errMsg, e);
        if (!cancelled) {
          setError(errMsg);
          log(errMsg);
        }
      }
    })();

    return () => {
      cancelled = true;
      const sess = sessionInstance;
      sessionInstance = null;
      if (sess) void sess.release().catch((e) => console.warn('[SpikeOnnx] session.release falló:', e));
    };
  }, []);

  // -----------------------------------------
  // Render
  // -----------------------------------------
  return (
    <div style={{ fontFamily: 'monospace', padding: '20px', maxWidth: 1000, margin: '0 auto' }}>
      <h1 style={{ borderBottom: '2px solid #333', paddingBottom: 8 }}>
        Spike Track 2a — ONNX Runtime Web parity check
      </h1>
      <p style={{ color: '#666', fontSize: 13 }}>
        Verifica que el modelo <code>w600k_r50.onnx</code> corre en el browser y produce embeddings cuasi
        bit-idénticos a la implementación Python. Resultado decide si el plan híbrido v0.1 es viable.
      </p>

      <h2 style={{ marginTop: 20 }}>Log</h2>
      <pre style={{
        background: '#f4f4f4',
        padding: 12,
        borderRadius: 4,
        fontSize: 12,
        lineHeight: 1.5,
        whiteSpace: 'pre-wrap',
        maxHeight: 500,
        overflow: 'auto',
      }}>
        {steps.join('\n')}
        {!done && !error && <span style={{ color: '#888' }}>{'\n…corriendo…'}</span>}
      </pre>

      {error && (
        <div style={{ background: '#fee', color: '#900', padding: 12, borderRadius: 4, marginTop: 12 }}>
          <strong>Error:</strong> {error}
        </div>
      )}

      {results.length > 0 && (
        <>
          <h2 style={{ marginTop: 20 }}>Resultados</h2>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ background: '#f0f4f8' }}>
                <th style={{ border: '1px solid #ccc', padding: 8, textAlign: 'left' }}>Path</th>
                <th style={{ border: '1px solid #ccc', padding: 8, textAlign: 'right' }}>cosine</th>
                <th style={{ border: '1px solid #ccc', padding: 8, textAlign: 'right' }}>max |diff|</th>
                <th style={{ border: '1px solid #ccc', padding: 8, textAlign: 'right' }}>infer (ms)</th>
                <th style={{ border: '1px solid #ccc', padding: 8, textAlign: 'center' }}>Status</th>
              </tr>
            </thead>
            <tbody>
              {results.map((r, i) => (
                <tr key={i} style={{ background: r.passed ? '#efffef' : '#ffeeee' }}>
                  <td style={{ border: '1px solid #ccc', padding: 8 }}>{r.path}</td>
                  <td style={{ border: '1px solid #ccc', padding: 8, textAlign: 'right' }}>{r.cosine.toFixed(6)}</td>
                  <td style={{ border: '1px solid #ccc', padding: 8, textAlign: 'right' }}>{r.maxAbsDiff.toExponential(3)}</td>
                  <td style={{ border: '1px solid #ccc', padding: 8, textAlign: 'right' }}>{r.inferMs.toFixed(0)}</td>
                  <td style={{ border: '1px solid #ccc', padding: 8, textAlign: 'center', fontWeight: 700 }}>
                    {r.passed ? '✓ PASS' : '✗ FAIL'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
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
    </div>
  );
}

export default SpikeOnnx;
