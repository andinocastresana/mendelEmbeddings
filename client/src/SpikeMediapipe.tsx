// =========================================
// ID: PHYLOFACE_SPIKE_002
// VERSION: v1.1
// =========================================
// Cambio v1.0 → v1.1 (Tarea #27, bugfix calentamiento sostenido):
// - Cleanup del `FaceLandmarker` al desmontar el componente. Sin esto, el
//   contexto GPU del Face Mesh queda vivo en el proceso GPU compartido del
//   browser hasta refresh de página.
//
// Componente del SPIKE menor — paridad MediaPipe Face Mesh JS vs Python.
//
// Qué hace:
// - Carga MediaPipe Tasks for Web (model "face_landmarker" via CDN de Google).
// - Carga el fixture generado por `scripts/verify_mediapipe_web_parity.py`:
//     * aligned_face_224.png      -> imagen 224x224 RGB alineada
//     * reference_landmarks.json  -> 478 landmarks (x,y en píxeles)
//     * metadata.json             -> criterio de éxito
// - Corre `faceLandmarker.detect(image)` sobre la imagen.
// - Convierte landmarks normalizados [0,1] a píxeles multiplicando por 224.
// - Compara punto a punto con la referencia Python:
//     mean_distance_per_landmark, max_distance_per_landmark.
// - Muestra PASS/FAIL según criterio (mean < 2px, max < 5px en imagen 224x224).
// - Render overlay del PNG con landmarks JS (verde) y Python (rojo) para
//   inspección visual.
//
// Razón del threshold "loose" (no exact match como en ONNX):
// MediaPipe Tasks for Web puede tener variaciones sub-pixel respecto a la
// versión Python por diferencias en cuantización del modelo o el orden de
// operadores en el grafo. Los landmarks son posiciones espaciales (no
// vectores abstractos como embeddings), así que toleramos algo de jitter.

import { useEffect, useRef, useState } from 'react';
import { FaceLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';

// -----------------------------------------
// Tipos del fixture
// -----------------------------------------
interface ReferenceLandmarks {
  shape: number[]; // [478, 2]
  dtype: string;
  coordinate_system: string;
  image_size_wh: number[]; // [W, H]
  values: number[][]; // 478 x [x_px, y_px]
}

interface FixtureMetadata {
  generated_at_utc: string;
  source_image_path: string;
  alignment: { image_size: number; margin_ratio: number; method: string };
  mediapipe: {
    model: string;
    static_image_mode: boolean;
    max_num_faces: number;
    refine_landmarks: boolean;
    min_detection_confidence: number;
    n_landmarks: number;
  };
  success_criteria_for_js_client: {
    mean_distance_per_landmark_max_px: number;
    max_distance_per_landmark_max_px: number;
    landmark_count_must_equal_reference: boolean;
    note: string;
  };
}

interface ComparisonResult {
  nLandmarksJs: number;
  nLandmarksRef: number;
  meanDistance: number;
  maxDistance: number;
  passed: boolean;
  inferMs: number;
}

// -----------------------------------------
// Componente
// -----------------------------------------
function SpikeMediapipe() {
  const [steps, setSteps] = useState<string[]>([]);
  const [metadata, setMetadata] = useState<FixtureMetadata | null>(null);
  const [result, setResult] = useState<ComparisonResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [done, setDone] = useState(false);

  // Para el overlay visual: necesitamos almacenar los landmarks JS y ref
  // para dibujarlos sobre el PNG.
  const [landmarksJs, setLandmarksJs] = useState<number[][] | null>(null);
  const [landmarksRef, setLandmarksRef] = useState<number[][] | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const log = (msg: string) => {
    console.log(msg);
    setSteps((prev) => [...prev, msg]);
  };

  useEffect(() => {
    let cancelled = false;
    // Capturado para que el cleanup del effect pueda liberarlo aunque el
    // unmount ocurra mientras el init aún está in-flight (Tarea #27).
    let landmarkerInstance: FaceLandmarker | null = null;

    (async () => {
      try {
        log('[1] Cargando metadata del fixture MediaPipe...');
        const metaResp = await fetch('/spike_fixtures_mediapipe/metadata.json');
        const meta: FixtureMetadata = await metaResp.json();
        if (cancelled) return;
        setMetadata(meta);
        log(`    Modelo: ${meta.mediapipe.model}`);
        log(`    refine_landmarks=${meta.mediapipe.refine_landmarks} -> n_landmarks=${meta.mediapipe.n_landmarks}`);
        log(`    Imagen: ${meta.alignment.image_size}x${meta.alignment.image_size} (margen ${meta.alignment.margin_ratio})`);
        log(`    Criterio: mean<${meta.success_criteria_for_js_client.mean_distance_per_landmark_max_px}px, max<${meta.success_criteria_for_js_client.max_distance_per_landmark_max_px}px`);

        log('[2] Cargando reference landmarks (Python)...');
        const refResp = await fetch('/spike_fixtures_mediapipe/reference_landmarks.json');
        const ref: ReferenceLandmarks = await refResp.json();
        const [imgW, imgH] = ref.image_size_wh;
        log(`    n=${ref.shape[0]} landmarks en imagen ${imgW}x${imgH}`);
        log(`    primeros 3: ${ref.values.slice(0,3).map(p => `(${p[0].toFixed(1)},${p[1].toFixed(1)})`).join(', ')}`);
        setLandmarksRef(ref.values);

        log('[3] Cargando imagen 224x224...');
        const img = new Image();
        img.crossOrigin = 'anonymous';
        await new Promise<void>((resolve, reject) => {
          img.onload = () => resolve();
          img.onerror = (e) => reject(new Error(`Failed to load image: ${e}`));
          img.src = '/spike_fixtures_mediapipe/aligned_face_224.png';
        });
        log(`    Imagen cargada: ${img.width}x${img.height}`);
        if (img.width !== imgW || img.height !== imgH) {
          throw new Error(`Tamaño inesperado: esperaba ${imgW}x${imgH}, recibí ${img.width}x${img.height}`);
        }

        log('[4] Inicializando MediaPipe FaceLandmarker (carga WASM + modelo .task ~3MB)...');
        const t0 = performance.now();
        // FilesetResolver carga los archivos WASM de tasks-vision desde CDN.
        const vision = await FilesetResolver.forVisionTasks(
          'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm',
        );
        const faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
          baseOptions: {
            // Modelo oficial de Google (float16 ~3-4 MB). Cuando no hay GPU
            // disponible, MediaPipe cae a CPU automáticamente.
            modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
            delegate: 'GPU',
          },
          runningMode: 'IMAGE',
          numFaces: 1,
          // Sin blendshapes ni transformation matrix; solo landmarks para el spike.
          outputFaceBlendshapes: false,
          outputFacialTransformationMatrixes: false,
          minFaceDetectionConfidence: 0.5,
          minFacePresenceConfidence: 0.5,
          minTrackingConfidence: 0.5,
        });
        // Asignar a la captura ANTES del check de cancelled — si la promesa
        // resolvió pero el desmontaje ya ocurrió, el cleanup la libera.
        landmarkerInstance = faceLandmarker;
        const initMs = performance.now() - t0;
        if (cancelled) return;
        log(`    FaceLandmarker creado en ${initMs.toFixed(0)} ms.`);

        log('[5] Detectando landmarks en la imagen...');
        const t1 = performance.now();
        const mpResult = faceLandmarker.detect(img);
        const inferMs = performance.now() - t1;

        if (mpResult.faceLandmarks.length === 0) {
          throw new Error('MediaPipe no detectó ninguna cara en la imagen del fixture.');
        }

        const rawLandmarks = mpResult.faceLandmarks[0];
        // MediaPipe devuelve normalized landmarks { x, y, z } en [0, 1].
        // Multiplicamos por (W, H) para tener píxeles, igual que en Python.
        const jsLandmarks: number[][] = rawLandmarks.map(lm => [lm.x * imgW, lm.y * imgH]);
        setLandmarksJs(jsLandmarks);
        log(`    n=${jsLandmarks.length} landmarks JS, infer=${inferMs.toFixed(0)} ms.`);
        log(`    primeros 3 JS: ${jsLandmarks.slice(0,3).map(p => `(${p[0].toFixed(1)},${p[1].toFixed(1)})`).join(', ')}`);

        log('[6] Comparando JS vs Python (distancia euclídea por landmark)...');
        const nMatching = Math.min(jsLandmarks.length, ref.values.length);
        if (jsLandmarks.length !== ref.values.length) {
          log(`    [!] Mismatch n_landmarks: JS=${jsLandmarks.length} vs Ref=${ref.values.length}. Comparando los primeros ${nMatching}.`);
        }
        let sumDist = 0, maxDist = 0;
        for (let i = 0; i < nMatching; i++) {
          const dx = jsLandmarks[i][0] - ref.values[i][0];
          const dy = jsLandmarks[i][1] - ref.values[i][1];
          const d = Math.sqrt(dx*dx + dy*dy);
          sumDist += d;
          if (d > maxDist) maxDist = d;
        }
        const meanDist = sumDist / nMatching;

        const passedCount = jsLandmarks.length === ref.values.length;
        const passedMean = meanDist < meta.success_criteria_for_js_client.mean_distance_per_landmark_max_px;
        const passedMax = maxDist < meta.success_criteria_for_js_client.max_distance_per_landmark_max_px;
        const passed = passedCount && passedMean && passedMax;

        log(`    mean_distance = ${meanDist.toFixed(3)} px  (criterio: <${meta.success_criteria_for_js_client.mean_distance_per_landmark_max_px})  ${passedMean ? '✓' : '✗'}`);
        log(`    max_distance  = ${maxDist.toFixed(3)} px  (criterio: <${meta.success_criteria_for_js_client.max_distance_per_landmark_max_px})  ${passedMax ? '✓' : '✗'}`);
        log(`    count match   = ${passedCount ? '✓' : '✗'}  (JS=${jsLandmarks.length}, Ref=${ref.values.length})`);

        if (cancelled) return;
        setResult({
          nLandmarksJs: jsLandmarks.length,
          nLandmarksRef: ref.values.length,
          meanDistance: meanDist,
          maxDistance: maxDist,
          passed,
          inferMs,
        });
        setDone(true);

        log('');
        if (passed) {
          log('[OK] Spike MediaPipe EXITOSO. Los landmarks JS coinciden con Python dentro de los thresholds.');
          log('     -> El pipeline cliente puede usar MediaPipe Tasks for Web como reemplazo directo del Face Mesh Python.');
        } else {
          log('[!!] FAIL. Investigar: ¿modelo distinto?, ¿coordenadas en otro sistema?, ¿pre-procesamiento del input?');
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
      landmarkerInstance?.close();
      landmarkerInstance = null;
    };
  }, []);

  // -----------------------------------------
  // Overlay: dibujar landmarks sobre el PNG
  // -----------------------------------------
  useEffect(() => {
    if (!landmarksJs || !landmarksRef || !canvasRef.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const img = new Image();
    img.onload = () => {
      canvas.width = img.width;
      canvas.height = img.height;
      ctx.drawImage(img, 0, 0);
      // Landmarks Python en rojo (referencia), JS en verde (test).
      ctx.fillStyle = 'rgba(255, 0, 0, 0.6)';
      landmarksRef.forEach(([x, y]) => {
        ctx.beginPath();
        ctx.arc(x, y, 1.5, 0, 2 * Math.PI);
        ctx.fill();
      });
      ctx.fillStyle = 'rgba(0, 255, 0, 0.6)';
      landmarksJs.forEach(([x, y]) => {
        ctx.beginPath();
        ctx.arc(x, y, 1.5, 0, 2 * Math.PI);
        ctx.fill();
      });
    };
    img.src = '/spike_fixtures_mediapipe/aligned_face_224.png';
  }, [landmarksJs, landmarksRef]);

  // -----------------------------------------
  // Render
  // -----------------------------------------
  return (
    <div style={{ fontFamily: 'monospace', padding: '20px', maxWidth: 1000, margin: '0 auto' }}>
      <h1 style={{ borderBottom: '2px solid #333', paddingBottom: 8 }}>
        Spike MediaPipe Face Mesh — paridad JS vs Python
      </h1>
      <p style={{ color: '#666', fontSize: 13 }}>
        Verifica que <code>@mediapipe/tasks-vision</code> en el browser produce los mismos 478 landmarks
        faciales que la versión Python de MediaPipe Face Mesh. Resultado decide si el pipeline de
        landmarks del cliente Track 2a usa MediaPipe Tasks for Web sin más fricción.
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

      {result && (
        <>
          <h2 style={{ marginTop: 20 }}>Resultado</h2>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <tbody>
              <tr><td style={{ border: '1px solid #ccc', padding: 8 }}>n landmarks (JS / Ref)</td>
                  <td style={{ border: '1px solid #ccc', padding: 8 }}>{result.nLandmarksJs} / {result.nLandmarksRef}</td></tr>
              <tr><td style={{ border: '1px solid #ccc', padding: 8 }}>mean distance (px)</td>
                  <td style={{ border: '1px solid #ccc', padding: 8 }}>{result.meanDistance.toFixed(3)}</td></tr>
              <tr><td style={{ border: '1px solid #ccc', padding: 8 }}>max distance (px)</td>
                  <td style={{ border: '1px solid #ccc', padding: 8 }}>{result.maxDistance.toFixed(3)}</td></tr>
              <tr><td style={{ border: '1px solid #ccc', padding: 8 }}>infer time (ms)</td>
                  <td style={{ border: '1px solid #ccc', padding: 8 }}>{result.inferMs.toFixed(0)}</td></tr>
              <tr style={{ background: result.passed ? '#efffef' : '#ffeeee', fontWeight: 700 }}>
                <td style={{ border: '1px solid #ccc', padding: 8 }}>Status</td>
                <td style={{ border: '1px solid #ccc', padding: 8 }}>{result.passed ? '✓ PASS' : '✗ FAIL'}</td>
              </tr>
            </tbody>
          </table>
        </>
      )}

      <h2 style={{ marginTop: 20 }}>Overlay visual</h2>
      <p style={{ fontSize: 12, color: '#666' }}>
        Landmarks: <span style={{ color: '#c00' }}>● rojo = Python (referencia)</span>,
        <span style={{ color: '#0a0', marginLeft: 12 }}>● verde = JS (MediaPipe Tasks for Web)</span>.
        Si los puntos coinciden el spike pasa.
      </p>
      <canvas
        ref={canvasRef}
        style={{ border: '1px solid #ccc', maxWidth: '100%', imageRendering: 'pixelated' }}
      />

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

export default SpikeMediapipe;
