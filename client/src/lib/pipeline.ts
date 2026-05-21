// =========================================
// ID: PHYLOFACE_LIB_PIPELINE
// VERSION: v1.1
// =========================================
// Cambio v1.0 → v1.1 (Tarea #25c — UI MVP comparador):
// - `PipelineOutput` ahora incluye `aligned: ImageData` (la cara warped a 112×112).
//   Lo necesita el comparador para mostrar preview visual de qué se está
//   comparando. El spike #004 lo ignora silenciosamente.
//
// =========================================
// Pipeline browser-only de embedding facial: detect (Face Mesh) → 5 kps en
// orden InsightFace → align canónico 112×112 (lib/alignment.ts) → preprocesar
// → ONNX (w600k_r50) → embedding 512-d.
//
// Por qué este módulo existe:
// Antes vivía inline en `SpikeDetection.tsx` (spike #004), enredado con la
// lógica de comparación contra fixtures Python. El comparador real (Tarea #25
// subtarea c) necesita esta pieza limpia: dada una imagen, devolver embedding
// + kps + timings, SIN nada de reference embeddings ni métricas de paridad.
// Lo que queda en el spike es solo la capa que envuelve esto y compara contra
// el embedding de referencia del fixture multi-caso.
//
// Componentes:
//   - MESH_INDICES_INSIGHTFACE_ORDER: mapping fijo Face Mesh → 5 kps InsightFace
//     (validado empíricamente en spike #004, 2026-05-21).
//   - cosineSimilarity(a, b): similitud coseno entre dos Float32Array.
//   - loadImage(url): fetch + decode + ImageData (HTMLImage + RGBA píxeles).
//   - imageDataToTensorRGB(imgData, mean, std): ImageData 112×112 → tensor
//     NCHW float32 (1, 3, 112, 112) normalizado.
//   - initFaceLandmarker(): MediaPipe FaceLandmarker en IMAGE mode, GPU delegate.
//   - initOnnxSession(modelUrl): ONNX session con providers webgpu+wasm.
//   - computeEmbedding(img, imageData, landmarker, session): pipeline e2e
//     puro, sin comparación, sin overlay. Devuelve embedding + kps + timings.
//
// Convenciones:
// - Las funciones de init no cachean: el caller decide si quiere instanciar
//   landmarker/session una vez y reusarlos (recomendado: el costo de init es
//   alto, ~segundos).
// - Coords de kps en image-space (px de la imagen original, NO 0..1).
// - "Left"/"right" desde el observador, como SCRFD (no como el sujeto).

import * as ort from 'onnxruntime-web';
import { FaceLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';
import {
  estimateNormSimilarity,
  arcfaceDstTemplate,
  warpAffineBilinearReplicate,
} from './alignment';

// -----------------------------------------
// Mapping Face Mesh → 5 kps InsightFace.
// Índices fijos del mesh con refine_landmarks=true.
// Orden InsightFace: [left_eye, right_eye, nose, left_mouth, right_mouth].
// 468=iris izq (refine), 473=iris der (refine), 4=punta nariz,
// 61=comisura izq boca, 291=comisura der boca.
// Validado en spike #004 contra refs Python InsightFace SCRFD.
// -----------------------------------------
export const MESH_INDICES_INSIGHTFACE_ORDER = [468, 473, 4, 61, 291] as const;

// Modelo ArcFace canónico: input 112×112, normalización (x-127.5)/127.5.
const MODEL_INPUT_SIZE = 112;
const MODEL_PIXEL_MEAN = 127.5;
const MODEL_PIXEL_STD = 127.5;

// -----------------------------------------
// Math
// -----------------------------------------
export function cosineSimilarity(a: Float32Array, b: Float32Array): number {
  if (a.length !== b.length) throw new Error('Length mismatch');
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB) + 1e-8);
}

// -----------------------------------------
// IO: cargar imagen desde URL como HTMLImage + ImageData RGBA.
// HTMLImage lo necesita MediaPipe (detecta sobre el elemento img);
// ImageData lo necesita warpAffine (lee píxeles RGBA directos).
// -----------------------------------------
export async function loadImage(
  url: string,
): Promise<{ img: HTMLImageElement; imageData: ImageData }> {
  const img = new Image();
  img.crossOrigin = 'anonymous';
  await new Promise<void>((resolve, reject) => {
    img.onload = () => resolve();
    img.onerror = (e) => reject(new Error(`Failed to load ${url}: ${e}`));
    img.src = url;
  });
  const canvas = document.createElement('canvas');
  canvas.width = img.width;
  canvas.height = img.height;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  if (!ctx) throw new Error('Cannot get 2D context');
  ctx.drawImage(img, 0, 0);
  const imageData = ctx.getImageData(0, 0, img.width, img.height);
  return { img, imageData };
}

// -----------------------------------------
// ImageData 112×112 RGBA → tensor NCHW float32 normalizado.
// Recorre RGBA, descarta alpha, ordena canal-major (R, G, B planos).
// -----------------------------------------
export function imageDataToTensorRGB(
  imgData: ImageData,
  mean: number = MODEL_PIXEL_MEAN,
  std: number = MODEL_PIXEL_STD,
): Float32Array {
  const W = imgData.width;
  const H = imgData.height;
  if (W !== MODEL_INPUT_SIZE || H !== MODEL_INPUT_SIZE) {
    throw new Error(`Expected ${MODEL_INPUT_SIZE}x${MODEL_INPUT_SIZE} ImageData, got ${W}x${H}`);
  }
  const rgba = imgData.data;
  const out = new Float32Array(1 * 3 * MODEL_INPUT_SIZE * MODEL_INPUT_SIZE);
  const planeSize = MODEL_INPUT_SIZE * MODEL_INPUT_SIZE;
  for (let y = 0; y < MODEL_INPUT_SIZE; y++) {
    for (let x = 0; x < MODEL_INPUT_SIZE; x++) {
      const pxIdx = (y * MODEL_INPUT_SIZE + x) * 4;
      const outIdx = y * MODEL_INPUT_SIZE + x;
      out[0 * planeSize + outIdx] = (rgba[pxIdx + 0] - mean) / std;
      out[1 * planeSize + outIdx] = (rgba[pxIdx + 1] - mean) / std;
      out[2 * planeSize + outIdx] = (rgba[pxIdx + 2] - mean) / std;
    }
  }
  return out;
}

// -----------------------------------------
// Init: MediaPipe FaceLandmarker (1 cara, IMAGE mode, GPU delegate).
// Costoso (descarga modelo desde CDN + WASM). El caller debería instanciar
// una sola vez y reusar entre imágenes.
// -----------------------------------------
export async function initFaceLandmarker(): Promise<FaceLandmarker> {
  const vision = await FilesetResolver.forVisionTasks(
    'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm',
  );
  return FaceLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
      delegate: 'GPU',
    },
    runningMode: 'IMAGE',
    numFaces: 1,
    outputFaceBlendshapes: false,
    outputFacialTransformationMatrixes: false,
    minFaceDetectionConfidence: 0.5,
    minFacePresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });
}

// -----------------------------------------
// Init: ONNX session. modelUrl debe servir w600k_r50.onnx (o compatible
// con ArcFace 112×112 input, 512-d output).
// -----------------------------------------
export async function initOnnxSession(
  modelUrl: string = '/models/w600k_r50.onnx',
): Promise<ort.InferenceSession> {
  return ort.InferenceSession.create(modelUrl, {
    executionProviders: ['webgpu', 'wasm'],
    graphOptimizationLevel: 'all',
  });
}

export interface PipelineTimings {
  detectMs: number;
  alignMs: number;
  preprocessMs: number;
  inferMs: number;
}

export interface PipelineOutput {
  embedding: Float32Array;       // 512-d, sin L2-normalize (cosineSimilarity normaliza)
  kps: number[][];                // 5×2 en image-space (px de la imagen original)
  aligned: ImageData;             // cara warped a 112×112 RGBA (útil para preview)
  timings: PipelineTimings;
}

// -----------------------------------------
// Pipeline e2e puro: imagen ya cargada → embedding + kps + timings.
// No conoce fixtures, refs, ni overlays. Esa lógica vive en el caller.
//
// Throws si Face Mesh no detecta cara o si falta algún landmark del mapping.
// -----------------------------------------
export async function computeEmbedding(
  img: HTMLImageElement,
  imageData: ImageData,
  faceLandmarker: FaceLandmarker,
  session: ort.InferenceSession,
): Promise<PipelineOutput> {
  const W = img.width;
  const H = img.height;

  // Detect.
  const tDet0 = performance.now();
  const mpResult = faceLandmarker.detect(img);
  const detectMs = performance.now() - tDet0;
  if (mpResult.faceLandmarks.length === 0) {
    throw new Error('Face Mesh no detectó cara');
  }
  const meshLandmarks = mpResult.faceLandmarks[0];

  // 5 kps image-space en orden InsightFace.
  const kps: number[][] = MESH_INDICES_INSIGHTFACE_ORDER.map((idx) => {
    const lm = meshLandmarks[idx];
    if (!lm) throw new Error(`Landmark ${idx} missing en mesh`);
    return [lm.x * W, lm.y * H];
  });

  // Align: similitud 2D que mapea los 5 kps al template ArcFace 112×112.
  const tAlign0 = performance.now();
  const dst = arcfaceDstTemplate(MODEL_INPUT_SIZE);
  const M = estimateNormSimilarity(kps, dst as number[][]);
  const aligned = warpAffineBilinearReplicate(imageData, M, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE);
  const alignMs = performance.now() - tAlign0;

  // Preprocess: RGBA → NCHW float32 normalizado.
  const tPre0 = performance.now();
  const tensorData = imageDataToTensorRGB(aligned);
  const preprocessMs = performance.now() - tPre0;

  // Infer.
  const tInf0 = performance.now();
  const inputName = session.inputNames[0];
  const outputName = session.outputNames[0];
  const tensor = new ort.Tensor('float32', tensorData, [1, 3, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE]);
  const outputs = await session.run({ [inputName]: tensor });
  const inferMs = performance.now() - tInf0;
  const embedding = new Float32Array(outputs[outputName].data as Float32Array);

  return {
    embedding,
    kps,
    aligned,
    timings: { detectMs, alignMs, preprocessMs, inferMs },
  };
}
