// =========================================
// ID: PHYLOFACE_LIB_ALIGNMENT
// VERSION: v1.0
// =========================================
// Helpers de alineación canónica de rostros — port a TypeScript del algoritmo
// que `phyloface.core.pairs.align_face_from_record` usa en Python.
//
// Por qué este módulo existe:
// El pipeline browser-only del Track 2a necesita reproducir la alineación
// 112×112 que entra al modelo ONNX. La validación pixel-a-pixel vs Python se
// hizo en el spike #003 (`SpikeAlignment.tsx`) y pasó. Acá centralizamos esos
// helpers para que los consuman tanto el spike #003 como el spike #004
// (pipeline end-to-end) y, después, la UI MVP del comparador.
//
// Componentes:
//   - estimateNormSimilarity(src, dst): Umeyama 2D cerrado sin SVD. Devuelve
//     la matriz afín 2×3 que mapea los 5 keypoints a los 5 del template
//     ArcFace por similitud (rotación + escala uniforme + traslación, sin
//     reflexión). Equivalente a `skimage.transform.SimilarityTransform.estimate`
//     en 2D, que es lo que usa `face_align.estimate_norm` de InsightFace.
//   - adjustMatrixForMargin(M, imageSize, marginRatio): ajusta la matriz para
//     contraer la cara dentro del canvas (deja margen para que MediaPipe vea
//     contexto). Para margin_ratio=0 (caso del modelo ONNX, image_size=112),
//     M_adj = M.
//   - invertAffine(M): inversa de matriz afín 2×3 (necesario para warpAffine,
//     que muestrea src en coords M^(-1)·(xo,yo,1)).
//   - warpAffineBilinearReplicate(src, M, Wout, Hout): emula
//     cv2.warpAffine(src, M, ..., borderMode=BORDER_REPLICATE,
//     flags=INTER_LINEAR). Interpolación bilineal en float64 con clamp-to-edge.
//   - arcfaceDstTemplate(imageSize): template 5×2 escalado para `image_size`
//     (múltiplo de 112 o 128, mismo criterio que InsightFace).
//
// Convención de keypoints (orden, sistema de coords):
// - Orden: [left_eye, right_eye, nose, left_mouth, right_mouth].
// - "Left"/"right" desde la perspectiva del observador (image-space), igual
//   que InsightFace SCRFD. NO desde la perspectiva del sujeto.
// - Coords en píxeles del sistema del crop que se le pasa al warpAffine
//   (típicamente la imagen completa o el crop expandido alrededor de la cara).
//
// Tolerancia de paridad (validada en spike #003 sobre uint8 RGB):
//   mean_abs_pixel_diff ≈ 0.0x   (criterio: < 1.0)
//   max_abs_pixel_diff  <= 5     (ruido numérico de bilineal en bordes)
//   max |M_js − M_ref|  ~1e-6    (ruido de float64 vs estimate_norm Python)

// -----------------------------------------
// Template ArcFace canónico (sistema 112×112)
// -----------------------------------------
// Mismos valores que `insightface.utils.face_align.arcface_dst`.
const ARCFACE_DST_112: [number, number][] = [
  [38.2946, 51.6963],
  [73.5318, 51.5014],
  [56.0252, 71.7366],
  [41.5493, 92.3655],
  [70.7299, 92.2041],
];

// Devuelve el template arcface_dst escalado para image_size. Mantiene la
// misma lógica de estimate_norm: si imageSize es múltiplo de 112 se escala
// directo; si es múltiplo de 128, además se desplaza diff_x = 8*ratio.
export function arcfaceDstTemplate(imageSize: number): [number, number][] {
  let ratio: number;
  let diffX: number;
  if (imageSize % 112 === 0) {
    ratio = imageSize / 112;
    diffX = 0;
  } else if (imageSize % 128 === 0) {
    ratio = imageSize / 128;
    diffX = 8 * ratio;
  } else {
    throw new Error(
      `image_size debe ser múltiplo de 112 o 128 (recibí ${imageSize}). ` +
      `Es la restricción de face_align.estimate_norm de InsightFace.`
    );
  }
  return ARCFACE_DST_112.map(([x, y]) => [x * ratio + diffX, y * ratio]);
}

// -----------------------------------------
// Umeyama 2D — similitud sin reflexión
// -----------------------------------------
// Forma cerrada (sin SVD) para la similitud 2D que mapea `src` → `dst`:
//   c = Σ (sx·dx + sy·dy)           "alineación" (= cos·norm)
//   s = Σ (sx·dy − sy·dx)           "rotación signed" (= sin·norm)
//   var_src = Σ (sx² + sy²)         (varianza × n)
//   norm    = sqrt(c² + s²)
//   scale   = norm / var_src
//   cos(θ)  = c / norm
//   sin(θ)  = s / norm
//   sR = scale · [[cos, −sin], [sin, cos]]
//   t  = mean_dst − sR · mean_src
//
// Equivalente a skimage.SimilarityTransform.estimate(src, dst) en 2D (que es
// lo que usa face_align.estimate_norm). El algoritmo SVD-based original de
// Umeyama (1991) se reduce a esta forma cuando el espacio es 2D y se prohíbe
// reflexión.
export function estimateNormSimilarity(
  src: [number, number][] | number[][],
  dst: [number, number][] | number[][],
): number[][] {
  if (src.length !== dst.length || src.length === 0) {
    throw new Error(
      `Mismatched or empty src/dst (src=${src.length}, dst=${dst.length})`
    );
  }
  const n = src.length;

  let mxSrc = 0, mySrc = 0, mxDst = 0, myDst = 0;
  for (let i = 0; i < n; i++) {
    mxSrc += src[i][0]; mySrc += src[i][1];
    mxDst += dst[i][0]; myDst += dst[i][1];
  }
  mxSrc /= n; mySrc /= n; mxDst /= n; myDst /= n;

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

  if (varSrc === 0) throw new Error('Degenerate src points (zero variance)');
  const norm = Math.sqrt(c * c + s * s);
  if (norm === 0) throw new Error('Degenerate dst alignment (zero rotation magnitude)');

  const scale = norm / varSrc;
  const cosT = c / norm;
  const sinT = s / norm;

  const a = scale * cosT;
  const b = -scale * sinT;
  const d = scale * sinT;
  const e = scale * cosT;

  const tx = mxDst - (a * mxSrc + b * mySrc);
  const ty = myDst - (d * mxSrc + e * mySrc);

  return [
    [a, b, tx],
    [d, e, ty],
  ];
}

// -----------------------------------------
// Ajuste de margen post-estimate_norm
// -----------------------------------------
// Lo que `align_face_from_record` hace después de estimate_norm cuando
// margin_ratio > 0: contrae el rostro al (1-2*margin_ratio) del canvas y lo
// recentra. Para margin_ratio=0 devuelve una copia de M sin cambios.
export function adjustMatrixForMargin(
  M: number[][],
  imageSize: number,
  marginRatio: number,
): number[][] {
  if (marginRatio === 0) return M.map(row => [...row]);
  const scale = 1.0 - 2.0 * marginRatio;
  const shift = (imageSize * (1.0 - scale)) / 2.0;
  return [
    [M[0][0] * scale, M[0][1] * scale, M[0][2] * scale + shift],
    [M[1][0] * scale, M[1][1] * scale, M[1][2] * scale + shift],
  ];
}

// -----------------------------------------
// Inversa de matriz afín 2×3
// -----------------------------------------
// cv2.warpAffine recibe M como transformación src → dst pero internamente
// invierte M para muestrear src en coords M^(-1)·(xo,yo,1). Hacemos lo mismo
// acá para reproducir bit-a-bit (dentro del ruido bilineal) la semántica.
export function invertAffine(M: number[][]): number[][] {
  const [a, b, c] = M[0];
  const [d, e, f] = M[1];
  const det = a * e - b * d;
  if (Math.abs(det) < 1e-12) throw new Error(`Singular affine matrix (det=${det})`);
  const inv = 1.0 / det;
  const ia = e * inv;
  const ib = -b * inv;
  const id = -d * inv;
  const ie = a * inv;
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
// Emula cv2.warpAffine(src, M, (Wout, Hout), borderMode=BORDER_REPLICATE,
// flags=INTER_LINEAR). Procesa los 3 canales RGB; alpha out queda en 255.
//
// Diferencias residuales esperadas vs cv2:
// - cv2 usa fixed-point por velocidad; acá vamos en float64. Puede haber ±1
//   en uint8 por redondeo en píxeles muy oblicuos.
// - cv2 usa cvRound (banker's rounding en algunas plataformas); acá
//   Math.round (round half away from zero).
// Tolerancia validada en spike #003: mean<1.0, max<=5 sobre uint8.
export function warpAffineBilinearReplicate(
  src: ImageData,
  M: number[][],
  Wout: number,
  Hout: number,
): ImageData {
  const Wsrc = src.width;
  const Hsrc = src.height;
  const srcData = src.data;
  const Minv = invertAffine(M);
  const [ia, ib, itx] = Minv[0];
  const [id, ie, ity] = Minv[1];

  const out = new ImageData(Wout, Hout);
  const outData = out.data;

  for (let yo = 0; yo < Hout; yo++) {
    for (let xo = 0; xo < Wout; xo++) {
      const xs = ia * xo + ib * yo + itx;
      const ys = id * xo + ie * yo + ity;

      const x0 = Math.floor(xs);
      const y0 = Math.floor(ys);
      const x1 = x0 + 1;
      const y1 = y0 + 1;
      const fx = xs - x0;
      const fy = ys - y0;

      // BORDER_REPLICATE = clamp-to-edge.
      const cx0 = x0 < 0 ? 0 : x0 >= Wsrc ? Wsrc - 1 : x0;
      const cx1 = x1 < 0 ? 0 : x1 >= Wsrc ? Wsrc - 1 : x1;
      const cy0 = y0 < 0 ? 0 : y0 >= Hsrc ? Hsrc - 1 : y0;
      const cy1 = y1 < 0 ? 0 : y1 >= Hsrc ? Hsrc - 1 : y1;

      const o00 = (cy0 * Wsrc + cx0) * 4;
      const o01 = (cy0 * Wsrc + cx1) * 4;
      const o10 = (cy1 * Wsrc + cx0) * 4;
      const o11 = (cy1 * Wsrc + cx1) * 4;
      const oOut = (yo * Wout + xo) * 4;

      const w00 = (1 - fx) * (1 - fy);
      const w01 = fx * (1 - fy);
      const w10 = (1 - fx) * fy;
      const w11 = fx * fy;

      for (let ch = 0; ch < 3; ch++) {
        const v = srcData[o00 + ch] * w00
                + srcData[o01 + ch] * w01
                + srcData[o10 + ch] * w10
                + srcData[o11 + ch] * w11;
        const r = Math.round(v);
        outData[oOut + ch] = r < 0 ? 0 : r > 255 ? 255 : r;
      }
      outData[oOut + 3] = 255;
    }
  }

  return out;
}
