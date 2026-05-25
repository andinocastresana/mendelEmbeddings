// =========================================
// ID: PHYLOFACE_LIB_REGIONS
// VERSION: v1.0
// =========================================
// Espejo en JS del contrato canónico de regiones del motor Python
// (`src/phyloface/regions/canonical.py`, `CANONICAL_REGIONS_VERSION =
// "regions-v2.0"`) + los índices de landmarks de `regions/geometry.py`.
//
// Por qué existe: el cliente browser-first necesita la MISMA lista de regiones y
// los MISMOS índices de Face Mesh que el motor, para que los scores regionales
// (geométricos, occlusion, etc.) hablen el mismo idioma que la calibración y la
// futura app primaria. Si cambia `regions-v2.0` en Python, hay que reflejarlo acá
// (y bumpear REGIONS_VERSION). No es un import — es una copia versionada a mano,
// con la fuente anotada para mantenerlas sincronizadas.
//
// Backend de landmarks: MediaPipe Face Mesh refine_landmarks (478 puntos), igual
// que el pipeline (`lib/pipeline.ts`).
//
// Notas de fidelidad respecto a geometry.py:
// - Ojos: uso el contorno poligonal (`*_EYE_POLYGON_IDX`, 6 pts) como conjunto
//   representativo — suficiente para bbox/centroide/geometría. (El set completo
//   `FACEMESH_*_EYE` se deriva de connection-sets en Python; el polígono basta acá.)
// - Boca: contorno externo de labios (`MOUTH_POLYGON_IDX`, 11 pts).
// - Frente: NO tiene índices propios (Face Mesh no da contorno cerrado). Se deriva
//   de cejas + ojos en el scorer (ver get_forehead_bbox en geometry.py). Acá queda
//   con landmarkIdx=null y derived=true.

export const REGIONS_VERSION = 'regions-v2.0';
export const LANDMARKS_BACKEND = 'mediapipe-face-mesh-478';

export type RegionSide = 'left' | 'right' | 'midline';
export type RegionSource = 'mediapipe-official' | 'manual-approx' | 'derived-approx';

export type RegionName =
  | 'left_eyebrow' | 'right_eyebrow'
  | 'left_eye' | 'right_eye'
  | 'left_cheekbone' | 'right_cheekbone'
  | 'left_cheek' | 'right_cheek'
  | 'nose' | 'mouth' | 'chin' | 'forehead';

export interface RegionSpec {
  name: RegionName;
  labelEs: string;
  group: string;
  side: RegionSide;
  pairedWith: RegionName | null;
  source: RegionSource;
  /** Índices Face Mesh que definen la región (bbox/centroide/geometría).
   *  `null` solo para `forehead` (derivada de cejas+ojos en el scorer). */
  landmarkIdx: number[] | null;
  /** true si la región no tiene landmarks propios y se deriva de otras. */
  derived: boolean;
}

// Índices transcritos de `regions/geometry.py` (regions-v2.0).
export const CANONICAL_REGIONS: readonly RegionSpec[] = [
  {
    name: 'left_eyebrow', labelEs: 'ceja izquierda', group: 'eyebrow', side: 'left',
    pairedWith: 'right_eyebrow', source: 'manual-approx', derived: false,
    landmarkIdx: [70, 63, 105, 66, 107, 55, 65, 52, 53, 46],
  },
  {
    name: 'right_eyebrow', labelEs: 'ceja derecha', group: 'eyebrow', side: 'right',
    pairedWith: 'left_eyebrow', source: 'manual-approx', derived: false,
    landmarkIdx: [336, 296, 334, 293, 300, 285, 295, 282, 283, 276],
  },
  {
    name: 'left_eye', labelEs: 'ojo izquierdo', group: 'eye', side: 'left',
    pairedWith: 'right_eye', source: 'mediapipe-official', derived: false,
    landmarkIdx: [33, 160, 158, 133, 153, 144],
  },
  {
    name: 'right_eye', labelEs: 'ojo derecho', group: 'eye', side: 'right',
    pairedWith: 'left_eye', source: 'mediapipe-official', derived: false,
    landmarkIdx: [362, 385, 387, 263, 373, 380],
  },
  {
    name: 'left_cheekbone', labelEs: 'pómulo izquierdo', group: 'cheekbone', side: 'left',
    pairedWith: 'right_cheekbone', source: 'manual-approx', derived: false,
    landmarkIdx: [50, 101, 100, 126, 142, 203, 206],
  },
  {
    name: 'right_cheekbone', labelEs: 'pómulo derecho', group: 'cheekbone', side: 'right',
    pairedWith: 'left_cheekbone', source: 'manual-approx', derived: false,
    landmarkIdx: [280, 330, 329, 355, 371, 423, 426],
  },
  {
    name: 'left_cheek', labelEs: 'mejilla izquierda', group: 'cheek', side: 'left',
    pairedWith: 'right_cheek', source: 'manual-approx', derived: false,
    landmarkIdx: [116, 117, 118, 119, 120, 100, 126, 142, 203, 205, 50],
  },
  {
    name: 'right_cheek', labelEs: 'mejilla derecha', group: 'cheek', side: 'right',
    pairedWith: 'left_cheek', source: 'manual-approx', derived: false,
    landmarkIdx: [345, 346, 347, 348, 349, 329, 355, 371, 423, 425, 280],
  },
  {
    name: 'nose', labelEs: 'nariz', group: 'nose', side: 'midline',
    pairedWith: null, source: 'manual-approx', derived: false,
    landmarkIdx: [1, 2, 4, 5, 6, 19, 20, 45, 48, 49, 64, 94, 97, 98, 115, 122, 129,
      168, 195, 197, 218, 275, 278, 279, 294, 327, 331, 344],
  },
  {
    name: 'mouth', labelEs: 'boca', group: 'mouth', side: 'midline',
    pairedWith: null, source: 'mediapipe-official', derived: false,
    landmarkIdx: [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291],
  },
  {
    name: 'chin', labelEs: 'mentón', group: 'chin', side: 'midline',
    pairedWith: null, source: 'manual-approx', derived: false,
    landmarkIdx: [152, 148, 176, 149, 150, 136, 172, 58, 132, 361, 397, 378, 379, 365, 288, 435],
  },
  {
    name: 'forehead', labelEs: 'frente', group: 'forehead', side: 'midline',
    pairedWith: null, source: 'derived-approx', derived: true,
    landmarkIdx: null, // derivada de cejas + ojos (ver geometry.get_forehead_bbox)
  },
];

export const REGION_NAMES: readonly RegionName[] = CANONICAL_REGIONS.map((r) => r.name);

export const REGION_BY_NAME: Record<RegionName, RegionSpec> = Object.fromEntries(
  CANONICAL_REGIONS.map((r) => [r.name, r]),
) as Record<RegionName, RegionSpec>;

/** Índices auxiliares para derivar la frente (cejas + ojos), igual que Python. */
export const FOREHEAD_REF_BROW_IDX = [
  ...(REGION_BY_NAME.left_eyebrow.landmarkIdx ?? []),
  ...(REGION_BY_NAME.right_eyebrow.landmarkIdx ?? []),
];
export const FOREHEAD_REF_EYE_IDX = [
  ...(REGION_BY_NAME.left_eye.landmarkIdx ?? []),
  ...(REGION_BY_NAME.right_eye.landmarkIdx ?? []),
];
