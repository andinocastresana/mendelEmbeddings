# DEVLOG — mendelEmbeddings

Historial técnico de cambios del proyecto.
Los IDs de tarea referencian `TAREAS_PENDIENTES.md`.

Notación de tareas: `T1` = relacionada · `T1↑` = tarea creada · `T1✓` = tarea cerrada

Cada entrada incluye: hash de commit, título de una línea, IDs de tarea relacionados y detalle del cambio (mismo nivel de descripción que las entradas existentes). Las entradas se agrupan por fecha en orden cronológico inverso (lo más nuevo arriba).

---

## 2026-05-20

### `[hash pendiente]` · Track 2a — Spike MediaPipe Face Mesh JS vs Python (PASS) + setup del cliente con tabs `T2`

- **Spike menor PHYLOFACE_SPIKE_002**: verifica que **MediaPipe Tasks for Web** (`@mediapipe/tasks-vision`) produce los mismos 478 landmarks faciales que la versión Python de MediaPipe Face Mesh.
  - `scripts/verify_mediapipe_web_parity.py`: genera fixture en `client/public/spike_fixtures_mediapipe/` (imagen 224x224 + 478 landmarks de referencia + metadata).
  - Cliente: nuevo componente `SpikeMediapipe.tsx` que carga MediaPipe Face Landmarker (modelo `.task` desde CDN de Google ~3-4 MB) y compara con la referencia Python. Overlay visual con landmarks JS (verde) y Python (rojo).
  - Criterios: mean < 2px, max < 5px sobre imagen 224x224 (loose porque MediaPipe Web puede tener variaciones sub-pixel por cuantización).
  - **Resultado**: PASS. Las dos piezas técnicas del Track 2a están confirmadas:
    - **Embedding** (ONNX Runtime Web + `w600k_r50.onnx`) ✅ desde `de8b1a2`.
    - **Landmarks** (MediaPipe Tasks for Web + Face Landmarker) ✅ ahora.
- **Refactor de App.tsx**: `App.tsx` ahora es un router simple con tabs entre los dos spikes (`SpikeOnnx` + `SpikeMediapipe`). Cuando arranque el comparador real (próxima sesión), `App.tsx` se reemplaza por el comparador con tabs como vista de "spikes legacy".
- **Decisión sobre fixtures**: ambos directorios `client/public/spike_fixtures*/` quedan gitignored (regenerables con sus scripts Python respectivos). Excepción explícita: PDF de snapshots de arquitectura via `!_meta/**/*.pdf`.
- Tarea #2 creada en TaskList: "Track 2a — comparador anónimo 100% browser (MVP)". Sigue in_progress; los dos spikes son hitos del avance pero el comparador real aún no está construido (próxima sesión: alineación canónica JS replicando `align_face_from_record`, después UI de subir 2 fotos + comparar).

### `de8b1a2` · Sesión de diseño stack web + snapshot v0.1 + spike Track 2a (con bugfix latente del motor)

- **Diseño arquitectónico del stack web público**: tras una sesión de discusión con el usuario, plan revisado a **2 tracks paralelos con arquitectura híbrida** (cliente-pesado / server-liviano):
  - Track 1: vitrina equipos estática (JSON pre-calculado offline).
  - Track 2a: comparador anónimo 100% en browser (inference con ONNX Runtime Web + MediaPipe Tasks for Web; imagen nunca sale del browser).
  - Track 2b: refinamiento server-side opt-in para registrados.
- **Snapshot inmutable versionado**: `_meta/arquitectura_web/v0.1_2026-05-20_arquitectura_web.md` (+ PDF generado con pandoc+Chrome headless). Patrón nuevo: doc vivo (`ARQUITECTURA.md` §5) apunta a snapshot inmutable; versiones futuras se crean al lado sin editar las viejas. Excepción al gitignore `*.pdf`: `!_meta/**/*.pdf` para incluir snapshots PDF en el repo.
- **ARQUITECTURA.md §5 reescrita**: resumen del diseño vigente + estado de W1–W8 (3 cerradas, 1 resuelta vía arquitectura, 1 parcial, 3 diferidas a Track 2b), link al snapshot detallado.
- **Spike Track 2a — paridad ONNX Runtime Web vs Python**: `scripts/verify_onnx_web_parity.py` (ID `PHYLOFACE_SPIKE_001 v1.0`) genera fixture (imagen alineada 112×112, tensor pre-procesado, embedding de referencia, metadata con criterio de éxito). Cliente Vite + React + TypeScript en `client/` carga el modelo ONNX en el browser (vía `onnxruntime-web`, backend WebGPU/WASM), corre inference y compara con el reference embedding.
  - **Path A** (tensor JSON → modelo): `cosine = 1.000000`, `max|diff| = 2.86e-6`, infer ~526ms. **PASS perfecto**. El modelo ONNX corre en el browser produciendo embeddings bit-idénticos a Python.
  - **Path B** (PNG → preprocess JS → modelo): tras corregir un bug propio de mi código JS (canales invertidos por mala simetría con el bugfix Python), también `cosine = 1.0`. **PASS**.
  - **Resultado**: plan híbrido v0.1 **confirmado viable**. Track 2a puede avanzar.
- **BUGFIX latente del motor Python descubierto durante el spike**: `extract_embedding_from_aligned` en `core/embedder.py` pasaba `aligned_rgb` (RGB) a `get_feat`, que internamente hace `cv2.dnn.blobFromImages(..., swapRB=True)` asumiendo BGR. Resultado: modelo recibía BGR cuando esperaba RGB → embedding con canales invertidos vs entrenamiento. Bug existía desde el archivo experimental original (heredado por la migración del 19-20). Silencioso porque ambas caras del par sufrían la misma inversión y los rankings se preservaban. **Fix de 1 línea**: `cv2.cvtColor(aligned_rgb, COLOR_RGB2BGR)` antes de `get_feat`. Documentado en cabecera del módulo `embedder.py` con bloque `=== BUGFIX 2026-05-20 (Spike Track 2a) ===`. Implicancia: los scores numéricos del notebook de la sesión 7b635a5 quedan obsoletos (plots no cambian); el usuario optó por no re-correr el notebook ahora (lo hará al agregar features nuevas).
- **Infraestructura del cliente** (`client/`):
  - Vite 8 + React 19 + TypeScript template estándar.
  - `onnxruntime-web` instalado.
  - `client/public/models/` (174MB modelo onnx, gitignored).
  - `client/public/spike_fixtures/` (~820KB, gitignored — regenerables con el script Python).
  - `npm run dev` levanta dev server en `localhost:5173`.

### `e1a6511` · T1 pasos 7–9: cierre de la Tarea #1 (viz/, notebook reescrito y archivo experimental archivado) `T1✓`

- **Cierre de la Tarea #1** (migración `src/phyloface_experimental_functions.py` → `src/phyloface/`). 40/40 funciones migradas + notebook funcional + archivo original archivado en `_toReview/`.
- **Paso 7**: subpaquete `src/phyloface/viz/` con 3 archivos nuevos (`detection.py`, `landmarks.py`, `regions.py`) — ID `PHYLOFACE_VIZ_001 v1.0`. 7 funciones de plotting migradas; `__init__.py` actualizado para re-exportar las nuevas + las 2 pre-existentes de `heatmap.py`. Primer smoke test como archivo versionado (`tests/smoke/test_paso_7_viz.py`, ID `PHYLOFACE_SMOKE_007 v1.0`) con backend `Agg`; convención documentada en `tests/smoke/README.md`.
- **Paso 8**: `notebooks/phyloface_expeimental_test1.py` reescrito (ID `PHYLOFACE_NOTEBOOK_002 v2.0`). Imports actualizados al paquete migrado; backend `Agg`; carpeta de salida `data/output/notebook_runs_<TIMESTAMP>/` con helper `save_current(name)`. Corrida end-to-end con `fraternos_jovenes.jpg` + `fraternosChacabuco8.jpg` produjo 15 plots PNG (revisados visualmente por el usuario, OK). Resultados rect vs masked numéricamente consistentes con el experimental original; NaN esperados en `left_eye`/`right_eye` masked (heredado, documentado).
- **Paso 9**: `src/phyloface_experimental_functions.py` movido con `git mv` a `_toReview/phyloface_experimental_functions_20260520_110102.py` (dentro del repo, Git lo trackea como rename → historia preservada). Sufijo de timestamp respeta la regla del proyecto.
- **Infraestructura de fluidez** agregada en paralelo:
  - `.claude/settings.json` (commit-able) con allowlist `Bash(rclone listremotes)`, `Bash(rclone lsjson *)`, `Bash(python3 tests/smoke/*)`.
  - `.claude/settings.local.json` con `defaultMode: "acceptEdits"` (gitignoreado — preferencia personal).
  - `.gitignore` extendido con `.claude/settings.local.json` (convención estándar de Claude Code).
- **Updates de docs**:
  - `TAREAS_PENDIENTES.md`: Tarea #1 movida a sección "Completadas" con notas de cierre.
  - `ARQUITECTURA.md` §1.3, 1.4, 1.5: estados actualizados con referencias a los nuevos módulos del paquete y clarificación de qué falta para Nivel B "real" (Tarea #5).
  - `_meta/MIGRACION_TAREA1.md`: pasos 7+8+9 marcados ✅, bitácora con detalle del cierre, sección de "Verificación al cierre" completa.

### `9586cf5` · T1 pasos 1–6: migración del notebook experimental al paquete `phyloface` (motor casi completo) `T1`

- Avance de la **Tarea #1** (migración de `src/phyloface_experimental_functions.py` → `src/phyloface/`). 34/40 funciones migradas en 6 pasos verificados con smoke tests independientes. Pendientes: paso 7 (`viz/`), 8 (reescribir notebook), 9 (archivar el archivo experimental original).
- **Paso 1**: tracker `_meta/MIGRACION_TAREA1.md` con mapa función→destino y bitácora granular del avance.
- **Paso 2**: subpaquete `src/phyloface/landmarks/` (ID `PHYLOFACE_LANDMARKS_001 v1.0`) — backend MediaPipe Face Mesh: `init_face_mesh`, `get_face_mesh_landmarks`, `add_dense_landmarks_to_pair`.
- **Paso 3**: subpaquete `src/phyloface/regions/` parte rectangular — `geometry.py` (5 helpers + 11 constantes de índices) y `extract_rect.py` (`extract_regions_v2`, `add_regions_v2_to_pair`). ID `PHYLOFACE_REGIONS_001 v1.0`.
- **Paso 4**: `regions/extract_masked.py` (ID `PHYLOFACE_REGIONS_002 v1.0`) — máscaras poligonales/convex hull, recortes enmascarados, 4 funciones. `geometry.py` extendido con 6 constantes `*_POLYGON_IDX`.
- **Paso 5**: `src/phyloface/comparator_regional.py` (ID `PHYLOFACE_COMPARATOR_REGIONAL_001 v1.0`) — 6 funciones (rect + masked + helpers + print). Se preservó la limitación heredada de `print_regional_summary` (sólo entiende clave `gray_cosine`) para no introducir cambios funcionales encubiertos.
- **Paso 6** (4 sub-pasos):
  - **6a** `core/io.py` (`load_image`) + `core/metrics.py` reescrito. ID `PHYLOFACE_CORE_BASE_001 v1.0`. **Cambio funcional decidido con el usuario**: `cosine_similarity` y `euclidean_distance` reemplazadas por las versiones experimentales (normalizadas vía `l2_normalize`). La euclídea pasa de "valores grandes proporcionales a la magnitud del embedding" a `[0, 2]`. Impacto bajo: no hay umbrales calibrados todavía; Tarea #6 (KinFaceW) los generará data-driven.
  - **6b** `core/embedder.py` (ID `PHYLOFACE_EMBEDDER_001 v1.0`) — `get_recognition_model`, `extract_embedding_from_aligned`.
  - **6c** `core/pairs.py` (ID `PHYLOFACE_PAIRS_001 v1.0`) — 7 funciones: detección + bbox expandida + selección por `face_id` + alineación canónica + `build_selected_pair`. Restricción heredada documentada: `image_size` debe ser múltiplo de 112 o 128 (assertion de `face_align.estimate_norm`).
  - **6d** `core/comparator_global.py` (ID `PHYLOFACE_COMPARATOR_GLOBAL_001 v1.0`) — integrador: `compute_global_metrics` (anida 5 métricas QC + 6 scores globales en el selected_pair) y `print_global_summary`.
- **Cambio de rumbo del proyecto** anotado en paralelo a la migración: `ARQUITECTURA.md` §5 documenta la decisión de servir el proyecto como **servicio web público** (FastAPI server + SPA frontend separada). 5 decisiones cerradas + 8 decisiones abiertas (W1–W8) pendientes de discutir cuando termine la Tarea #1. El motor `phyloface` queda 100% agnóstico al transporte (no importa FastAPI desde el motor).
- Convención general adoptada: todo script nuevo lleva cabecera `# === ID / VERSION ===` + bloque `# FILE:` + comentarios densos en español (guardada en memoria de Claude Code).

---

## 2026-05-19

### `9586cf5` · Alta de arquitectura, integración de charla externa y bibliografía kinship `T1↑ T2↑ T3↑ T4↑ T5↑ T6↑ T7↑ T8↑ T9↑ T10↑ T11↑ T12↑ T13↑ T14↑ T15↑ T16↑ T17↑ T18↑ T19↑ T20↑ T21↑ T22↑ T23↑ T24↑`

- `ARQUITECTURA.md` creado (raíz del proyecto): documento vivo con el esqueleto del motor de comparación, las dos apps (primaria = niño↔progenitores; secundaria = grupos/equipos), capa de visualización y bloque de validación. Cada componente marcado con ✅/🔄/⏳ reflejando estado actual. Incluye tabla de decisiones cerradas (derivadas de la charla externa con ChatGPT en `data/input/docs/charlaChatGPT.md`) y decisiones abiertas.
- `TAREAS_PENDIENTES.md` reescrito desde cero. Se descartan las 8 tareas iniciales (puramente estructurales) y se reemplazan por 24 tareas colgadas de cada bloque numerado de `ARQUITECTURA.md`. Códigos de área = referencia a bloques (M1.x = motor, A2.x = app, V = viz, B = benchmark).
- `data/input/datasets/DATASETS.md` creado: catálogo detallado de los 4 datasets descargados (KinFaceW-I, KinFaceW-II, TSKinFace_Data, TSKinFace_SIFT) con tabla de pares por relación, estructura interna, protocolo de evaluación (5-fold CV), convenciones de nombres, uso previsto en el proyecto, datasets relacionados aún no descargados (FIW, KFVW, Family101) y bibliografía principal. La inspección de los zip se hizo sin descomprimir a disco (`unzip -l` y `unzip -p`).
- Inputs incorporados al diseño:
  - `data/input/docs/Ideas&Progress.pdf` (presentación con estado previo y objetivos planteados).
  - `data/input/docs/charlaChatGPT.md` (charla externa con propuesta de pipeline: ArcFace global → MediaPipe Face Mesh → regiones en dos niveles geométrico + visual → occlusion sensitivity → pesos diferenciados).
  - `data/input/datasets/sources.md` (URLs de los datasets).
- Hallazgo: el notebook `notebooks/phyloface_expeimental_test1.py` ya implementa el camino propuesto por la charla (MediaPipe + regiones v2 rectangulares + regiones con máscara poligonal + comparación regional), pero todo vive en `src/phyloface_experimental_functions.py` y aún no se migró al paquete `src/phyloface/`. Tarea #1 cubre esa migración.
