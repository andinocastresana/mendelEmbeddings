# DEVLOG — mendelEmbeddings

Historial técnico de cambios del proyecto.
Los IDs de tarea referencian `TAREAS_PENDIENTES.md`.

Notación de tareas: `T1` = relacionada · `T1↑` = tarea creada · `T1✓` = tarea cerrada

Cada entrada incluye: hash de commit, título de una línea, IDs de tarea relacionados y detalle del cambio (mismo nivel de descripción que las entradas existentes). Las entradas se agrupan por fecha en orden cronológico inverso (lo más nuevo arriba).

---

## 2026-05-20

### `[hash pendiente]` · T1 pasos 1–6: migración del notebook experimental al paquete `phyloface` (motor casi completo) `T1`

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

### `[hash pendiente]` · Alta de arquitectura, integración de charla externa y bibliografía kinship `T1↑ T2↑ T3↑ T4↑ T5↑ T6↑ T7↑ T8↑ T9↑ T10↑ T11↑ T12↑ T13↑ T14↑ T15↑ T16↑ T17↑ T18↑ T19↑ T20↑ T21↑ T22↑ T23↑ T24↑`

- `ARQUITECTURA.md` creado (raíz del proyecto): documento vivo con el esqueleto del motor de comparación, las dos apps (primaria = niño↔progenitores; secundaria = grupos/equipos), capa de visualización y bloque de validación. Cada componente marcado con ✅/🔄/⏳ reflejando estado actual. Incluye tabla de decisiones cerradas (derivadas de la charla externa con ChatGPT en `data/input/docs/charlaChatGPT.md`) y decisiones abiertas.
- `TAREAS_PENDIENTES.md` reescrito desde cero. Se descartan las 8 tareas iniciales (puramente estructurales) y se reemplazan por 24 tareas colgadas de cada bloque numerado de `ARQUITECTURA.md`. Códigos de área = referencia a bloques (M1.x = motor, A2.x = app, V = viz, B = benchmark).
- `data/input/datasets/DATASETS.md` creado: catálogo detallado de los 4 datasets descargados (KinFaceW-I, KinFaceW-II, TSKinFace_Data, TSKinFace_SIFT) con tabla de pares por relación, estructura interna, protocolo de evaluación (5-fold CV), convenciones de nombres, uso previsto en el proyecto, datasets relacionados aún no descargados (FIW, KFVW, Family101) y bibliografía principal. La inspección de los zip se hizo sin descomprimir a disco (`unzip -l` y `unzip -p`).
- Inputs incorporados al diseño:
  - `data/input/docs/Ideas&Progress.pdf` (presentación con estado previo y objetivos planteados).
  - `data/input/docs/charlaChatGPT.md` (charla externa con propuesta de pipeline: ArcFace global → MediaPipe Face Mesh → regiones en dos niveles geométrico + visual → occlusion sensitivity → pesos diferenciados).
  - `data/input/datasets/sources.md` (URLs de los datasets).
- Hallazgo: el notebook `notebooks/phyloface_expeimental_test1.py` ya implementa el camino propuesto por la charla (MediaPipe + regiones v2 rectangulares + regiones con máscara poligonal + comparación regional), pero todo vive en `src/phyloface_experimental_functions.py` y aún no se migró al paquete `src/phyloface/`. Tarea #1 cubre esa migración.
