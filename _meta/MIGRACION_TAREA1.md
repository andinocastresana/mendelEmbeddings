# Migración Tarea #1 — del notebook experimental al paquete `phyloface`

**Origen:** `src/phyloface_experimental_functions.py` (1475 líneas, ~40 funciones)
**Consumidor:** `notebooks/phyloface_expeimental_test1.py`
**Tracker creado:** 2026-05-20

Este archivo es el mapa vivo de la migración. Cada paso del plan toca un grupo de funciones; cuando un grupo queda 100% migrado y verificado, su sección se marca ✅ y se anota la fecha + el hash del commit (si lo hubo).

Estados: ⏳ pendiente · 🔄 en progreso · ✅ migrado y verificado

---

## Plan general

| Paso | Bloque | Destino | Estado |
|------|--------|---------|--------|
| 1    | Tracker MIGRACION_TAREA1.md                | `_meta/MIGRACION_TAREA1.md`                                  | ✅ |
| 2    | Landmarks densos (MediaPipe)               | `src/phyloface/landmarks/`                                   | ✅ |
| 3    | Regiones geometría + extracción rectangular| `src/phyloface/regions/geometry.py` + `extract_rect.py`      | ✅ |
| 4    | Regiones con máscara poligonal             | `src/phyloface/regions/extract_masked.py`                    | ✅ |
| 5    | Comparador regional                        | `src/phyloface/comparator_regional.py`                       | ✅ |
| 6    | Core: io + embedder + pairs + comparator_global + metrics+= | `src/phyloface/core/*`                       | ✅ |
| 7    | Viz: detection + landmarks + regions       | `src/phyloface/viz/*`                                        | ✅ |
| 8    | Reescritura del notebook con imports nuevos| `notebooks/phyloface_expeimental_test1.py`                   | ✅ |
| 9    | Archivar el archivo experimental original  | movido a `_toReview/phyloface_experimental_functions_20260520_110102.py` (in-repo) | ✅ |

---

## Detalle por función

Cada fila indica: línea original, función, destino propuesto, estado.

### Grupo A — I/O

| Línea orig. | Función       | Destino                              | Estado |
|------------:|---------------|--------------------------------------|--------|
| 42          | `load_image`  | `core/io.py`                         | ✅ |

### Grupo B — Detección y alineación (modelo `face_records` + `selected_pair`)

| Línea orig. | Función                       | Destino                  | Estado |
|------------:|-------------------------------|--------------------------|--------|
| 74          | `init_face_app`               | `core/pairs.py`          | ✅ |
| 94          | `expand_bbox`                 | `core/pairs.py`          | ✅ |
| 122         | `detect_faces_in_image`       | `core/pairs.py`          | ✅ |
| 179         | `detect_faces_in_images`      | `core/pairs.py`          | ✅ |
| 243         | `get_face_record`             | `core/pairs.py`          | ✅ |
| 259         | `align_face_from_record`      | `core/pairs.py`          | ✅ |
| 310         | `build_selected_pair`         | `core/pairs.py`          | ✅ |

### Grupo C — Embeddings globales + métricas

| Línea orig. | Función                              | Destino                       | Estado |
|------------:|--------------------------------------|-------------------------------|--------|
| 379         | `get_recognition_model`              | `core/embedder.py`            | ✅ |
| 392         | `l2_normalize`                       | `core/metrics.py` (extender)  | ✅ |
| 406         | `cosine_similarity`                  | `core/metrics.py` — **REEMPLAZADA** por la versión experimental (usa `l2_normalize`) | ✅ |
| 418         | `cosine_distance`                    | `core/metrics.py` (extender)  | ✅ |
| 428         | `euclidean_distance`                 | `core/metrics.py` — **REEMPLAZADA** por la versión experimental (normaliza antes; **cambio funcional**, ver bitácora) | ✅ |
| 440         | `extract_embedding_from_aligned`     | `core/embedder.py`            | ✅ |
| 454         | `compute_global_metrics`             | `core/comparator_global.py`   | ✅ |
| 940         | `print_global_summary`               | `core/comparator_global.py`   | ✅ |

### Grupo D — Landmarks densos (MediaPipe)

| Línea orig. | Función                          | Destino                            | Estado |
|------------:|----------------------------------|------------------------------------|--------|
| 505         | `init_face_mesh`                 | `landmarks/mediapipe_mesh.py`      | ✅ |
| 526         | `get_face_mesh_landmarks`        | `landmarks/mediapipe_mesh.py`      | ✅ |
| 550         | `add_dense_landmarks_to_pair`    | `landmarks/mediapipe_mesh.py`      | ✅ |

### Grupo E — Regiones v2 rectangulares

| Línea orig. | Función                          | Destino                       | Estado |
|------------:|----------------------------------|-------------------------------|--------|
| 588         | `connection_set_to_index_list`   | `regions/geometry.py`         | ✅ |
| 623         | `get_region_bbox`                | `regions/geometry.py`         | ✅ |
| 651         | `crop_from_bbox`                 | `regions/geometry.py`         | ✅ |
| 662         | `get_forehead_bbox`              | `regions/geometry.py`         | ✅ |
| 694         | `get_chin_bbox_refined`          | `regions/geometry.py`         | ✅ |
| 745         | `extract_regions_v2`             | `regions/extract_rect.py`     | ✅ |
| 816         | `add_regions_v2_to_pair`         | `regions/extract_rect.py`     | ✅ |

### Grupo F — Comparación regional rectangular

| Línea orig. | Función                       | Destino                       | Estado |
|------------:|-------------------------------|-------------------------------|--------|
| 877         | `resize_to_match`             | `comparator_regional.py`      | ✅ |
| 890         | `grayscale_patch_cosine`      | `comparator_regional.py`      | ✅ |
| 909         | `compare_regions_v2`          | `comparator_regional.py`      | ✅ |
| 964         | `print_regional_summary`      | `comparator_regional.py`      | ✅ |

### Grupo G — Regiones v2 con máscara poligonal

| Línea orig. | Función                              | Destino                          | Estado |
|------------:|--------------------------------------|----------------------------------|--------|
| 1016        | `create_region_mask_from_points`     | `regions/extract_masked.py`      | ✅ |
| 1047        | `crop_mask_and_image`                | `regions/extract_masked.py`      | ✅ |
| 1116        | `extract_regions_v2_masked`          | `regions/extract_masked.py`      | ✅ |
| 1287        | `add_regions_v2_masked_to_pair`      | `regions/extract_masked.py`      | ✅ |
| 1000-1010 (constantes) | 6 × `*_POLYGON_IDX`        | `regions/geometry.py` (agregadas)| ✅ |

### Grupo H — Comparación regional enmascarada

| Línea orig. | Función                              | Destino                       | Estado |
|------------:|--------------------------------------|-------------------------------|--------|
| 1413        | `masked_grayscale_patch_cosine`      | `comparator_regional.py`      | ✅ |
| 1435        | `compare_regions_v2_masked`          | `comparator_regional.py`      | ✅ |

### Grupo VIZ — Visualizaciones (transversal a B, D, E, G)

| Línea orig. | Función                          | Destino                | Estado |
|------------:|----------------------------------|------------------------|--------|
| 208         | `plot_detected_faces`            | `viz/detection.py`     | ✅ |
| 340         | `plot_face_triplet`              | `viz/detection.py`     | ✅ |
| 566         | `plot_face_with_landmarks`       | `viz/landmarks.py`     | ✅ |
| 839         | `plot_regions_v2`                | `viz/regions.py`       | ✅ |
| 1070        | `plot_face_regions_overlay`      | `viz/regions.py`       | ✅ |
| 1310        | `plot_regions_v2_masked`         | `viz/regions.py`       | ✅ |
| 1365        | `plot_region_detail`             | `viz/regions.py`       | ✅ |

---

## Bitácora de migración

Cada vez que un paso se completa, se agrega aquí una línea con fecha + qué pasó.

- **2026-05-20** — Paso 1 ✅. Tracker creado. Plan confirmado con el usuario.
- **2026-05-20** — Paso 2 ✅. Subpaquete `landmarks/` creado:
  - `src/phyloface/landmarks/__init__.py` (re-export público)
  - `src/phyloface/landmarks/mediapipe_mesh.py` (ID `PHYLOFACE_LANDMARKS_001 v1.0`)
  - 3 funciones migradas: `init_face_mesh`, `get_face_mesh_landmarks`, `add_dense_landmarks_to_pair`.
  - Smoke test pasó: imports OK + `FaceMesh` construido sin error.
  - Notebook todavía importa de `phyloface_experimental_functions` (no roto; la actualización del notebook es el Paso 8).
- **2026-05-20** — Pasos 8 + 9 ✅ → **cierra la Tarea #1 completa (40/40 funciones migradas + notebook funcional + archivo original archivado)**.
  - **Paso 8**: `notebooks/phyloface_expeimental_test1.py` reescrito (ID `PHYLOFACE_NOTEBOOK_002 v2.0`). Imports actualizados al paquete migrado; backend `matplotlib.use("Agg")`; carpeta de salida `data/output/notebook_runs_<TIMESTAMP>/` con helper `save_current(name)` que guarda el plot activo después de cada celda de viz. Notebook corrió end-to-end con `fraternos_jovenes.jpg` + `fraternosChacabuco8.jpg`; 15 plots generados en `data/output/notebook_runs_20260520_103726/`; resultados rect vs masked numéricamente consistentes con el comportamiento del experimental. NaN esperados en `left_eye`/`right_eye` del path masked (heredado del comparator_regional, documentado).
  - **Paso 9**: `src/phyloface_experimental_functions.py` movido con `git mv` a `_toReview/phyloface_experimental_functions_20260520_110102.py` (dentro del repo para que viaje versionado; Git lo trackea como rename, preserva historia). Sufijo de timestamp respeta la regla del proyecto (CLAUDE.md de `Proyectos/`).
  - **Infraestructura agregada en paralelo a la migración**:
    - `tests/smoke/` con README + primer smoke versionado (`test_paso_7_viz.py`).
    - `.claude/settings.json` con allowlist commit-able (`rclone listremotes`, `rclone lsjson *`, `python3 tests/smoke/*`).
    - `.claude/settings.local.json` con `defaultMode: "acceptEdits"` (local, no commit-able).
    - Memorias nuevas: `feedback_script_header_convention.md`, `project_web_stack_decision.md`, `reference_ssh_github_multi_account.md`.
    - Nota global en `~/Proyectos/0_code_(gitHub)/NOTAS_CONFIGURACION.md` sobre el fix de SSH multi-cuenta.
    - 2 episodios capturados en el KG (`IA/memories/_meta/episodes/2026-05-20-*`) + activación del slot `IA/memories/mendelEmbeddings/`.
- **2026-05-20** — Paso 7 ✅. Subpaquete `viz/` (3 archivos nuevos + __init__ actualizado):
  - `src/phyloface/viz/detection.py` (ID `PHYLOFACE_VIZ_001 v1.0`) — `plot_detected_faces`, `plot_face_triplet`.
  - `src/phyloface/viz/landmarks.py` (mismo ID) — `plot_face_with_landmarks`.
  - `src/phyloface/viz/regions.py` (mismo ID) — `plot_regions_v2`, `plot_face_regions_overlay`, `plot_regions_v2_masked`, `plot_region_detail`.
  - `src/phyloface/viz/__init__.py` re-exporta las 7 funciones nuevas + las 2 pre-existentes de `heatmap.py` (`plot_similarity_heatmap`, `add_face_thumbnail`).
  - **Cambio de patrón**: este es el primer smoke test como archivo versionado (no `python3 -c "..."`). Vive en `tests/smoke/test_paso_7_viz.py` con ID `PHYLOFACE_SMOKE_007 v1.0`. Permite ejecutarlo sin prompt vía la nueva regla `Bash(python3 tests/smoke/*)` en `.claude/settings.json`. Convención documentada en `tests/smoke/README.md`.
  - Smoke test pasó: 7 funciones ejecutadas con backend `matplotlib.use('Agg')` sobre datos sintéticos. Cubrió: panel multi-imagen y mono-imagen para `plot_detected_faces`, panel 1×3 de triplete, scatter de 478 landmarks, grid N×2 para `plot_regions_v2`, overlay color-mapped de máscaras, los 3 modos de `plot_regions_v2_masked` (rect/mask/masked) + caso `len(region_names)==1`, detalle 1×4 para A y B, y validación de `side` inválido en `plot_region_detail`.
- **2026-05-20** — Paso 6d ✅ → cierra el Paso 6 entero. Comparator global:
  - `src/phyloface/core/comparator_global.py` (ID `PHYLOFACE_COMPARATOR_GLOBAL_001 v1.0`) — 2 funciones: `compute_global_metrics` y `print_global_summary`.
  - Integrador: usa embedder (6b) + metrics (6a) + selected_pair (6c). Anida `embedding_qc` (5 métricas de control) y `global_scores` (6 métricas principales) dentro del `selected_pair`.
  - Smoke test pasó: 6 claves nuevas + 4 embeddings (512, float32) + QC con delta = post − original consistente + scores con `cosine_distance = 1 - cosine_similarity` + euclídeas en [0, 2] (nueva semántica del 6a) + caso degenerado "misma cara A=B" → cosine_post=1.0, euclídea≈0 + `print_global_summary` legible.
- **2026-05-20** — Paso 6c ✅. Pairs (detección + selección + alineación) — el subpaso más grande del Paso 6:
  - `src/phyloface/core/pairs.py` (ID `PHYLOFACE_PAIRS_001 v1.0`) — 7 funciones migradas tal cual: `init_face_app`, `expand_bbox`, `detect_faces_in_image`, `detect_faces_in_images`, `get_face_record`, `align_face_from_record`, `build_selected_pair`.
  - Smoke test cubrió: `expand_bbox` con padding normal + clamp en dos esquinas + `get_face_record` con OK y StopIteration + mock de FaceAnalysis para `detect_faces_in_image[s]` verificando orden L→R por x1 e IDs F1_R1/F1_R2/F2_R1/F2_R2 + `align_face_from_record` con `face_align.estimate_norm` REAL (sin modelo cargado) + casos de error (margin_ratio inválido, kps=None) + `build_selected_pair` integrador.
  - **Restricción heredada de InsightFace documentada**: `align_face_from_record` requiere `image_size` múltiplo de 112 o 128 (assertion interna de `face_align.estimate_norm`). Anotado en el docstring del módulo, no se cambia comportamiento.
  - Coexistencia con `core/detector.py` clarificada en el header del módulo: ambos siguen vivos en paralelo (`detector.py` para caché multi-cara, `pairs.py` para flujo experimental notebook-like). Unificación postergada para tarea futura.
- **2026-05-20** — Paso 6b ✅. Embedder:
  - `src/phyloface/core/embedder.py` (ID `PHYLOFACE_EMBEDDER_001 v1.0`) — 2 funciones migradas tal cual: `get_recognition_model`, `extract_embedding_from_aligned`.
  - Smoke test con mocks (sin cargar InsightFace, para evitar los 3-5s de init): `get_recognition_model` encuentra recognition entre detection/landmark/recognition + lanza RuntimeError si no hay ninguno con `get_feat` + `extract_embedding_from_aligned` tolera shapes (1,D) y (D,) + integración con `l2_normalize` + `cosine_similarity` del Paso 6a.
- **2026-05-20** — Paso 6a ✅. Núcleo base del `core/`:
  - `src/phyloface/core/io.py` (ID `PHYLOFACE_CORE_BASE_001 v1.0`) — `load_image` migrada tal cual.
  - `src/phyloface/core/metrics.py` reescrito (mismo ID) — agregadas `l2_normalize` y `cosine_distance`; **reemplazadas** `cosine_similarity` y `euclidean_distance` por las versiones del experimental (que normalizan vía `l2_normalize`).
  - **Cambio funcional decidido con el usuario (opción "Reemplazar por la versión experimental"):**
    - `euclidean_distance` ANTES: `norm(v1 - v2)` sin normalizar → valores grandes proporcionales a la magnitud de los embeddings (ArcFace tiene norma ~22).
    - `euclidean_distance` AHORA: `norm(l2norm(v1) - l2norm(v2))` → valores en `[0, 2]`.
    - **Consecuencia**: cualquier umbral o resultado numérico ya calculado con la euclídea vieja no es comparable. Hoy no hay umbrales calibrados (Tarea #6 del proyecto va a generarlos contra KinFaceW), así que el impacto es chico. Consumidores actuales (`core/comparator.py`, `app/run_pairwise_heatmap.py`, etc.) siguen funcionando — solo cambian sus valores numéricos.
  - `cosine_similarity`: cambio defensivo (maneja norm=0, fuerza float32, ravel); resultados matemáticamente equivalentes para vectores no-cero.
  - Smoke test pasó: l2_normalize con norm=0 y shapes raras + identidad/ortogonal/opuesto para coseno + euclídea con misma dirección (=0), ortogonales (=√2), opuestos (=2) + `get_metric_function` devuelve las nuevas + `load_image` sobre imagen real (892×819, BrunoFondoBlanco.jpeg) + FileNotFoundError con detalle.
- **2026-05-20** — Paso 5 ✅. Módulo `comparator_regional.py` creado:
  - `src/phyloface/comparator_regional.py` (ID `PHYLOFACE_COMPARATOR_REGIONAL_001 v1.0`) — 6 funciones (rect + masked + helpers + print).
  - Smoke test pasó: `grayscale_patch_cosine(idénticos)=1.0`, ruido vs ruido cerca de 0, `masked_*(all-zero)=NaN`, `compare_regions_v2`/`_masked` sobre selected_pair sintético con left_eye + mouth, skip silencioso con crop vacío, `print_regional_summary` no crashea.
  - **Decisión arquitectónica anotada en el módulo**: `print_regional_summary` solo entiende la clave `gray_cosine` (path rect). Se preservó tal cual el original; si se quiere unificar masked + rect, hacerlo en una tarea separada para no introducir cambios funcionales encubiertos.
  - **Cambio de rumbo del proyecto** (2026-05-20, paralelo a este paso): el proyecto va a montarse como servicio web público (FastAPI + SPA). Decisiones anotadas en `ARQUITECTURA.md` §5 y memoria `project_web_stack_decision.md`. La Tarea #1 sigue su curso normal sin cambios.
- **2026-05-20** — Paso 4 ✅. Subpaquete `regions/` completado con el path enmascarado:
  - `src/phyloface/regions/extract_masked.py` (ID `PHYLOFACE_REGIONS_002 v1.0`) — 4 funciones.
  - `src/phyloface/regions/geometry.py` extendido con 6 constantes `*_POLYGON_IDX` (contornos ordenados para `cv2.fillPoly`).
  - `src/phyloface/regions/__init__.py` actualizado para re-exportar todo lo nuevo.
  - Smoke test pasó: máscara por polígono + máscara por convex hull + máscara vacía + crop alineado con fondo a 0 + las 12 regiones con la estructura completa de 8 claves (`bbox, mask, crop_rgb, crop_mask, crop_masked_rgb, landmark_idx, polygon_idx, source`) + frente con máscara rectangular + sobrescritura intencional de `regions_v2`.
- **2026-05-20** — Paso 3 ✅. Subpaquete `regions/` (parte rectangular) creado:
  - `src/phyloface/regions/__init__.py` (re-export público de helpers, constantes y extractor)
  - `src/phyloface/regions/geometry.py` (ID `PHYLOFACE_REGIONS_001 v1.0`) — 5 funciones + 11 constantes de índices por región
  - `src/phyloface/regions/extract_rect.py` (ID `PHYLOFACE_REGIONS_001 v1.0`) — 2 funciones
  - 7 funciones migradas en total.
  - Smoke test pasó: imports + constantes válidas + `extract_regions_v2` devuelve las 12 regiones esperadas con la estructura `{bbox, crop_rgb, landmark_idx, source}` + `add_regions_v2_to_pair` pobla A y B.
  - Pendiente para Tarea #2 del proyecto: formalizar lista canónica de regiones (los subconjuntos `*_IDX` manuales se migraron tal cual del archivo original).

---

## Verificación al cierre

Criterios para considerar la Tarea #1 completamente cerrada:

1. ✅ Todas las funciones migradas — 40/40 en ✅.
2. ✅ El notebook `phyloface_expeimental_test1.py` corre end-to-end con los imports nuevos y produce 15 plots PNG visualmente validados por el usuario.
3. ✅ `src/phyloface_experimental_functions.py` movido a `_toReview/phyloface_experimental_functions_20260520_110102.py` (sufijado con fecha, no borrado).
4. ⏳ `TAREAS_PENDIENTES.md` actualizado: Tarea #1 → Completadas. (a cerrar en el próximo commit)
5. ⏳ `DEVLOG.md` con la entrada de cierre. (a cerrar en el próximo commit)

**Tarea #1 CERRADA** el 2026-05-20.
