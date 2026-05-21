# Tareas pendientes — mendelEmbeddings

Estados posibles: `pendiente` | `en progreso` | `bloqueada` | `hecha`

Los IDs son inmutables: al cerrar una tarea no se reusa el número.
Las tareas hechas se mueven a la sección **Completadas** al final, conservando su ID.

Los códigos de área (`M1.4`, `M2.1`, etc.) refieren a los bloques numerados en `ARQUITECTURA.md` (M = Motor, A = App, V = Viz, B = Benchmark).

---

## Activas

| #  | Descripción | Área | Estado | Creado | Actualizado |
|----|-------------|------|--------|--------|-------------|
| 2  | Formalizar la **lista canónica de regiones** (nombres, landmarks fuente, polígonos) como contrato del motor — documentar en `phyloface/regions/canonical.py` o YAML de config | M1.4 | pendiente | 2026-05-19 | 2026-05-19 |
| 3  | Anotar qué fue "regions v1" y por qué se descartó (deuda histórica para no repetir) | M1.4 | pendiente | 2026-05-19 | 2026-05-19 |
| 4  | Implementar **features geométricas Nivel A** — distancias entre landmarks, proporciones, ángulos, simetrías | M1.5 | pendiente | 2026-05-19 | 2026-05-19 |
| 5  | Validar la calidad de los **embeddings por región Nivel B** (re-aplicar `w600k_r50` a crops/máscaras) — sanity check contra pares de KinFaceW | M1.5 | pendiente | 2026-05-19 | 2026-05-19 |
| 6  | Implementar **calibración de umbrales** sobre KinFaceW-I/II — estimar umbral data-driven para coseno y euclídea | M1.6, B | pendiente | 2026-05-19 | 2026-05-19 |
| 7  | Extender `cache.py` para soportar embeddings por región + versión-de-regiones en la clave (evitar mezclar v1, v2, v2-masked) | M1.7 | pendiente | 2026-05-19 | 2026-05-19 |
| 8  | Arreglar carga de modelo **`antelopev2`** en InsightFace (hoy roto) y comparar contra `buffalo_l` | M1.8 | pendiente | 2026-05-19 | 2026-05-19 |
| 9  | Implementar **heatmap por regiones** sobre la cara — coloreo por contribución al score global | M1.9, V | pendiente | 2026-05-19 | 2026-05-19 |
| 10 | Implementar **occlusion sensitivity** unidireccional (ventana 12×12 / 16×16, stride 4–6) sobre el rostro alineado | M1.9 | pendiente | 2026-05-19 | 2026-05-19 |
| 11 | Diseñar **sistema de pesos por región** (ojos/perioculares > nariz, boca, contorno) — calibrado contra KinFaceW | M1.10 | pendiente | 2026-05-19 | 2026-05-19 |
| 12 | Construir **App primaria — Parecido niño ↔ progenitores** (entrada: 3 fotos · salida: score global ×2 + scores regionales ×2 + visualización) | A2.1 | pendiente | 2026-05-19 | 2026-05-19 |
| 13 | Robustecer `run_pairwise_heatmap.py` y migrarlo bajo la API unificada del paquete | A2.2, M | pendiente | 2026-05-19 | 2026-05-19 |
| 14 | **Boxplots intra-grupo** de similitud (uno por selección) | A2.2, V | pendiente | 2026-05-19 | 2026-05-19 |
| 15 | **Distancia inter-grupo** + dendrograma / grafo entre selecciones (idea árbol de países) | A2.3, V | pendiente | 2026-05-19 | 2026-05-19 |
| 16 | **Radar/spider chart** de scores regionales para la app primaria | V | pendiente | 2026-05-19 | 2026-05-19 |
| 17 | Implementar el **protocolo KinFaceW** (5-fold CV, 3 settings) — reportar accuracy + ROC | B4.4 | pendiente | 2026-05-19 | 2026-05-19 |
| 18 | Implementar el **protocolo TSKinFace** (5-fold CV con folds del ReadMe) — accuracy por relación FM-S / FM-D | B4.4 | pendiente | 2026-05-19 | 2026-05-19 |
| 19 | Evaluar incorporar **FIW (Families In the Wild)** como tercer dataset de validación | B4.1 | pendiente | 2026-05-19 | 2026-05-19 |
| 20 | Definir convención de rutas relativas (`../data/mendelEmbeddings/...`) acorde a la estructura objetivo de `Proyectos/` | Infra | pendiente | 2026-05-19 | 2026-05-19 |
| 21 | Escribir `README.md` (descripción, instalación, cómo correr, estructura de `data/`, link a `ARQUITECTURA.md` y `DATASETS.md`) | Docs | pendiente | 2026-05-19 | 2026-05-19 |
| 22 | Smoke test mínimo del pipeline en `tests/` (detección + embedding + comparación) | Tests | pendiente | 2026-05-19 | 2026-05-19 |
| 23 | Documentar uso de `sync_notebooks.sh` y `jupytext.toml` en el README | Docs | pendiente | 2026-05-19 | 2026-05-19 |
| 24 | Documentar los submodelos de InsightFace (`det_10g`, `w600k_r50`, `landmark_*`, `genderage`) en `phyloface/core/models.py` o en el README | Docs | pendiente | 2026-05-19 | 2026-05-19 |
| 26 | **Track 2b — Comparador con árbol genealógico (pedigree formal)**. Página nueva (no reemplaza al comparador 3-slot del Track 2a). Cada persona admite máx. 1 padre + 1 madre (pedigree formal). Fotos ilimitadas por persona pendiente decidir (MVP 1 foto/persona). Persistencia IndexedDB local por default + export/import (JSON con metadata del árbol + imágenes empaquetadas, ej. base64 o ZIP). Selección interactiva: el usuario clickea 2 personas del árbol y se computa el cosine on-demand. Las imágenes nunca salen del browser por default (export es opt-in). Decisiones abiertas: layout (visual de pedigree clásico vs lista + edges), múltiples fotos por persona (para promediar embeddings o elegir mejor), matriz NxN automática vs solo pares on-demand. | Web | pendiente | 2026-05-21 | 2026-05-21 |

## Completadas

| #  | Descripción | Área | Estado | Creado | Cerrado |
|----|-------------|------|--------|--------|---------|
| 25 | **Track 2a — MVP comparador anónimo browser**. (a) spike #003 alineación canónica JS ✅ commit `7af68d6`. (b) spike #004 detección JS + pipeline e2e ✅ commit `b3129af` (GLOBAL PASS sobre 4 imágenes diversas, mean cosine 0.98). (c) UI MVP del comparador ✅ — `client/src/Comparator.tsx` v2.1 con 3 slots (P1 · Hijo/a · P2), drag-and-drop por slot, dropdown de rol en laterales (default Padre/Madre, opciones Hermano/a, Tío/a, Abuelo/a, Otro), botón ✕ Quitar, comparación de 2 o 3 slots (1 o 2 cosines según combinación), preview de caras alineadas 112×112. Pipeline e2e reusado vía `lib/pipeline.ts`. Imágenes nunca salen del browser. | Web | **hecha** | 2026-05-21 | 2026-05-21 |
| 1  | Migrar funciones del notebook experimental (`src/phyloface_experimental_functions.py`) al paquete `src/phyloface/` — separar en `regions/`, `landmarks/`, `comparator_regional.py` según corresponda | M1.3, M1.4, M1.5 | **hecha** | 2026-05-19 | 2026-05-20 |

Notas sobre el cierre de Tarea #1 (ver `_meta/MIGRACION_TAREA1.md` para detalle por sub-paso):
- 40/40 funciones migradas en 9 pasos. Cada sub-paso verificado con smoke test.
- Notebook reescrito con imports nuevos, corre end-to-end con datos reales, 15 plots PNG validados.
- Cambio funcional decidido durante la migración: `euclidean_distance` y `cosine_similarity` reemplazadas por las versiones del experimental que normalizan con `l2_normalize` (euclídea ahora en `[0, 2]` en vez de magnitud cruda). Sin impacto en código ya productivo: no había umbrales calibrados todavía (Tarea #6 los generará data-driven).
- Archivo original archivado en `_toReview/phyloface_experimental_functions_20260520_110102.py` (no borrado).
- Infraestructura extra agregada en paralelo: `tests/smoke/` (convención de smoke tests versionados), `.claude/settings.json` (allowlist commit-able), `_meta/MIGRACION_TAREA1.md` (tracker fino), `ARQUITECTURA.md` §5 (decisión de stack web público).
