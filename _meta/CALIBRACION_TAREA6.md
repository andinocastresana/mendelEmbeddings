# Calibración de umbrales — Tarea #6 (procedimiento)

> Doc vivo del procedimiento de calibración data-driven de umbrales de
> parentesco sobre KinFaceW-I/II, y de la visualización web asociada
> (histogramas + marcador del cosine propio). Append/edit a medida que avanza.
> Iniciado 2026-05-23.

## Objetivo

Hoy el comparador (Track 2a) y el árbol (Track 2b) producen un **cosine crudo**
sin interpretación. #6 convierte ese número en un **veredicto de parentesco**
con respaldo cuantitativo: umbral data-driven + ROC/AUC sobre datasets
etiquetados (KinFaceW-I/II), reportado por relación y por métrica.

Es prerequisito de la App primaria #12 (necesita scores interpretables) y de los
protocolos de benchmark #17/#18.

## Stack de embedding (lo que se calibra)

- Motor Python: InsightFace `buffalo_l` → submodelo de reconocimiento
  `w600k_r50` (512-d), vía `phyloface.core.embedder.extract_embedding_from_aligned`.
- Métrica: `phyloface.core.metrics.cosine_similarity` (y `euclidean_distance`
  sobre vectores L2-normalizados, rango [0,2]).
- El mismo embedding corre client-side en el browser (ONNX Runtime Web), así que
  los umbrales calibrados acá son directamente aplicables a la web.

## Plan en dos fases

### Fase A — Calibración (Python, núcleo de #6)
1. Correr el pipeline de embedding sobre KinFaceW-I/II.
2. Para cada par etiquetado (kin / non-kin) computar coseno + euclídea.
3. Distribuciones positiva vs negativa, por relación (FS/FD/MS/MD) y agregada.
4. Umbral data-driven con **5-fold CV** (umbral elegido en 4 folds, evaluado en
   el 5º, promediado) — número honesto sin fuga de datos. Criterios: Youden's J
   (max TPR−FPR), EER, max accuracy. Reportar ROC + AUC.
5. Salida = **artefacto JSON versionado** (contrato hacia la web): ver schema abajo.

### Fase B — Visualización (web)
1. Solapa nueva que lee el JSON y dibuja los **histogramas** kin vs non-kin por
   relación/métrica.
2. Al clickear una métrica en el comparador/árbol, **modal chico** que muestra
   dónde cae ese cosine propio sobre la distribución calibrada (percentil +
   veredicto vs umbral).

## Decisiones tomadas

- **Caras de KinFaceW tratadas como pre-alineadas** (resize 64→112), NO se corre
  detección Face Mesh sobre 64px. Validado por el spike (ver abajo): el camino
  "pre-aligned → `extract_embedding_from_aligned`" da señal clara. Razón: las
  imágenes ya vienen recortadas/alineadas a 64×64 por los autores.
- **Embeddings incluidos en el artefacto de export del árbol** etiquetados con
  `modelVersion` (ver Tarea #26 paso 6) — mismo `w600k_r50`, así que la
  calibración aplica a los embeddings cacheados del browser sin recomputar.
- Por ahora pares positivos del spike por convención de nombres; la Fase A
  completa parseará los `meta_data/*_pairs.mat` para usar los folds y negativos
  oficiales (protocolo reproducible, solapa con #17).

## Spike de de-risking (2026-05-23) — `scripts/spike_kinfacew_embeddings.py`

**Pregunta**: ¿hay señal de parentesco usable con coseno crudo sobre las caras
64px de KinFaceW? **Sí.**

| relación | n | cos_pos | cos_neg | AUC rank |
|----------|---|---------|---------|----------|
| Father-Son (mismo género) | 60 | 0.223±0.121 | 0.102±0.089 | **0.791** |
| Mother-Daughter (mismo género) | 60 | 0.278±0.129 | 0.143±0.122 | **0.764** |
| Mother-Son (cross-género) | 60 | 0.240±0.115 | 0.136±0.102 | **0.756** |
| Father-Daughter (cross-género) | 60 | 0.180±0.116 | 0.112±0.112 | **0.669** |
| **GLOBAL** | 240 | 0.230±0.125 | 0.123±0.108 | **0.740** |

- AUC rank-based = P(cos_pos > cos_neg). 0.74 global = señal clara.
- Patrón esperado confirmado: mismo-género > cross-género; Father-Daughter el
  más difícil (señal débil). → **los umbrales y pesos deben ser por relación**,
  no globales.
- Sanity: embeddings dim 512, norma mínima 13.06 (no degenerados).
- Negativos "justos": padre_i ↔ hijo_perm(i) dentro de la misma relación
  (composición de género balanceada).
- Cap de 60 pares/relación para acotar runtime del spike; la Fase A usa todos.

## Procedimiento de ejecución (monitoreado — regla de recursos)

Toda corrida Python de #6 va envuelta en el wrapper de monitoreo:

```bash
./scripts/test-monitored.sh python3 scripts/spike_kinfacew_embeddings.py
```

Loguea CPU/temp a `.test-resources.log` (gitignored) e imprime resumen al cierre.

**Aviso térmico observado en el spike**: pico **94°C**, CPU 83%, WARN sostenido
a 87°C — buffalo_l corre los 5 submodelos en CPU. Para la Fase A completa
(miles de imágenes) mitigar: cargar **solo** el modelo de reconocimiento (evitar
detección/landmark/genderage que no usamos en el camino pre-aligned), cachear
embeddings por archivo, y/o batching con pausas. A monitorear en cada corrida.

## Schema del artefacto (contrato Fase A → Fase B) — borrador

```jsonc
{
  "v": 1,
  "computedAt": <epoch>,
  "modelVersion": "w600k_r50",
  "dataset": "KinFaceW-I",            // + entrada paralela para -II
  "metric": "cosine",                 // y "euclidean"
  "protocol": "5fold-cv-unrestricted",
  "perRelation": {
    "FS": {
      "n_pos": 156, "n_neg": 156,
      "threshold": 0.18,              // Youden's J promediado sobre folds
      "accuracy": 0.71, "auc": 0.79,
      "histogram": { "bins": [...], "pos_counts": [...], "neg_counts": [...] }
    }
    // FD, MS, MD, ALL
  }
}
```

(El histograma pre-binneado viaja en el JSON para que la web no necesite los
embeddings ni las imágenes — solo dibuja barras y ubica el cosine propio.)

## Resultados Fase A — KinFaceW-I (2026-05-23) — `scripts/run_calibration_kinfacew.py`

Calibración completa, 5-fold CV oficial (folds del `.mat`), umbral de Youden por
relación sobre coseno. Artefacto: `data/output/calibration/KinFaceW-I_calibration.json`.

| relación | n_pos | acc 5-CV | umbral coseno | AUC |
|----------|-------|----------|---------------|-----|
| Father-Son (mismo género) | 156 | 73.1% | 0.123 | 0.812 |
| Mother-Daughter (mismo género) | 127 | 65.5% | 0.231 | 0.746 |
| Father-Daughter (cross-género) | 134 | 62.4% | 0.108 | 0.677 |
| Mother-Son (cross-género) | 116 | 59.9% | 0.173 | 0.681 |
| **ALL** | 533 | **66.6%** | 0.138 | **0.727** |

- Baseline honesto del **coseno crudo** (sin aprendizaje de parentesco), en línea
  con NRML clásico (72-77%). El SOTA ~81% (FaCoRNet) requiere cabeza aprendida.
- **Umbrales muy distintos por relación** (0.108–0.231) → la calibración por
  relación era necesaria. MD tiene umbral alto porque sus cosines (pos y neg)
  son más altos en general.
- Artefacto JSON incluye coseno y euclídea + histogramas pre-binneados (contrato
  Fase B).
- **Térmico**: pico 94°C, batching adaptativo activado (pausas 6s sobre 85°C),
  promedio 80°C. La corrida completa (~1066 imágenes) cerró sin runaway.
- El smoke `--limit 20` dio `nan` (las primeras 20 filas del `.mat` son todas
  positivas → sin negativos); la corrida completa balancea labels y folds. Nota
  para futuros smokes: usar muestreo estratificado, no slice de las primeras N.

## Hallazgos del informe bibliográfico que ajustan el plan

(De `_meta/BIBLIOGRAFIA_KINSHIP_DATASETS.md`, 2026-05-23.)

- **KinFaceW-I es la métrica PRIMARIA, no -II.** Dawson et al. 2018 mostraron que
  KinFaceW-II tiene sesgo "from same photo" (los positivos son recortes de la
  MISMA foto familiar → un clasificador puede ganar >90% detectando "misma foto"
  sin medir parentesco). Nuestro spike usó -I (fotos distintas) → el 0.74 es
  honesto. Reportar -II solo como referencia, marcando el sesgo.
- **El resize 64→112 puede estar capando el AUC.** Las imágenes distribuidas son
  64×64 ya recortadas; ArcFace espera 112×112. Mitigación a evaluar en Fase A:
  pedir/usar las imágenes originales de KinFaceW y re-alinear con nuestro pipeline
  InsightFace, en vez del crop 64px. (Decidir si vale el esfuerzo según cuánto
  sube el AUC en un sub-test.)
- **Usar SIEMPRE los folds y negativos oficiales** (`*_pairs.mat`). Re-partir al
  azar mete leakage family-aware y negativos no reproducibles.
- **Reportar por relación + mean**, nunca solo el mean (oculta que cross-género
  es mucho peor — confirmado por nuestro spike: FD 0.67 vs FS 0.79).
- **El umbral por relación (Mejora 1 del informe) ES el núcleo de #6** (esfuerzo
  muy bajo, +2-5 pts). La **Mejora 2 (cabeza MLP de metric-learning sobre los
  embeddings ArcFace)** es el salto grande (+7-15 pts), **portable a cliente**
  como segundo modelo ONNX chico — candidata natural a tarea siguiente tras #6.

## Próximos pasos (Fase A)
1. Parser de `meta_data/*_pairs.mat` (folds + positivos/negativos oficiales).
2. Runner de embeddings con cache + carga de solo-reconocimiento (mitigación térmica).
3. Calibración 5-fold (Youden's J / EER / max-acc) + ROC/AUC por relación y métrica.
   Reportar -I primario; -II con disclaimer de sesgo.
4. Emitir el JSON versionado en `data/output/calibration/` (gitignored) + copia
   liviana servible al cliente.
5. (Opcional, según gap) sub-test con imágenes originales re-alineadas a 112.
6. Tras #6: evaluar Mejora 2 (cabeza MLP ONNX-portable) como tarea nueva.

## Relacionado
- Bibliografía de datasets + métodos SOTA + mejoras: `_meta/BIBLIOGRAFIA_KINSHIP_DATASETS.md` (en preparación por agente de investigación).
- Episodio KG de la advertencia "verificar convención antes que números": el spike confirma el camino pre-aligned.
