# Smoke FairFace

Fuente revisada: `HuggingFaceM4/FairFace`.

## Qué es

FairFace en Hugging Face es un dataset de imágenes, no un modelo de inferencia. La variante probada fue:

- dataset: `HuggingFaceM4/FairFace`
- config: `0.25`
- split: `train`
- imágenes: 224x224
- etiquetas: `age`, `gender`, `race`, `service_test`

Etiquetas decodificadas usadas en el smoke:

- `age`: `0-2`, `3-9`, `10-19`, `20-29`, `30-39`, `40-49`, `50-59`, `60-69`, `70+`
- `gender`: `Male`, `Female`
- `race`: `East Asian`, `Indian`, `Black`, `White`, `Middle Eastern`, `Latino_Hispanic`, `Southeast Asian`

## Artefactos locales

Los outputs quedaron en `data/output/fairface_smoke/sample_0_25_train_n16/`:

- `manifest.json`: 16 imágenes FairFace guardadas localmente con etiquetas.
- `qc_embeddings.json`: corrida InsightFace `buffalo_l`, `det_thresh=0.20`.
- `qc_embeddings_det005.json`: corrida InsightFace `buffalo_l`, `det_thresh=0.05`.

`data/` está gitignored, así que estos artefactos no entran al repo.

## Resultado

| Corrida | Aceptados | Rechazados | Observación |
|---|---:|---:|---|
| `det_thresh=0.20` | 5/16 | 11/16 | Compatible, pero umbral demasiado alto para crops FairFace |
| `det_thresh=0.05` | 15/16 | 1/16 | Genera embeddings 512D, pero aparecen detecciones múltiples/espurias |

En `det_thresh=0.05`, todos los aceptados tienen embedding de 512 dimensiones. El rango de `det_score` fue `0.0503..0.3139`.

## Lectura técnica

FairFace ya viene como crop facial 224x224. El detector de InsightFace espera imagen natural y, con umbral bajo, detecta varias cajas pequeñas dentro del crop. Por eso el pipeline FIFA/Transfermarkt no se debe reutilizar sin ajustes.

Para combinar FairFace con nuestros datos conviene:

1. usar FairFace como dataset etiquetado para auditar sesgos de embeddings/agrupamientos;
2. seleccionar siempre rostro dominante por área;
3. guardar `det_score`, `n_faces`, bbox y razón de área como QC;
4. probar una alternativa que alinee desde crop completo si la detección falla;
5. no mezclar métricas FairFace con FIFA sin marcar explícitamente el dominio de imagen.

## Siguiente prueba útil

Tomar una muestra estratificada por `race/gender/age`, extraer embeddings y medir:

- distribución de cosenos intra-grupo FairFace;
- separación por etiqueta demográfica;
- vecinos kNN cruzados por etiqueta;
- si los clusters FIFA se correlacionan con las etiquetas FairFace al proyectar ambos dominios.
