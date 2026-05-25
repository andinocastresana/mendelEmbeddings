# Bloque regional #4/#5/#7

## Objetivo

Primer tramo posterior al contrato canonico de regiones:

- #4: features geometricas Nivel A sobre landmarks.
- #5: validar embeddings por region Nivel B con sanity KinFaceW.
- #7: cache versionada para no mezclar regiones, modos y modelos.

## Implementacion

### #4 Features geometricas Nivel A

Modulo: `src/phyloface/regions/geometric_features.py`.

Expone:

- `region_geometry(landmarks, image_shape)`: bbox, centroide, ancho, alto y area
  por region canonica.
- `face_geometric_features(landmarks, image_shape)`: distancias/proporciones,
  angulo del eje ocular y simetrias izquierda/derecha, normalizadas por distancia
  interocular.
- `pair_geometric_differences(...)`: diferencias absolutas feature-a-feature entre
  dos caras.

Es logica pura sobre landmarks; no toca pixeles ni embeddings.

### #5 Embeddings regionales Nivel B

Modulo: `src/phyloface/regions/regional_embeddings.py`.

Expone:

- `REGIONAL_EMBEDDINGS_VERSION = "regions-v2.0+arcface-crop-v0.1"`.
- `extract_region_embeddings(...)`: re-aplica ArcFace a `crop_masked_rgb` o
  `crop_rgb` por region.
- `compare_region_embeddings(...)`: coseno por region.
- `region_embeddings_to_arrays(...)`: serializacion compacta para cache `.npz`.

Script: `scripts/validate_region_embeddings_kinfacew.py`.

Advertencia metodologica: ArcFace fue entrenado para rostros completos alineados,
no para parches regionales. Este camino se trata como sanity de senal, no como
modelo validado.

### #7 Cache versionada por regiones

`src/phyloface/core/cache.py` ahora acepta campos opcionales en
`make_config_dict`:

- `regions_version`
- `region_extraction_mode`
- `region_embedding_model`

Estos campos entran en el hash de `config_id` y tambien en un sufijo legible:

`...__regions-v2.0__masked__w600k_r50__<hash>`

`save_image_cache` tambien puede persistir arrays opcionales:

- `region_names`
- `region_embeddings`
- `region_bboxes`
- `region_mask_fill`
- `region_valid`

El cache viejo sigue siendo compatible porque esos campos son opcionales.

## Verificacion

Smoke puro:

```bash
/home/diego/miniconda3/envs/face-sim/bin/python tests/smoke/test_regions_level_a_and_cache.py
```

Resultado:

```text
[OK] regions level A features + cache regional contract
```

Sanity KinFaceW-I limitado para #5:

```bash
TEST_LOG_FILE=_meta/TAREA5_region_embeddings_sanity_resources.log \
MPLCONFIGDIR=/tmp/mpl-codex \
./scripts/test-monitored.sh /home/diego/miniconda3/envs/face-sim/bin/python \
  scripts/validate_region_embeddings_kinfacew.py \
  --dataset KinFaceW-I \
  --limit 12 \
  --cool-threshold 88 \
  --out /tmp/KinFaceW-I_region_embeddings_sanity_limit12.json
```

Resumen del sanity:

| Region | Acc | AUC |
|---|---:|---:|
| left_eyebrow | 0.433 | 0.510 |
| right_eyebrow | 0.615 | 0.632 |
| left_eye | 0.492 | 0.500 |
| right_eye | 0.492 | 0.500 |
| left_cheekbone | 0.538 | 0.583 |
| right_cheekbone | 0.683 | 0.674 |
| left_cheek | 0.467 | 0.467 |
| right_cheek | 0.573 | 0.612 |
| nose | 0.633 | 0.632 |
| mouth | 0.412 | 0.529 |
| chin | 0.523 | 0.603 |
| forehead | 0.520 | 0.649 |

Fallos de imagen: 4. Recursos: `cpu_avg=40%`, `cpu_max=62%`,
`temp_avg=80C`, `temp_max=96C`.

## Decision

#4 y #7 quedan cerradas: hay API estable y smoke.

#5 queda en progreso: el pipeline regional funciona y hay sanity KinFaceW-I
limitado, pero no alcanza para declarar calidad regional. Proxima corrida:
usar limite mayor con parametros termicos mas conservadores (`--limit 40` o 80,
`TEMP_THRESHOLD=80`, y pausas mas largas), o redisenar batching/pausas por imagen
antes de intentar KinFaceW-I completo.
