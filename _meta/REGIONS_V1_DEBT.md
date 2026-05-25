# Regions v1 - deuda historica

Fecha: 2026-05-25

## Resumen

En el proyecto no queda una API versionada llamada literalmente `regions_v1`.
La deuda historica es conceptual: antes de la migracion de la Tarea #1, la
segmentacion regional vivia dentro del notebook/archivo experimental como bloques
sueltos de funciones y constantes. No habia un contrato canonico independiente
para nombres, landmarks, poligonos, version de regiones o semantica de cache.

La migracion dejo ese estado operativo como `regions_v2`:

- `extract_regions_v2`: bbox rectangular + `crop_rgb`.
- `extract_regions_v2_masked`: bbox + mascara + `crop_masked_rgb`.
- `selected_pair["regions_v2"]`: clave historica compartida por ambos paths.

## Que fue "regions v1"

Para efectos del proyecto, "regions v1" se refiere al periodo experimental previo
a `regions_v2`, no a un modulo estable:

- regiones definidas de forma local en el notebook o en
  `src/phyloface_experimental_functions.py`;
- nombres y orden repetidos a mano en extractores, comparadores y visualizadores;
- listas manuales de landmarks sin registry central;
- sin version explicita para cachear o comparar resultados;
- sin separacion clara entre contrato anatomico, estrategia de bbox, estrategia
  de mascara y metrica regional;
- metricas regionales basadas en crops de gris/z-score/coseno, utiles como
  baseline visual pero no equivalentes a embeddings faciales regionales.

## Por que se descarto como base

`regions v1` no era una superficie confiable para construir features nuevas:

- cualquier cambio de nombre o landmarks podia romper visualizaciones/cache sin
  quedar documentado;
- la diferencia entre regiones oficiales de MediaPipe y aproximaciones manuales
  no estaba codificada como contrato;
- frente y menton tenian logica especial, pero esa excepcion no estaba disponible
  para consumidores externos;
- no habia forma robusta de versionar futuros cambios (`v2`, `v2-masked`,
  `v3-periocular`, etc.);
- la comparacion regional historica mezclaba "region visual" con "evidencia de
  parentesco", cuando aun falta validar embeddings regionales reales.

## Decision actual

La Tarea #2 formaliza el contrato en `src/phyloface/regions/canonical.py`.
Ese modulo define `CANONICAL_REGIONS_VERSION = "regions-v2.0"` y una lista
estable de 12 regiones. Por ahora no cambia la extraccion; documenta y congela el
comportamiento vigente para que las tareas siguientes puedan migrar hacia el
registry sin romper compatibilidad.

## Episodios rescatados

La nota de migracion (`_meta/MIGRACION_TAREA1.md`) menciona dos episodios en el
KG externo (`IA/memories/_meta/episodes/2026-05-20-*`). No los encontre en las
rutas locales disponibles durante esta sesion. Se rescata lo versionado:

- la bitacora de migracion de Tarea #1;
- los comentarios de cabecera de `geometry.py`, `extract_rect.py` y
  `extract_masked.py`;
- el archivo archivado `_toReview/phyloface_experimental_functions_20260520_110102.py`.

Si aparecen los episodios externos, se deben contrastar contra este documento y
agregar solo decisiones no duplicadas.
