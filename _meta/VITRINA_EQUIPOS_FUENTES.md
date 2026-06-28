# Vitrina equipos — fuentes de fotos y plan de ingesta

Fecha: 2026-05-26

## Alcance actualizado

La vitrina objetivo es el **Mundial Mexico-USA-Canada 2026**. Las formaciones
viejas, jugadores historicos o selecciones que no hayan clasificado siguen siendo
utiles como material auxiliar para comparar estilos, probar cobertura de fotos y
tener datos mientras se estabilizan los planteles 2026.

Al 2026-05-26, FIFA ya esta publicando anuncios de convocatorias 2026, pero las
listas son **provisionales** hasta el cierre final de asociaciones (2026-06-01) y
la publicacion FIFA esperada del 2026-06-02. Por eso el manifiesto debe guardar
`squad_status` y fuente/fecha de generacion.

## Estado local observado

- `data/input/img/teams/` tiene 4 fotos grupales: Argentina, Francia, España y
  Argelia.
- El motor offline ya puede detectar hasta 11 caras por imagen, cachear embeddings y
  generar heatmaps N x M.
- La vista web especifica de equipos todavia no existe. El snapshot de arquitectura
  preveia un JSON estatico generado offline (`teams_embeddings.json`) y una ruta
  `/teams`, pero esos artefactos no estan implementados.
- `seleccion-de-argelia.jpg` no sirve como dato Qatar 2022 ni necesariamente 2026,
  pero puede quedar como fixture historico/no-clasificado para probar el pipeline.

## Fuentes candidatas

### 1. FIFA oficial

Uso recomendado: roster canonico y datos deportivos 2026.

- Pros: fuente primaria para planteles, dorsales, posiciones, clubes y stats.
- Contras: fotos y endpoints web pueden estar sujetos a terminos mas restrictivos;
  no conviene descargar ni redistribuir imagenes sin revisar condiciones.
- Fuente 2026 util: pagina FIFA de anuncios de squads.
- Fuente historica util: PDF oficial Qatar 2022 `SquadLists-English.pdf`.

### 2. Wikimedia Commons / Wikidata

Uso recomendado: primera fuente para imagenes trazables.

- Pros: cada imagen trae URL, autor, licencia y pagina de descripcion; permite
  filtrar por licencia y mantener atribucion.
- Contras: cobertura irregular; no todos los jugadores tienen foto buena, frontal o
  de la epoca del mundial. Requiere control de calidad facial.
- Estrategia: tomar roster desde Wikipedia/FIFA, resolver jugador en Wikidata, leer
  propiedad `P18` (imagen), consultar Commons `imageinfo.extmetadata`, generar
  manifiesto auditable antes de descargar.

### 3. Kaggle / datasets empaquetados

Uso recomendado: fallback exploratorio local, no primera fuente publica.

- Pros: puede tener fotos ya reunidas para muchos jugadores.
- Contras: procedencia/licencia suele ser opaca; puede servir para prototipar, pero
  es debil para una vitrina publica si no trae atribucion verificable.

## Giro de criterio 2026-05-27: primero estandarizacion visual

Para la etapa de comparacion facial, se priorizan las fotos mas estandarizadas
aunque todavia no sean publicables. El criterio es: retrato frontal o semi-frontal,
una sola persona, cabeza completa, encuadre repetible entre jugadores, fondo simple,
resolucion suficiente y poca expresion/pose extrema.

Esto separa dos decisiones:

- **Dataset de trabajo / comparacion**: puede usar fuentes no resueltas legalmente,
  siempre marcadas como `UNREVIEWED_NONPUBLIC_RESEARCH` y sin publicarlas.
- **Dataset publicable / vitrina externa**: solo imagenes con licencia/permiso y
  atribucion resueltos.

Prioridad tecnica actual:

1. **Transfermarkt profile portraits**: fuente candidata #1 para comparar por
   estandarizacion. Transfermarkt documenta criterios de perfil cercanos a lo que
   necesitamos (cabeza visible, resolucion minima, sin logos tapando, sin otras
   personas). Nuevo extractor:

   ```bash
   python scripts/build_transfermarkt_headshot_manifest.py \
     --input data/output/teams/manifest_wikimedia_northamerica2026_all_max8_downloaded.json \
     --output data/output/teams/manifest_transfermarkt_northamerica2026_headshots.json \
     --download-images
   ```

   Todo queda marcado como no publicable hasta revisar permisos/licencia.

2. **FIFA / federaciones / clubes**: si aparece roster final con headshots
   oficiales, pueden superar a Transfermarkt en consistencia por torneo/equipo.
   Por ahora se tratan como fuente candidata a investigar, no como base automatica.

3. **Wikimedia/Commons/Wikipedia page images**: buena para trazabilidad/publicacion,
   pero visualmente irregular; queda como fallback publicable y fuente de cobertura.

## Decision inicial

Arrancar por 2026 y Wikimedia/Wikidata, construyendo primero un manifiesto JSON con:

- torneo, seleccion, jugador, dorsal, posicion, club;
- entidad Wikidata y pagina Wikipedia si se encuentra;
- archivo Commons, URL de imagen, pagina de descripcion;
- licencia, autor, credito y restricciones reportadas por Commons;
- estado de cobertura (`image_found`, `needs_review`, `missing_image`).

Solo despues de revisar cobertura y calidad se descarga una copia local en `data/`
(gitignored) y se generan embeddings/JSON estatico para el cliente.

## Proximo flujo tecnico

1. Generar manifiesto 2026 para selecciones piloto:

   ```bash
   python scripts/build_teams_photo_manifest.py --teams Argentina France Spain Mexico "United States" Canada --max-per-team 5
   ```

2. Generar manifiesto historico/fallback 2022 si hace falta comparar cobertura:

   ```bash
   python scripts/build_teams_photo_manifest.py --tournament qatar2022 --teams Argentina France Spain --max-per-team 5
   ```

3. Revisar cobertura y licencias del JSON en `data/output/teams/`.
4. Descargar thumbs/crops solo para registros aceptables:

   ```bash
   python scripts/build_teams_photo_manifest.py --teams Argentina France Spain Mexico "United States" Canada --download-images
   ```

5. Pasar control de calidad: una cara detectable, bbox suficiente, pose razonable,
   score de deteccion alto.
6. Generar `client/public/teams/*.json` particionado por seleccion para la vitrina.

## Riesgos a controlar

- No mezclar fotos grupales con headshots individuales sin marcar el origen.
- No asumir que `P18` es foto de 2026; puede ser vieja, de club o de otro torneo.
- No tratar las listas 2026 como finales antes del 2026-06-02.
- No publicar imagenes sin preservar atribucion/licencia.
- Evitar datasets con licencias ambiguas como fuente principal.
