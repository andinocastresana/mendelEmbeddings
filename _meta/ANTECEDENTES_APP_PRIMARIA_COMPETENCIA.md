# Antecedentes — App primaria de semejanza familiar

Fecha: 2026-05-27

Objetivo de esta nota: guardar el barrido rapido de productos/lineas similares a
la App primaria ("hijo/a vs madre/padre") para decidir si vale la pena seguir
invirtiendo tiempo en producto, diferenciacion y validacion.

## Resumen ejecutivo

Si existe competencia directa. Hay apps y webs que hacen el flujo basico:
subir foto del hijo/a y de ambos padres, calcular porcentajes de semejanza y
declarar si se parece mas a madre o padre.

La oportunidad no esta en que la idea sea inedita, sino en diferenciarse por:

- procesamiento local / privacidad real;
- resultado explicable por regiones faciales;
- reporte descargable;
- tono serio/exploratorio en vez de viral/gimmick;
- metodologia documentada, calibracion y disclaimers claros.

## Competidores directos

### DNAI — Family Face Match AI

URL: https://www.dnaiapp.com/

Posicionamiento: app iOS de entretenimiento para responder "who did the kid
take after?". El flujo declarado es subir hijo/a + ambos padres, obtener scores
de semejanza, ganador y explicacion.

Senales relevantes:

- "Upload a child + both parents."
- porcentajes "Mom vs Dad";
- breakdown por rasgos: ojos, menton, nariz, "vibe";
- foco fuerte en compartir resultados y resolver discusiones familiares;
- declara "For entertainment purposes only";
- dice no entrenar con fotos salvo opt-in.

Lectura: es el competidor conceptual mas cercano por producto. Su tono parece
mas viral/comico que metodologico.

### LooksCompare — AI Family Face Similarity Test

URL: https://lookscompare.com/

Posicionamiento: web + app Android. Tiene modo "Baby-Mom-Dad (Family test)".
Permite subir fotos de child, mom y dad, y devuelve similitudes.

Senales relevantes:

- web publica para cargar tres fotos;
- calcula child-mom y child-dad similarity percentages;
- declara no guardar fotos a largo plazo;
- incluye modo normal/strict;
- disclaimer explicito: resultado indicativo, no medico/forense/legal.

Lectura: competidor directo web. Parece menos rico visualmente, pero valida que
hay demanda para una version browser-first.

### Face Match & Baby Similarity

URL: https://play.google.com/store/apps/details?hl=en&id=com.softupper.facesim

Posicionamiento: app Android. Promete responder si el bebe se parece mas a padre
o madre y analizar cada feature facial.

Senales relevantes:

- 10K+ descargas al momento del relevamiento;
- rating bajo/moderado (2.6 con ~50 reviews visibles);
- declara deteccion automatica de caras;
- devuelve score de semejanza contra madre y padre;
- promete analisis por facciones;
- usa lenguaje cuestionable: "A quick genetics test in a second with AI".

Lectura: valida demanda, pero tambien muestra una oportunidad de hacer algo mas
cuidadoso en lenguaje y UX.

### YouOrMe / You or me App

URLs:

- https://huntscreens.com/en/products/youorme
- https://www.toolify.ai/tool/youorme

Posicionamiento: web/app simple para determinar a que padre se parece mas un
hijo/a mediante IA.

Lectura: otro antecedente directo, aparentemente mas pequeno. Sirve como senal
de que el nicho esta poblado por herramientas simples.

### Baby Like Dad or Mom?

URL: https://apps.apple.com/ie/app/baby-like-dad-or-mom/id1499816664

Posicionamiento: app de entretenimiento con reconocimiento facial para comparar
padres e hijo/a.

Lectura: antecedente historico de la misma categoria; la idea lleva varios anos
en tiendas, aunque no necesariamente con buena implementacion.

## Productos cercanos pero no equivalentes

### Baby generators

Ejemplos:

- Bae: https://www.bae-app.com/
- Fotor baby generator: https://www.fotor.com/features/baby-generator/
- TryBabyFace: https://trybabyface.com/

Estos no comparan un hijo existente contra sus padres. Generan una imagen de un
posible bebe futuro desde fotos de los padres. Son competencia por atencion del
usuario, pero no por el flujo de App primaria.

Lectura: no deberian guiar la arquitectura actual, salvo como referencia de UX
de onboarding y disclaimers ("entretenimiento, no prediccion genetica").

## Antecedente academico / tecnico

### Kinship verification

Ejemplo:

- FIW / Recognizing Families in the Wild:
  https://arxiv.org/abs/2110.07020

La literatura de kinship verification busca determinar si existe una relacion
familiar entre dos imagenes: padre-hijo, madre-hijo, hermanos, abuelos, etc.
Esto es mas serio que las apps de entretenimiento, pero normalmente no produce
una experiencia de producto tipo "a quien se parece mas y por que region".

Lectura: la App primaria ya esta mejor alineada con esta linea que la mayoria de
apps virales porque usa calibracion, datasets KinFaceW y disclaimers.

## Diferenciacion posible para mendelEmbeddings

Angulos donde seguiria teniendo sentido invertir:

- **Privacidad**: procesamiento client-side real, sin subir imagenes por defecto.
- **Explicabilidad regional**: no solo "madre 62% / padre 38%", sino ojos, nariz,
  boca, mandibula, pomulos, etc. con barras/radar/heatmap.
- **Reporte**: PDF descargable local con caras, metricas, regiones y disclaimer.
- **Trazabilidad tecnica**: documentar embeddings, calibracion, posterior
  calibrado, limitaciones y sesgos.
- **Tono de producto**: exploratorio y familiar, evitando promesas de paternidad,
  genetica o certeza legal.
- **Arbol familiar**: integracion con el pedigree ya construido. La competencia
  revisada parece centrarse en una comparacion aislada de tres fotos.

## Riesgos / razones para no seguir

- La idea base no es nueva.
- El mercado esta lleno de apps de entretenimiento y baby generators.
- Si el resultado se reduce a porcentajes simples, queda commoditizado.
- Hay riesgo de mala interpretacion: paternidad, parentesco legal, genetica,
  sesgo por edad/calidad de foto/pose.
- El procesamiento de menores exige una postura de privacidad especialmente
  clara si se agrega sharing via servidor.

## Recomendacion

No descartaria la App primaria solo por competencia. Si se continua, conviene
evitar construir "otra app que dice si se parece a mama o papa" y concentrarse en
la version diferenciada:

1. app local-first;
2. resultado regional interpretable;
3. reporte/export;
4. disclaimers visibles;
5. integracion con arbol familiar;
6. opcion server-side solo despues de definir consentimiento, retencion y que se
   sube exactamente.

La pregunta de producto pendiente no es "existe algo parecido?", porque si
existe. La pregunta util es: "podemos hacer una version suficientemente mas
privada, explicable y seria como para que valga la pena frente a las apps de
entretenimiento?".
