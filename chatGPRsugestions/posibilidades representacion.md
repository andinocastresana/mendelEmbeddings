Tengo un problema de visualización y agrupamiento. Tengo entidades discretas formadas por elementos no homogéneos, de tal forma que sería posible formar grupos dentro de cada entidad. A su vez, estos grupos suelen parecerse más a grupos de otras entidades que a los de su propia entidad. He probado técnicas como umap, tsne, etc pero los elementos se dispersan por toda la superficie. Hay algún algoritmo que contemple esta estructura de datos, que me permita conservar estructuras locales y globales al mismo tiempo?

Sí. De hecho, el problema que describes es uno de los casos donde **UMAP y t-SNE suelen fallar conceptualmente**, no porque estén mal implementados, sino porque la hipótesis de partida del algoritmo no coincide con la estructura de tus datos.

Por cómo lo describes, tienes algo parecido a una estructura **jerárquica de dos niveles**:

* **Entidad** (A, B, C...)  
  * Grupo interno 1  
  * Grupo interno 2  
  * Grupo interno 3

pero los grupos obedecen a una organización global como

Grupo tipo X:  
A1 \----- B2 \----- C3

Grupo tipo Y:  
A2 \----- B1 \----- C1

Grupo tipo Z:  
A3 \----- B3 \----- C2

es decir, la similitud viene dada por los **subgrupos**, no por la pertenencia a la entidad.

Esto hace que un algoritmo basado únicamente en distancias entre puntos tienda a "romper" las entidades.

---

## **El problema fundamental**

UMAP y t-SNE trabajan con un grafo de vecinos entre **observaciones individuales**.

No saben que existe una relación adicional:

"estos puntos pertenecen a la misma entidad"

Para ellos

Entidad A  
  a1  
  a2  
  a3

es exactamente igual que

tres observaciones independientes

Si

a1 ≈ b2  
a2 ≈ c1  
a3 ≈ d5

entonces inevitablemente dispersarán la entidad A por todo el espacio.

Y en realidad están haciendo lo correcto según su función objetivo.

---

# **Lo que necesitas es preservar dos grafos**

En realidad tienes dos relaciones distintas.

### **Grafo 1**

Similitud entre elementos

a1 \---- b2  
|  
c3

Este es el grafo clásico.

---

### **Grafo 2**

Pertenencia a entidad

a1  
|  
a2  
|  
a3

Es otro grafo completamente distinto.

El problema es que UMAP sólo usa el primero.

---

# **Algunas alternativas**

## **1\. PaCMAP**

Probablemente sería lo primero que probaría.

PaCMAP fue diseñado precisamente para intentar conservar simultáneamente:

* estructura local  
* estructura intermedia  
* estructura global

Muchos conjuntos donde UMAP rompe grupos funcionan bastante mejor con PaCMAP.

---

## **2\. TriMAP**

TriMAP intenta preservar relaciones de largo alcance mediante tríos de puntos.

En problemas donde existen varios "clusters dentro de clusters" suele mantener mucho mejor la estructura global.

---

## **3\. PHATE**

Muy interesante para datos con estructura jerárquica.

PHATE utiliza difusión sobre el grafo y suele preservar:

* continuidad  
* ramificaciones  
* jerarquías

Se usa bastante en datos de célula única precisamente porque aparecen este tipo de mezclas.

---

## **4\. HSNE**

Si los datos son grandes, HSNE construye una representación jerárquica.

No fuerza toda la información en una sola proyección.

---

## **5\. Diffusion Maps**

Si la estructura es realmente una variedad con subgrupos, Diffusion Maps suele capturarla mejor que UMAP.

---

# **Si conoces las entidades**

Aquí viene algo interesante.

Si sabes que

a1  
a2  
a3

pertenecen a la entidad A,

entonces puedes construir un **grafo multiplex**.

Por ejemplo

peso \=

0.8 \* similitud

\+

0.2 \* pertenencia a entidad

o incluso

W \= αW\_sim \+ βW\_entity

y ejecutar el embedding sobre ese grafo.

UMAP permite pasar un grafo precomputado, por lo que puedes modificar la conectividad antes de optimizar.

Esta aproximación suele funcionar sorprendentemente bien cuando la pertenencia a entidad aporta información que la distancia entre puntos no refleja.

---

# **Otra idea: embedding bipartito**

En lugar de representar únicamente elementos, representar simultáneamente

Entidad  
     \\  
      \\  
Elemento  
      /  
Grupo latente

Esto es un grafo heterogéneo.

Hay algoritmos específicos para esto:

* Node2Vec  
* metapath2vec  
* GraphSAGE  
* GAT  
* embeddings sobre grafos heterogéneos

Estos métodos mantienen simultáneamente:

* similitud  
* pertenencia  
* relaciones de segundo orden

y luego proyectas el embedding final con PCA o UMAP.

---

# **Si el objetivo es visualizar**

Existe otro enfoque que pocas veces se considera.

En lugar de proyectar directamente los elementos, construir primero un **árbol jerárquico** mediante clustering (por ejemplo, clustering aglomerativo) y luego representar el árbol con un algoritmo de distribución de grafos como ForceAtlas2 o Kamada–Kawai.

De esta forma puedes visualizar tanto:

Entidad

├── grupo 1  
│      ├── elementos  
│  
├── grupo 2  
│  
└── grupo 3

como las conexiones entre grupos de distintas entidades mediante aristas adicionales, obteniendo una representación mucho más fiel de la organización jerárquica que una proyección 2D de puntos.

---

## **Mi impresión**

Por cómo describes el problema, no creo que te falte "el algoritmo mágico" de reducción de dimensionalidad. El problema parece ser que tus datos tienen una **estructura relacional de múltiples niveles** (elemento → grupo → entidad) que un embedding clásico no modela.

Yo exploraría este orden:

1. **PaCMAP**, por su buen equilibrio entre estructura local y global.  
2. **PHATE**, si sospechas una organización jerárquica o continua.  
3. Construir un **grafo combinado** que incorpore tanto similitud entre elementos como pertenencia a entidad y hacer el embedding sobre ese grafo.  
4. Si la relación entidad–elemento es realmente importante, pasar a un **embedding de grafos heterogéneos** en lugar de un algoritmo clásico de reducción de dimensionalidad.

Una pregunta que ayudaría a afinar la recomendación: ¿cada elemento pertenece **a una única entidad**, o puede pertenecer a varias? Y, además, ¿los "grupos" internos están ya identificados o son precisamente lo que intentas descubrir?

Cada elemento pertenece a una entidad y los grupos internos debería definirlos con algún algoritmo que aún no he definido

Ese detalle cambia bastante el enfoque. En realidad, tu problema ya no es únicamente de visualización, sino de **clustering con estructura conocida parcialmente**.

Formalmente tienes:

* Un conjunto de elementos xix\_ixi​.  
* Cada elemento pertenece a una entidad EjE\_jEj​ (esto lo conoces).  
* Dentro de cada entidad existen varios grupos latentes (desconocidos).  
* Los grupos "equivalentes" entre distintas entidades son más similares entre sí que a otros grupos de la misma entidad.

Es decir, la entidad es una variable de contexto, no la estructura que quieres descubrir.

## **Lo que probablemente está ocurriendo**

Imagina tres entidades:

Entidad A  
   A1  
   A2  
   A3

Entidad B  
   B1  
   B2  
   B3

Entidad C  
   C1  
   C2  
   C3

Pero en realidad la organización es

Grupo α  
A1  B2  C3

Grupo β  
A2  B1  C1

Grupo γ  
A3  B3  C2

En ese caso, pedir a UMAP que mantenga juntas las entidades es imposible, porque geométricamente no están juntas.

## **Yo no empezaría por UMAP**

Creo que el orden debería ser justo el contrario:

Datos  
    ↓

Clustering

    ↓

Definir grupos

    ↓

Visualizar

No al revés.

---

# **Un enfoque interesante: clustering con restricciones**

Como conoces la entidad de cada elemento, puedes introducir una restricción muy útil.

Por ejemplo, podrías penalizar que un cluster contenga demasiados elementos de la misma entidad.

En otras palabras:

Cluster 1

✓ A1  
✓ B3  
✓ C2

✗ A2  
✗ A3

Porque esperas que cada grupo represente un "tipo funcional" presente en múltiples entidades, no una entidad concreta.

Esto pertenece a la familia de los **Constrained Clustering** o **Semi-supervised Clustering**, donde se incorporan restricciones como "must-link" o "cannot-link". En tu caso, una restricción de "como mucho un elemento por entidad en cada clúster" (o una penalización suave equivalente) parece muy natural.

## **También podrías pensar en un modelo generativo**

Lo que describes recuerda mucho a un modelo de mezcla jerárquico.

Cada elemento se genera por

Entidad  
     ↓  
Distribución de grupos  
     ↓  
Grupo latente  
     ↓  
Observación

Eso es precisamente el tipo de estructura que modelan técnicas como los modelos bayesianos jerárquicos o mezclas jerárquicas, que permiten que distintas entidades compartan componentes latentes.

## **Otra idea que me parece especialmente prometedora**

Si cada entidad tiene un número razonable de elementos (por ejemplo, entre 20 y 500), podrías tratar cada entidad como un **conjunto** en lugar de una colección de puntos independientes.

Existen arquitecturas de aprendizaje para este tipo de datos, como los modelos tipo **Deep Sets** o redes sobre conjuntos, que aprenden representaciones respetando la estructura "elementos dentro de entidades". Una vez obtenidas esas representaciones, el clustering suele ser mucho más estable que sobre los datos originales.

## **Una posibilidad muy potente: co-clustering**

Hay algo en tu descripción que me llama la atención.

Quizá no quieras descubrir únicamente grupos de elementos.

Quizá quieras descubrir simultáneamente:

* tipos de elementos,  
* tipos de entidades.

Eso es un problema clásico de **co-clustering** o **biclustering**, donde se agrupan simultáneamente filas y columnas de una matriz de relaciones. Si tus datos pueden representarse como una matriz "entidad × características" o "entidad × elementos", este enfoque puede revelar patrones que un clustering convencional pasa por alto.

## **Una pregunta importante**

Hay un aspecto que todavía no está claro y que determina qué algoritmos son más adecuados:

**¿Cada entidad contiene aproximadamente el mismo número de elementos?**

Por ejemplo,

Entidad A → 150 elementos  
Entidad B → 143 elementos  
Entidad C → 160 elementos

o más bien

Entidad A → 8 elementos  
Entidad B → 250 elementos  
Entidad C → 37 elementos

Y una segunda pregunta aún más importante:

**¿Esperas que cada entidad contenga aproximadamente un representante de cada grupo latente?** Es decir, algo como:

Entidad A → α β γ δ  
Entidad B → α β γ δ  
Entidad C → α β γ δ

aunque no necesariamente en el mismo número o con exactamente las mismas características.

Si la respuesta es sí, entonces el problema deja de ser un clustering convencional y se parece mucho más a un **emparejamiento entre entidades** con componentes compartidos. Ese tipo de estructura permite diseñar algoritmos mucho más específicos y potentes que aplicar UMAP seguido de K-means.

La entidades tienen el mismo número de elementos. Y hay entidades que serán homogéneas (un único cluster) y otras que tendrán un número variado de clústers (e incluso elementos aislados). Mi idea es, ver si puedo de alguna forma definir la entidades más heterogéneas como originadas por aportes de las entidades más homogéneas (aunque a veces este intercambio podría ser bidireccional)

Ahora creo que el problema se entiende mucho mejor, y diría que **ya no lo formularía como un problema de clustering**, sino como un problema de **descomposición de mezclas**.

Lo que describes me recuerda más a problemas de:

* deconvolución de poblaciones,  
* factorización matricial,  
* modelos de mezcla,  
* o incluso transferencia evolutiva (en biología ocurre algo muy parecido).

La idea sería:

Una entidad homogénea representa un "estado puro". Una entidad heterogénea sería una combinación de varios estados puros.

Eso es bastante distinto de buscar simplemente clusters.

## **Una forma de verlo**

Supongamos que tienes 100 entidades de 200 elementos cada una.

Algunas son

Entidad A  
AAAAAAAAAAAAAAAAAAAA

Entidad B  
BBBBBBBBBBBBBBBBBBBB

Otras son

Entidad C  
AAAAAAABBBBBCCCCDDD

Lo que quieres descubrir es si realmente

C ≈ 40% A  
  35% B  
  15% D  
  10% ruido

Eso ya es un problema de mezcla.

---

## **Yo empezaría por construir un grafo**

No trabajaría directamente sobre los puntos.

Construiría un grafo donde

* cada nodo es un elemento  
* el peso es la similitud

Después aplicaría detección de comunidades.

¿Por qué?

Porque los algoritmos de comunidades (Louvain, Leiden, Infomap...) permiten:

* comunidades grandes  
* comunidades pequeñas  
* nodos aislados

sin tener que fijar un número de clusters.

Y además funcionan muy bien cuando existen relaciones transversales entre entidades.

---

## **Después pasaría al nivel de entidad**

Una vez identificadas las comunidades, cada entidad deja de ser un conjunto de puntos y pasa a ser un **vector de composición**.

Por ejemplo

| Entidad | Com1 | Com2 | Com3 | Com4 |
| ----- | ----- | ----- | ----- | ----- |
| A | 198 | 2 | 0 | 0 |
| B | 0 | 200 | 0 | 0 |
| C | 70 | 90 | 40 | 0 |
| D | 0 | 30 | 170 | 0 |

Ahora sí aparecen claramente las mezclas.

---

## **Esto recuerda muchísimo a topic modeling**

De hecho, matemáticamente es casi el mismo problema.

Si haces la analogía

* documento → entidad  
* palabra → elemento  
* tema → grupo latente

entonces una entidad es

60 % tema A

30 % tema B

10 % tema C

Y eso es exactamente lo que hacen modelos como **LDA** (cuando los elementos son discretos) o enfoques de factorización cuando las características son continuas.

---

## **O incluso mejor: Non-negative Matrix Factorization**

Si construyes una matriz

              características

Entidad A

Entidad B

Entidad C

la NMF busca

X ≈ W·H

donde

* H son los componentes "puros"  
* W indica cuánto aporta cada componente a cada entidad

Y esos componentes suelen ser mucho más interpretables que los componentes principales de un PCA.

---

## **Hay algo que me llama especialmente la atención**

Has dicho:

el intercambio puede ser bidireccional.

Eso sugiere que no existe una dirección temporal.

Entonces yo evitaría modelos filogenéticos o árboles.

Pensaría más bien en una **red de intercambio**.

Algo como

A ─────── C  
│        ╱  
│      ╱  
│    ╱  
B────D

donde el peso de la arista representa cuánto comparten.

Esa representación suele ser mucho más realista que intentar forzar una jerarquía.

---

# **Una idea que creo que puede ser muy potente**

Como todas las entidades tienen exactamente el mismo número de elementos, tienes una ventaja enorme que normalmente no existe.

Puedes calcular una **distancia de transporte óptimo** entre entidades.

No compararías elemento a elemento, sino distribuciones completas.

En concreto miraría **Optimal Transport** y, especialmente, las distancias de **Wasserstein**.

¿Por qué?

Porque permiten responder preguntas del tipo:

¿Cuál es el coste mínimo de transformar la distribución de la entidad A en la de la entidad C?

Y aún mejor, existen extensiones como **Gromov-Wasserstein** y **Fused Gromov-Wasserstein** que comparan conjuntos cuando no existe una correspondencia uno a uno entre sus elementos, preservando tanto las características individuales como la estructura interna de cada entidad.

Creo que encaja sorprendentemente bien con tu problema, porque:

* no necesitas que los clusters estén definidos de antemano;  
* las entidades pueden tener uno o varios grupos;  
* admite elementos "huérfanos" o poco representados;  
* las mezclas aparecen de forma natural como distribuciones intermedias entre estados más puros.

## **Mi intuición**

Si estuviera investigando este problema, probablemente no dedicaría más tiempo a UMAP o t-SNE. Los usaría únicamente al final para ilustrar resultados.

Mi flujo de trabajo sería algo como:

1. Construir un grafo de similitud entre elementos.  
2. Detectar comunidades (preferiblemente con Leiden, por su estabilidad y capacidad para manejar comunidades de distintos tamaños).  
3. Representar cada entidad como un vector de abundancias de comunidades.  
4. Descomponer esos vectores mediante NMF o un modelo de mezclas para identificar los "componentes puros".  
5. Construir una red entre entidades basada en esas proporciones de mezcla y validar si las entidades homogéneas actúan efectivamente como fuentes de las heterogéneas.

La razón por la que me inclino por este enfoque es que responde directamente a la hipótesis científica que quieres poner a prueba: **no solo agrupa elementos, sino que estima de qué "componentes" está formada cada entidad y en qué proporción**, que es exactamente la noción de "aporte" que planteas. Creo que esa formulación está mucho más alineada con tu objetivo que intentar encontrar una proyección bidimensional que conserve simultáneamente todas las relaciones.

