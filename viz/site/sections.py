"""Section content for the editorial site.

Each section is a dict. Three kinds:
  - "hero": the opening
  - "part":  divider page introducing a major part of the work
  - "figure": a section with a figure embed

Body paragraphs are written in a deliberately direct, non-AI Spanish.
No em dashes (—). Short sentences. Concrete numbers. First-person
when natural. The figures' captions go in the metadata block below
each one in monospace.
"""

from __future__ import annotations


SECTIONS: list[dict] = [
    # ── PART 1: el sujeto ────────────────────────────────────────────────
    {
        "kind": "part",
        "id": "parte-1",
        "num": "Parte 01",
        "title": "El sujeto",
        "intro": [
            "El objeto de estudio es BERT-base. 12 capas. 144 cabezas de "
            "atención. 36 864 neuronas en los bloques FFN. 109 millones "
            "de parámetros.",
            "Sobre eso, un fine-tuning en GoEmotions: 23 emociones con "
            "etiquetado multi-label. Cada frase puede llevar varias a la "
            "vez.",
            "Antes de operar nada, primero hay que conocer al sujeto.",
        ],
    },
    {
        "kind": "figure",
        "id": "arquitectura",
        "chapter": "§ 01.1 · Arquitectura",
        "title": "BERT entero, en 3D",
        "subtitle": "12 capas apiladas, cada una con 12 cabezas de atención y un bloque FFN.",
        "body": [
            "12 capas. Cada capa tiene 12 cabezas (las esferas) y un "
            "bloque FFN (el anillo turquesa). Por el centro pasa el "
            "residual stream, la columna por la que viaja el [CLS] "
            "desde abajo hasta el classifier de arriba.",
            "El color de cada esfera es su categoría funcional según el "
            "ablation study del notebook 6. Roja: Critical Specialist. "
            "Azul: Critical Generalist. Amarilla: Minor. Gris: "
            "Dispensable. El tamaño codifica importancia agregada.",
            "Mira la capa 11. Casi todo rojo y azul. Mira la capa 0. "
            "Casi todo gris. Es la primera pista de que el trabajo "
            "emocional no está distribuido uniformemente.",
        ],
        "figure": "bert_architecture",
        "caption": ("Esferas: cabezas de atención. Anillos: bloques FFN. "
                    "Columna central punteada: residual stream. Diamante "
                    "rojo abajo: token [CLS]. Cuadrado verde arriba: "
                    "classifier de 23 emociones. Datos: head_categories.csv."),
        "fig_id": "01.1",
    },
    {
        "kind": "figure",
        "id": "lex2sem",
        "chapter": "§ 01.2 · Base teórica",
        "title": "De diccionario a clasificador",
        "subtitle": "Cómo BERT olvida palabras y aprende emociones.",
        "body": [
            "Cuando una palabra entra en BERT, al salir ya no es esa "
            "palabra. Lo es a la entrada, claro. Pero el modelo sustituye "
            "\"qué palabra es\" por \"qué papel juega\" capa a capa. Es "
            "un fenómeno conocido (Tenney 2019, Ethayarajh 2019).",
            "Tres curvas miden la transición desde tres ángulos. La "
            "azul: retención léxica, cuánto del embedding original queda "
            "en cada capa. La amarilla: anisotropía, cuán parecidos son "
            "los tokens entre sí (al final convergen todos al mismo "
            "vector). La roja: F1 del probe lineal de emoción, cuánta "
            "info de la etiqueta es extraíble.",
            "Las tres se cruzan en L8-L9. Ese es el punto de bisagra. "
            "Antes, fase léxica. Después, fase semántica.",
            "Las cuatro mini-matrices de abajo lo hacen visceral. Cada "
            "celda es la similitud coseno del token a su embedding "
            "inicial. Filas tempranas: colores variados, cada token "
            "preserva identidad. Filas tardías: monocromas, todos los "
            "tokens han colapsado al mismo vector contextual.",
        ],
        "figure": "lexical_to_semantic",
        "caption": ("Tres curvas sobre 46 frases del test set. Heatmaps "
                    "de cuatro frases ejemplo: cos(hidden[L,t], "
                    "hidden[0,t]). Bibliografía: Tenney et al. ACL 2019; "
                    "Ethayarajh EMNLP 2019."),
        "fig_id": "01.2",
    },

    # ── PART 2: compresión ───────────────────────────────────────────────
    {
        "kind": "part",
        "id": "parte-2",
        "num": "Parte 02",
        "title": "El cuerpo en pedazos",
        "intro": [
            "El plan: aplicar SVD a las matrices de pesos del modelo, "
            "quedarte con los k mayores valores singulares, y ver qué "
            "tanto cae el F1.",
            "La sorpresa: el modelo ya está comprimiendo por dentro. "
            "Los tokens al final viven en un subespacio de baja "
            "dimensión. La SVD no introduce compresión donde no la "
            "había. Materializa la que ya existe.",
        ],
    },
    {
        "kind": "figure",
        "id": "internal-compression",
        "chapter": "§ 02.1 · El puente",
        "title": "El modelo se comprime <em>a sí mismo</em>",
        "subtitle": "Antes de aplicar SVD, la red ya está reduciendo dimensión.",
        "body": [
            "Tomamos las representaciones de tokens en cada capa, las "
            "apilamos en una matriz de 469 × 768, y le hacemos SVD. La "
            "pregunta: cuántas dimensiones está usando realmente el "
            "modelo en cada capa.",
            "Arriba, dos cosas. Las curvas azules son la norma "
            "euclídea de los vectores. Crecen con la profundidad "
            "(Kobayashi 2021). Las rojas son rango efectivo y k95: "
            "cuántos valores singulares cubren el 95% de la varianza. "
            "Caen brutalmente. De ~130 dimensiones en las capas "
            "tempranas a 22 en L12. El k95 baja a 35.",
            "De 768 dimensiones posibles, el modelo termina usando un 5%.",
            "El heatmap de abajo es la prueba completa. Para cada capa, "
            "qué fracción de varianza cubren los k mejores valores. La "
            "línea negra es el k95 capa por capa. Esa frontera se "
            "desplaza a la izquierda en L9-L12.",
        ],
        "pull": ("Si la representación interna en L12 vive en 22 de 768 "
                 "dimensiones, las matrices que la producen son "
                 "low-rank-aproximables. La SVD no introduce compresión "
                 "donde no la había. Materializa la que ya existe."),
        "figure": "internal_compression",
        "caption": ("Norma media de hidden states (azul) vs rango "
                    "efectivo y k95 (terra). Heatmap de energía espectral "
                    "acumulada. Datos: 46 frases del split de test sobre "
                    "23emo-final. Refs: Kobayashi EMNLP 2021; Dong et al. "
                    "ICML 2021."),
        "fig_id": "02.1",
    },
    {
        "kind": "figure",
        "id": "pareto",
        "chapter": "§ 02.2 · El acantilado",
        "title": "22 estrategias, una transición de fase",
        "subtitle": "Aplicas SVD uniforme a todas las capas. Entre rango 384 y 256, el F1 se desploma.",
        "body": [
            "El gráfico de la derecha es una superficie F1 sobre el "
            "plano (rango × profundidad). Cada celda es una "
            "configuración de compresión. Color = retención de F1.",
            "Las capas tardías (8-11) caen al vacío antes que las "
            "tempranas. Esa asimetría es lo que motiva la compresión "
            "informada del capítulo 6.",
            "El algoritmo greedy domina la frontera de Pareto en 8 de "
            "9 puntos óptimos. La estrategia greedy_90 retiene un 93% "
            "del F1 con un 14% menos de parámetros.",
        ],
        "figure": "pareto_3d",
        "caption": ("Frontera de Pareto: 22 estrategias evaluadas. "
                    "Datos: compression_comparison.csv. Eje rango "
                    "uniforme, eje profundidad de capa, color F1 macro."),
        "fig_id": "02.2",
    },
    {
        "kind": "figure",
        "id": "spectral-landscape",
        "chapter": "§ 02.3 · 3D",
        "title": "La asimetría espectral, hecha topografía",
        "subtitle": "Las 72 matrices del modelo apiladas como filas de un terreno.",
        "body": [
            "Cada fila del relieve es una matriz de pesos del modelo. "
            "Eje X: índice del valor singular. Eje Z: magnitud "
            "normalizada σᵢ/σ₁.",
            "Lo que ves: las matrices Q y K forman picos abruptos. "
            "Pocos valores singulares dominan, espectro concentrado, "
            "rango efectivo bajo, comprimibles. Las FFN forman mesetas "
            "casi planas. Espectro distribuido, cada dimensión aporta, "
            "frágiles bajo SVD.",
            "Los diamantes marcan k95 por matriz. Q/K cerca de 395. FFN "
            "cerca de 620. Es la Tabla 6 de la memoria, pero en "
            "relieve.",
        ],
        "figure": "spectral_landscape",
        "caption": ("SVD computada sobre el checkpoint 23emo-final. "
                    "72 matrices = 12 capas × 6 componentes (Q, K, V, "
                    "Attn-O, FFN-i, FFN-o)."),
        "fig_id": "02.3",
    },
    {
        "kind": "figure",
        "id": "spectral-flowers",
        "chapter": "§ 02.4 · Iconográfico",
        "title": "72 flores espectrales",
        "subtitle": "Una matriz, una flor. Forma del pétalo = compresibilidad.",
        "body": [
            "Cada flor es una matriz. Cada pétalo es un valor singular "
            "σᵢ, normalizado a σ₁. 32 pétalos por flor.",
            "Las columnas Q y K son flores con 2-3 pétalos enormes que "
            "dominan, el resto diminutos. Eso es espectro concentrado. "
            "Rango efectivo bajo. Buenas para comprimir.",
            "Las columnas FFN son flores rellenas, casi circulares. "
            "Todos los pétalos comparables. Espectro plano. Cada "
            "dimensión aporta. Frágiles bajo SVD.",
            "La forma de la flor predice cuánto aguanta cada matriz.",
        ],
        "figure": "spectral_flowers",
        "caption": ("Una flor por cada una de las 72 matrices del "
                    "encoder. Pasa el ratón por encima para ver el k95 "
                    "exacto."),
        "fig_id": "02.4",
    },
    {
        "kind": "figure",
        "id": "decay",
        "chapter": "§ 02.5 · Animación",
        "title": "La galaxia se deshace",
        "subtitle": "Misma proyección que galaxy formation, ahora con la SVD activa.",
        "body": [
            "Las mismas 588 frases en L12, proyectadas con LDA. Las "
            "vemos en seis configuraciones de compresión: rangos 768 "
            "(sin tocar), 512, 384, 256, 128, 64.",
            "A r=512 todo casi igual. Entre r=384 y r=256 está el "
            "acantilado. Los clusters se difuminan. A r=128 la "
            "geometría desaparece. A r=64 los embeddings colapsan a un "
            "blob.",
            "La gráfica de la derecha cuantifica: silhouette en azul, "
            "retención de F1 en terra. Las dos curvas caen juntas a "
            "partir de r=384. La transición de fase, vista como "
            "geometría que se deshace.",
        ],
        "figure": "compression_decay",
        "caption": ("588 frases del test set, L12, LDA-3D fija. "
                    "SVD aplicada uniformemente. Slider o Play."),
        "fig_id": "02.5",
    },
    {
        "kind": "figure",
        "id": "ft-diff",
        "chapter": "§ 02.6 · Pre vs post",
        "title": "Qué cambió el fine-tuning",
        "subtitle": "Diff Frobenius entre bert-base-uncased y 23emo-final.",
        "body": [
            "Cargo los dos modelos. Para cada una de las 72 matrices, "
            "calculo el cambio relativo: ‖W_ft − W_pre‖ / ‖W_pre‖. "
            "Cuánto se ha movido cada matriz durante el fine-tuning.",
            "La predicción de §5.5 era simple. El gradiente fluye más "
            "fuerte hacia las capas finales. Las tardías deberían "
            "cambiar mucho. Las tempranas, ya buenas en el "
            "pre-entrenamiento, deberían quedarse casi igual.",
            "El heatmap lo confirma. Los valores en capas 8-11 son "
            "claramente más altos. Es evidencia empírica directa de "
            "la arquitectura de dos fases.",
        ],
        "figure": "finetuning_diff",
        "caption": ("Norma Frobenius del cambio relativo, 72 matrices "
                    "del encoder. bert-base-uncased vs 23emo-final."),
        "fig_id": "02.6",
    },

    # ── PART 3: localizando emociones ───────────────────────────────────
    {
        "kind": "part",
        "id": "parte-3",
        "num": "Parte 03",
        "title": "Dónde viven las emociones",
        "intro": [
            "Si las capas tardías son las que más cambian con el "
            "fine-tuning, y las que más rango efectivo pierden, también "
            "deberían ser donde se decide qué emoción detectar.",
            "Probing por capa, geometría 3D, logit lens. Tres maneras "
            "distintas de decir lo mismo.",
        ],
    },
    {
        "kind": "figure",
        "id": "crystallization",
        "chapter": "§ 03.1 · Probing",
        "title": "Cristalización por capas",
        "subtitle": "Cuándo aparece cada emoción dentro del modelo.",
        "body": [
            "Para cada capa, entrenas un clasificador lineal sobre el "
            "[CLS]. Te dice cuánta información de cada emoción es "
            "linealmente extraíble en ese punto.",
            "Gratitude sale ya en L0. Tiene vocabulario obvio "
            "(\"thanks\", \"thank you\"). Realization aguanta hasta L11. "
            "Necesita contexto entero.",
            "La frecuencia en el dataset NO predice la profundidad. "
            "Annoyance tiene 3× más ejemplos que disgust. Cristaliza 6 "
            "capas más tarde. Lo que importa es la complejidad "
            "semántica, no el volumen de datos.",
            "Los diamantes marcan la capa de cristalización: donde el "
            "F1 alcanza el 80% de su máximo. El ribbon de la izquierda "
            "es el cluster psicológico de cada emoción.",
        ],
        "figure": "crystallization",
        "caption": ("Probing lineal por capa, 23 emociones × 13 capas. "
                    "Datos: probe_results.csv del notebook 4."),
        "fig_id": "03.1",
    },
    {
        "kind": "figure",
        "id": "galaxy",
        "chapter": "§ 03.2 · Geometría",
        "title": "Galaxy formation",
        "subtitle": "23 emociones cristalizando en el espacio LDA, capa por capa.",
        "body": [
            "Tomamos el [CLS] de cada capa. Le aplicamos el pooler "
            "real del modelo (Linear + tanh). Proyectamos con LDA "
            "supervisada, ajustada en L12. Los tres ejes son las "
            "direcciones que mejor separan las 23 emociones.",
            "En L0 los 23 diamantes (los centroides de cada emoción) "
            "están casi superpuestos en el origen. En L11 ocupan "
            "regiones distintas, cerca de las frases que les "
            "corresponden.",
            "Cuantitativo: separation ratio de 4.3 (los centroides están "
            "4× más separados que la dispersión interna). 40% de "
            "accuracy por nearest-centroid contra 4% de random.",
            "Click en la leyenda para aislar una emoción.",
        ],
        "figure": "galaxy_formation",
        "caption": ("2300 frases del test set. Pooler aplicado, LDA-3D "
                    "fija ajustada en L12. Mismas coordenadas para todas "
                    "las capas."),
        "fig_id": "03.2",
    },
    {
        "kind": "figure",
        "id": "tokens",
        "chapter": "§ 03.3 · Geometría · extendida",
        "title": "Token trajectories",
        "subtitle": "No sólo el [CLS]. Todos los tokens viajando por el residual stream.",
        "body": [
            "La galaxia anterior muestra sólo el viaje del [CLS]. "
            "Pero cada token tiene su propia trayectoria a través de "
            "las 13 capas. Aquí los proyecto todos a la misma LDA-3D.",
            "El [CLS] (rojo, diamante grande) viaja muy lejos del "
            "origen, hacia el centroide de la emoción gold. Es el "
            "agregador en acción.",
            "Los tokens de contenido (azules) y las palabras función "
            "(grises) apenas se mueven. El residual stream les "
            "preserva la información local mientras el [CLS] integra el "
            "significado global.",
        ],
        "figure": "token_trajectories",
        "caption": ("Hidden states completos para 46 frases. "
                    "Cada token proyectado capa a capa con la misma "
                    "LDA-3D que galaxy formation."),
        "fig_id": "03.3",
    },
    {
        "kind": "figure",
        "id": "iterative",
        "chapter": "§ 03.4 · Logit lens",
        "title": "La curva en U",
        "subtitle": "Aplicar el classifier real a cada capa, no sólo a la última.",
        "body": [
            "Logit lens. La técnica es de Nostalgebraist (2020) y la "
            "refinan Belrose et al. (NeurIPS 2023). Aplicas el pooler "
            "+ classifier reales del modelo a cada una de las 13 "
            "capas, no sólo a L12.",
            "Promediado sobre 2300 frases, el sigmoid medio traza una "
            "U. Capas 0-3: las estadísticas del [CLS] saturan el tanh "
            "del pooler. Muchas emociones disparan a la vez con "
            "magnitud media. Capas 4-9: el [CLS] está en transición. "
            "El tanh se queda cerca de cero. El modelo no decide nada. "
            "Capas 10-11: el [CLS] llega al régimen donde el "
            "classifier fue entrenado. Una emoción pega un salto.",
            "Eso explica por qué el activation patching de L11 "
            "recupera el 100% del F1. Restauras exactamente esta "
            "calibración.",
        ],
        "figure": "iterative_inference",
        "caption": ("Sigmoid promedio del pooler+classifier aplicado por "
                    "capa. Top-1, gold, suma de las 23. 2300 frases."),
        "fig_id": "03.4",
    },
    {
        "kind": "figure",
        "id": "lens-vs-probe",
        "chapter": "§ 03.5 · Comparación",
        "title": "Lo que sabe vs lo que <em>sabe leer</em>",
        "subtitle": "Probing y logit lens contestan preguntas distintas. La diferencia es informativa.",
        "body": [
            "Dos curvas, tres paneles. Las dos curvas miden información "
            "de emoción en cada capa. Las dos usan los mismos datos. "
            "Pero contestan a preguntas distintas.",
            "El probe lineal (rojo) es un clasificador nuevo entrenado "
            "sobre cada capa. Te dice cuánta info hay ahí, linealmente "
            "extraíble. Sube monótono. La información se acumula con la "
            "profundidad.",
            "El logit lens (verde) es la cabeza del modelo entrenada en "
            "L11, aplicada a cada capa. Te dice qué predeciría ese "
            "classifier si lo atornillas a una capa intermedia. Hace U "
            "porque solo sabe leer activaciones estilo L11. En las "
            "capas tempranas y medias lee mal, aunque la información "
            "esté ahí.",
            "La banda gris entre las dos curvas es la diferencia. Info "
            "que <strong>existe pero el modelo no usa</strong>. Eso "
            "explica por qué basta con restaurar L11 en el activation "
            "patching para recuperar el 100% del F1: no es que falte "
            "información en el resto del modelo, es que la cabeza solo "
            "sabe leerla en una capa concreta.",
            "Los tres paneles separan las emociones por su capa de "
            "cristalización. En tempranas (panel 1), el probe ya está "
            "alto en L0 pero el lens aún no decide: brecha máxima. En "
            "tardías (panel 3), el probe sube tarde y el lens también, "
            "casi juntos: brecha mínima.",
        ],
        "figure": "lens_vs_probe",
        "caption": ("Probe F1 macro de notebook 4 contra logit lens "
                    "(pooler+classifier reales aplicados al [CLS] de "
                    "cada capa) sobre 2300 frases del test set. Bandas "
                    "de fase compartidas con la curva U."),
        "fig_id": "03.5",
    },
    {
        "kind": "figure",
        "id": "fingerprint",
        "chapter": "§ 03.6 · Multi-label",
        "title": "Decision fingerprint",
        "subtitle": "Las métricas agregadas esconden la firma multi-label de cada frase.",
        "body": [
            "Multi-label significa que el modelo no produce una "
            "emoción. Produce un vector de 23 sigmoides, varias "
            "activas a la vez. Ese vector es la firma de la frase.",
            "Aplico el classifier real al [CLS] de cada capa. Cada "
            "pétalo del polar es una emoción. Su longitud es el "
            "sigmoid. El pétalo con borde negro es la emoción gold.",
            "Selecciona una frase y dale a Play. Vas viendo el "
            "fingerprint emerger desde el ruido (Emb) hasta la "
            "decisión cristalizada (L11). Cuando varios pétalos "
            "crecen juntos: el modelo cree que coexisten varias "
            "emociones. Eso es lo que la BCE multi-label produce y "
            "ningún heatmap promediado deja ver.",
        ],
        "figure": "decision_fingerprint",
        "caption": ("Sigmoid de las 23 emociones por capa, vector "
                    "polar. Aplicado al pooler+classifier real."),
        "fig_id": "03.6",
    },

    # ── PART 4: atención ────────────────────────────────────────────────
    {
        "kind": "part",
        "id": "parte-4",
        "num": "Parte 04",
        "title": "144 cabezas",
        "intro": [
            "En cada capa hay 12 cabezas de atención. 12 × 12 = 144 en "
            "total. Cada una tiene su propio patrón de atención y su "
            "propia función.",
            "No todas son iguales. No todas son críticas. No todas "
            "hacen lo mismo.",
        ],
    },
    {
        "kind": "figure",
        "id": "heads",
        "chapter": "§ 04.1 · Categorías",
        "title": "Las 144 cabezas",
        "subtitle": "Critical Specialist, Critical Generalist, Minor, Dispensable.",
        "body": [
            "Cada cabeza es una celda en una matriz 12×12. Color = "
            "categoría según el ablation study. La capa 11 (fila "
            "inferior) no contiene ni una cabeza prescindible. Sus 12 "
            "son críticas.",
            "Las estrellas blancas marcan las cabezas que cada "
            "emoción necesita específicamente, según la Tabla 19. "
            "L11-H6 aparece dos veces. La comparten sadness y "
            "realization.",
            "El 77% de las cabezas en capas 8-11 son críticas. En las "
            "capas 0-4 es sólo el 25%. Hay un gradiente que no "
            "aparece en arquitecturas no fine-tuneadas.",
        ],
        "figure": "heads_matrix",
        "caption": ("12 capas × 12 cabezas = 144 celdas. Datos: "
                    "head_categories.csv del notebook 6."),
        "fig_id": "04.1",
    },
    {
        "kind": "figure",
        "id": "atlas",
        "chapter": "§ 04.2 · Patrones",
        "title": "Attention atlas",
        "subtitle": "Las 144 cabezas, todas a la vez, sobre una frase.",
        "body": [
            "Eliges una frase. El sistema te enseña los 144 patrones "
            "de atención simultáneamente. Bordes coloreados por la "
            "categoría funcional de cada cabeza.",
            "Patrones reconocibles. Las capas tempranas atienden "
            "diagonalmente: token a sí mismo, vecinos. Las tardías "
            "concentran atención sobre [CLS] o [SEP], rayas "
            "verticales. Es el patrón típico de las cabezas "
            "agregadoras.",
            "Click en cualquier cabeza para ampliar con etiquetas de "
            "tokens.",
        ],
        "figure": "attention_atlas",
        "caption": ("Pesos de atención reales sobre 46 frases del test "
                    "set. 12 capas × 12 cabezas = 144 mini-mapas."),
        "fig_id": "04.2",
    },
    {
        "kind": "figure",
        "id": "constellations",
        "chapter": "§ 04.3 · Superposition",
        "title": "Las 23 emociones como direcciones",
        "subtitle": "Los vectores de peso del classifier, proyectados a 3D.",
        "body": [
            "El classifier tiene 23 vectores de 768 dimensiones, uno "
            "por emoción. Esos vectores son las direcciones que el "
            "modelo usa para detectar cada emoción. Aquí los proyecto "
            "a PCA-3D y los pinto como flechas radiando del origen.",
            "Vectores casi-ortogonales: emociones que el modelo "
            "distingue limpiamente. Vectores casi-paralelos: emociones "
            "que el modelo confunde. Gratitude y love apuntan parecido.",
            "El heatmap de la derecha es la similitud coseno completa "
            "en 768-d, reordenada por cluster. Bloques diagonales: "
            "alta similitud dentro de un cluster.",
            "Esto es §2.7.5 (superposition) hecho geometría. El modelo "
            "codifica más conceptos que dimensiones tendría si fueran "
            "ortogonales puras.",
        ],
        "figure": "probe_constellations",
        "caption": ("23 vectores del classifier proyectados con PCA "
                    "fija. Heatmap de cosine similarity en 768 "
                    "dimensiones, reordenado por cluster."),
        "fig_id": "04.3",
    },
    {
        "kind": "figure",
        "id": "circuit",
        "chapter": "§ 04.4 · Circuitos",
        "title": "El circuito compartido",
        "subtitle": "Cuando dos emociones reutilizan la misma maquinaria.",
        "body": [
            "Aquí el grafo se construye en D3.js. Sin Plotly. Cada "
            "emoción a la izquierda. Cabezas críticas en el centro. "
            "Clusters psicológicos a la derecha.",
            "L11-H6 es el nodo rojo. Es la única cabeza compartida "
            "por más de una emoción. La usan sadness y realization. "
            "Probable circuito de \"expectativa frustrada\" que el "
            "fine-tuning reutiliza.",
            "Click en cualquier nodo para ver su entorno.",
        ],
        "figure": "circuit_network",
        "caption": ("Grafo construido a partir de top_heads_per_emotion."
                    "csv. Nodos en rojo: cabezas compartidas."),
        "fig_id": "04.4",
    },

    # ── PART 5: causalidad y neuronas ───────────────────────────────────
    {
        "kind": "part",
        "id": "parte-5",
        "num": "Parte 05",
        "title": "Causalidad",
        "intro": [
            "Hasta aquí: correlación. Saber que el F1 sube en una capa "
            "no demuestra que esa capa cause la decisión. Para eso, "
            "lesionar.",
            "Apagar una capa. Apagar una neurona. Ver qué se rompe.",
        ],
    },
    {
        "kind": "figure",
        "id": "lesion",
        "chapter": "§ 05.1 · Lesion",
        "title": "Lesion theater",
        "subtitle": "Restaurar capa por capa, ver el modelo revivir.",
        "body": [
            "Empezamos con un modelo a F1 = 0. Todas las barras a "
            "cero. Cada etapa restaura los pesos originales de UNA "
            "capa más. 12 etapas en total.",
            "Las primeras 8 etapas no mueven nada. L8 enciende un "
            "destello. L9 empuja arriba a las emociones léxicas. L10 "
            "recupera la mitad. L11 hace explotar todas las barras al "
            "baseline simultáneamente.",
            "La capacidad emocional del modelo no está distribuida. "
            "Vive concentrada en las capas finales.",
        ],
        "figure": "lesion_theater",
        "caption": ("Activation patching secuencial por capa. F1 macro "
                    "y por emoción. 12 etapas + estado inicial."),
        "fig_id": "05.1",
    },
    {
        "kind": "figure",
        "id": "neurons",
        "chapter": "§ 05.2 · Neuronas",
        "title": "Las 80 neuronas más selectivas",
        "subtitle": "Sus 5 frases más activantes, una al lado de otra.",
        "body": [
            "El notebook 7 documenta 36 864 neuronas FFN con scores "
            "de selectividad Cohen's d. Pero los números no cuentan la "
            "historia entera. Aquí ves QUÉ significa cada neurona.",
            "Cada tarjeta es una neurona: capa, índice, emoción "
            "dominante, selectividad Cohen's d, y las 5 frases que "
            "más la activan.",
            "La top-1 de admiration (L11 N944) se enciende con frases "
            "tipo elogio. Eso es el significado que el modelo asignó "
            "a esa neurona. Es lo más cerca que se llega a hablar el "
            "idioma de BERT.",
            "Filtra por emoción, capa o dirección (excitatoria vs "
            "inhibitoria).",
        ],
        "figure": "neuron_gallery",
        "caption": ("80 neuronas con activaciones reales sobre 2300 "
                    "frases. Selectividad calculada como Cohen's d "
                    "(diferencia de medias normalizada por desviación "
                    "estándar)."),
        "fig_id": "05.2",
    },
    {
        "kind": "figure",
        "id": "conf-evolution",
        "chapter": "§ 05.3 · Confusión",
        "title": "Confusion matrix evolution",
        "subtitle": "23 × 23 confusiones a través de las 13 capas.",
        "body": [
            "Aplicamos el logit lens en bloque. Para cada par (gold, "
            "predicho), calculo el sigmoid medio. Animado por capa.",
            "L0: mush uniforme, todo emborronado. L7: el hueco del "
            "valle, casi negro. L11: diagonal limpia.",
            "Lo interesante son las celdas off-diagonal que sobreviven "
            "en L11. Pares que el modelo confunde incluso al final. "
            "Annoyance ↔ disapproval. Fear ↔ sadness. Approval ↔ "
            "realization. Justamente vecinos en los seis clusters "
            "emocionales. El modelo no los separa porque "
            "psicológicamente están cerca.",
        ],
        "figure": "confusion_evolution",
        "caption": ("23 emociones × 23 emociones × 13 capas. Sigmoid "
                    "medio por par sobre 2300 frases."),
        "fig_id": "05.3",
    },
    {
        "kind": "figure",
        "id": "conf-volume",
        "chapter": "§ 05.4 · 3D",
        "title": "Confusion volume",
        "subtitle": "Las 13 confusion matrices apiladas como cubo.",
        "body": [
            "La animación anterior era 2D. Esta es la versión 3D "
            "estática y rotable. Las 13 matrices 23 × 23 apiladas "
            "verticalmente. Sólo las celdas con sigmoid > 0.15 que "
            "aparezcan en al menos 7 de 13 capas.",
            "Las celdas que persisten verticalmente son confusiones "
            "estructurales. Pares que NUNCA se separan completamente.",
            "Rota el cubo. Verás que algunos pares (annoyance ↔ "
            "disapproval, fear ↔ sadness) tienen columnas verticales "
            "tan altas como la diagonal. El modelo confunde estos "
            "pares en TODAS las capas.",
        ],
        "figure": "confusion_volume",
        "caption": ("Sigmoid medio filtrado a > 0.15 con persistencia "
                    "≥ 7/13 capas. Las celdas verticales muestran "
                    "confusiones estructurales."),
        "fig_id": "05.4",
    },

    # ── PART 6: síntesis ─────────────────────────────────────────────────
    {
        "kind": "part",
        "id": "parte-6",
        "num": "Parte 06",
        "title": "Síntesis",
        "intro": [
            "Una taxonomía emocional emergente, una compresión "
            "informada por interpretabilidad, y un dashboard que "
            "junta todo.",
        ],
    },
    {
        "kind": "figure",
        "id": "clusters",
        "chapter": "§ 06.1 · Taxonomía",
        "title": "Seis clusters que aparecen solos",
        "subtitle": "El modelo redescubre la psicología sin que se la impongan.",
        "body": [
            "Sin pedirle nada, un clustering jerárquico sobre los "
            "vectores de selectividad neuronal produce seis grupos "
            "con coherencia psicológica reconocible. Positivas "
            "energéticas. Negativas reactivas. Internas. Epistémicas. "
            "Orientadas al otro. Baja especificidad.",
            "La barra de la derecha mide la norma de selectividad. "
            "Es el mejor predictor (ρ = 0.64, p = 0.001) de la caída "
            "de F1 bajo SVD. Las emociones \"escritas en negrita\" en "
            "los pesos del modelo son las más vulnerables.",
            "La SVD no ataca selectivamente las neuronas emocionales. "
            "La geometría espectral es ortogonal a la función. Pero "
            "las emociones que requieren más capacidad neuronal son "
            "más frágiles a CUALQUIER perturbación.",
        ],
        "figure": "sunburst",
        "caption": ("Sunburst con 6 clusters, 23 emociones. Áreas "
                    "proporcionales a frecuencia en train. "
                    "Bibliografía: Russell 1980 (circumplex)."),
        "fig_id": "06.1",
    },
    {
        "kind": "figure",
        "id": "landscape",
        "chapter": "§ 06.2 · Mapa",
        "title": "El paisaje emocional",
        "subtitle": "Cada emoción en (cristalización × intensidad).",
        "body": [
            "Las 23 emociones, posicionadas en un plano 2D. Eje X: "
            "capa de cristalización. Eje Y: norma de selectividad.",
            "Cuadrante superior izquierdo: gratitude, love. Tempranas "
            "e intensas. Cuadrante inferior derecho: realization, "
            "disappointment. Tardías y difusas.",
            "La línea punteada conecta sadness y realization. Las dos "
            "comparten L11-H6.",
            "Selecciona una emoción en el menú de la derecha y verás "
            "su huella radial: 6 dimensiones funcionales en una "
            "figura polar.",
        ],
        "figure": "emotional_landscape",
        "caption": ("23 emociones sobre el plano (cristalización × "
                    "norma de selectividad). Datos: "
                    "crystallization_layers.csv y neuron_catalog.csv."),
        "fig_id": "06.2",
    },
    {
        "kind": "figure",
        "id": "trajectory",
        "chapter": "§ 06.3 · Síntesis",
        "title": "Una frase, cuatro vistas",
        "subtitle": "El experimento que se pide al inicio: ver BERT pensar.",
        "body": [
            "Aquí se juntan todas las viz en una sola experiencia. "
            "Eliges una frase. Arrastras el slider de capa. Los "
            "cuatro paneles se mueven sincronizados. Trayectoria 3D "
            "del [CLS]. Atención de las 3 cabezas más críticas. "
            "Sigmoides multi-label. Curva del gold.",
            "Pulsa Play. El [CLS] arranca cerca del origen y se "
            "desplaza hacia su centroide. Las cabezas críticas se "
            "activan capa a capa. Los pétalos saltan del valle a la "
            "cristalización. La curva del gold traza la U que ya "
            "conoces.",
        ],
        "figure": "sentence_trajectory",
        "caption": ("Cuatro paneles síncronos. Datos reales del "
                    "modelo 23emo-final aplicado en vivo a la frase "
                    "elegida."),
        "fig_id": "06.3",
    },
    {
        "kind": "figure",
        "id": "greedy",
        "chapter": "§ 06.4 · Algoritmo",
        "title": "Greedy en acción",
        "subtitle": "Cómo el algoritmo construye la compresión paso a paso.",
        "body": [
            "El greedy elige movimientos por eficiencia: parámetros "
            "ahorrados / coste F1. Aquí lo ves en acción. Empezamos "
            "con baseline (todo a rango 768) y avanzamos: greedy_95 "
            "→ greedy_90 → … → greedy_50.",
            "La matriz 12 × 6 se va iluminando célula a célula. Las "
            "primeras decisiones son Q y K. Gratis, sin coste F1. "
            "Eso es exactamente lo que predice §4.3 sobre la "
            "inmunidad de Q/K. Después vienen FFN-output en capas "
            "tempranas. Las capas tardías (8-11) se mantienen "
            "intactas hasta el final.",
            "La línea derecha sigue el F1 vs ratio de compresión "
            "paso a paso. Es la prueba algorítmica de que el greedy "
            "reproduce los hallazgos de interpretabilidad sin acceso "
            "a ellos. Se entera sólo con datos empíricos de "
            "sensibilidad.",
        ],
        "figure": "greedy_replay",
        "caption": ("Replay del algoritmo greedy_50, _60, …, _95. "
                    "Datos: greedy_*_ranks.csv del notebook 9."),
        "fig_id": "06.4",
    },
    {
        "kind": "figure",
        "id": "sandbox",
        "chapter": "§ 06.5 · Sandbox",
        "title": "Diseña tu propia compresión",
        "subtitle": "18 sliders, una cuestión: ¿puedes vencer al greedy?",
        "body": [
            "18 sliders. 6 componentes (Q, K, V, AttnOut, FFN-int, "
            "FFN-out) × 3 bandas de profundidad (early, middle, "
            "late). Cada slider es un rango SVD de 32 a 768.",
            "El simulador estima F1 macro y posición Pareto en "
            "tiempo real. Damage model fitado a las Tablas 9 y 10. "
            "Los presets te dan puntos de partida (uniform_r256, "
            "greedy_90).",
            "Reto: ¿puedes vencer al algoritmo greedy que retiene "
            "93% del F1 con 14% menos de parámetros?",
        ],
        "figure": "compression_sandbox",
        "caption": ("Damage model interpolado a partir de la "
                    "sensibilidad medida en notebook 3."),
        "fig_id": "06.5",
    },
    {
        "kind": "figure",
        "id": "cards",
        "chapter": "§ 06.6 · Dashboard",
        "title": "Trading cards",
        "subtitle": "23 emociones, 23 tarjetas, una por una.",
        "body": [
            "La síntesis ejecutiva. Cada emoción reducida a una "
            "tarjeta. Perfil completo: F1 baseline, F1 tras "
            "compresión, recuperación tras fine-tuning, capa de "
            "cristalización, cabeza crítica, neuronas significativas, "
            "selectividad neuronal, radar de 6 dimensiones "
            "funcionales.",
            "Las tarjetas se reordenan: por cluster (default), por "
            "F1, por capa de cristalización, por número de neuronas, "
            "por mejora con fine-tuning, alfabético. Cada vista "
            "ofrece una historia distinta.",
            "Es el dossier que abres cuando el tribunal pregunta "
            "sobre cualquier emoción concreta.",
        ],
        "figure": "emotion_cards",
        "caption": ("23 tarjetas con datos cruzados de los notebooks "
                    "3, 4, 6, 7, 8 y 9."),
        "fig_id": "06.6",
    },
]


# ── Hero stats ────────────────────────────────────────────────────────────
HERO_STATS = [
    ("12 × 144",  "capas × cabezas"),
    ("36 864",    "neuronas FFN"),
    ("23",        "emociones, multi-label"),
    ("27",        "visualizaciones"),
]


# ── Figure heights (px) — author-declared, no runtime measurement ─────────
# These are the heights the figure is designed to render at. The iframe
# wrap is sized to height + 64 (2× 32 padding around the iframe).
FIG_HEIGHTS = {
    "bert_architecture":     880,
    "lexical_to_semantic":   940,
    "internal_compression":  940,
    "lens_vs_probe":         960,
    "pareto_3d":             620,
    "spectral_landscape":    880,
    "spectral_flowers":     1240,
    "compression_decay":     820,
    "finetuning_diff":       920,
    "crystallization":       780,
    "galaxy_formation":      880,
    "token_trajectories":    900,
    "iterative_inference":   980,
    "decision_fingerprint":  780,
    "heads_matrix":          740,
    "attention_atlas":      1100,
    "probe_constellations":  800,
    "circuit_network":       760,
    "lesion_theater":        900,
    "neuron_gallery":       1100,
    "confusion_evolution":   920,
    "confusion_volume":      880,
    "sunburst":              680,
    "emotional_landscape":   720,
    "sentence_trajectory":   980,
    "greedy_replay":         820,
    "compression_sandbox":   880,
    "emotion_cards":        1300,
}
