"""Build every visualization HTML and the master scrollytelling page.

Usage:
    .viz_venv/bin/python viz/build_all.py
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from viz.plots import (pareto_3d, heads_matrix, crystallization, sunburst,
                       fingerprints, lesion_theater, spectral_flowers)
from viz.interactive import (compression_sandbox, circuit_network,
                             compression_decay, decision_fingerprint,
                             galaxy_formation, iterative_inference,
                             attention_atlas, confusion_evolution,
                             greedy_replay, sentence_trajectory,
                             probe_constellations, spectral_landscape,
                             confusion_volume, token_trajectories,
                             emotion_cards, finetuning_diff,
                             bert_architecture, neuron_gallery,
                             lexical_to_semantic, internal_compression)


OUT = pathlib.Path(__file__).resolve().parent / "output"


# ─── Sections of the scrollytelling page ─────────────────────────────────────

SECTIONS = [
    {
        "id": "intro",
        "kind": "hero",
        "chapter": "PRÓLOGO",
        "title": "Anatomía emocional de un<br>Transformer",
        "subtitle": "Compresión selectiva e interpretabilidad mecánica de BERT-base sobre GoEmotions",
        "body": [
            ("Esta es la versión interactiva del TFG de <b>Guido Biosca Lasa</b> "
             "(FIB-UPC, 2026). Aquí no leerás texto y mirarás figuras estáticas: "
             "navegarás por las activaciones reales del modelo. Cada visualización "
             "responde a tu scroll, a tu click, a tu input."),
            ("Modelo: <code>bert-base-uncased</code> fine-tuned en GoEmotions "
             "con 28 categorías emocionales. Las activaciones que ves más abajo "
             "son reales — extraídas en el momento sobre 588 frases del split de test."),
        ],
        "html": None,
    },
    {
        "id": "architecture",
        "kind": "fullbleed",
        "chapter": "TU MODELO",
        "title": "BERT-base · arquitectura",
        "subtitle": "El objeto de estudio entero en un solo modelo 3D",
        "body": [
            ("Antes de empezar a abrir BERT en pedazos, aquí lo tienes <b>"
             "entero</b>. 12 capas apiladas, cada una con 12 cabezas de "
             "atención (esferas) + un bloque FFN (anillo turquesa). El "
             "stream residual atraviesa todas las capas como una columna "
             "central — el [CLS] entra por abajo y sale por arriba al "
             "classifier."),
            ("Color de cada cabeza = su categoría funcional según tu notebook 6 "
             "(Critical Specialist roja / Generalist azul / Minor Specialist "
             "amarilla / Dispensable gris). El TAMAÑO de cada esfera es la "
             "importancia agregada de esa cabeza. Verás visualmente: <b>la "
             "capa 11 está llena de esferas grandes rojas y azules</b>; las "
             "capas tempranas tienen esferas pequeñas y grises."),
            ("Rota el modelo. Es la imagen icónica del proyecto — la cosa "
             "que tu memoria está abriendo y comprimiendo a lo largo de 70 "
             "páginas."),
        ],
        "html": "26_bert_architecture.html",
        "iframe_height": 880,
    },
    {
        "id": "lex2sem",
        "kind": "fullbleed",
        "chapter": "CAP. 5 · base teórica",
        "title": "De diccionario a clasificador",
        "subtitle": "Cómo BERT olvida palabras y aprende emociones, capa a capa",
        "body": [
            ("Antes de hablar de cabezas, neuronas o cristalización, hay un "
             "fenómeno que conviene fijar: cuando una palabra entra en BERT, "
             "<b>al salir ya no es esa palabra</b>. Lo es de entrada, pero el "
             "modelo va sustituyendo \"qué palabra es\" por \"qué papel "
             "juega\" capa a capa."),
            ("Las tres curvas miden la misma transición desde tres ángulos: "
             "(1) <b>retención léxica</b> azul — cuánto de la palabra "
             "original queda en su posición; cae monótona desde 1.0. "
             "(2) <b>anisotropía</b> amarilla — cuán parecidos son los tokens "
             "<i>entre sí</i>; estable hasta L8 y luego explota. "
             "(3) <b>emergencia semántica</b> roja — F1 del probe lineal de "
             "emoción. Las tres se cruzan en <b>L8-L9</b> — el punto de "
             "bisagra entre fase léxica y fase semántica de tu modelo."),
            ("Las cuatro mini-matrices de abajo lo hacen visceral. Cada celda "
             "es la similitud coseno del token en esa capa con su embedding "
             "inicial. Las filas tempranas son coloridas (cada token preserva "
             "su identidad). Las tardías son monocromas (todos los tokens han "
             "perdido la palabra original y comparten un mismo \"vector de "
             "contexto\")."),
            ("<b>Origen del fenómeno</b>: Tenney et al., \"BERT Rediscovers "
             "the NLP Pipeline\" (ACL 2019); Ethayarajh, \"How Contextual are "
             "Contextualized Word Representations?\" (EMNLP 2019). Esta viz "
             "es una <b>replicación en tu modelo fine-tuneado</b>, no un "
             "descubrimiento — pero da el marco conceptual que justifica por "
             "qué la cristalización emocional <i>tiene</i> que ocurrir tarde."),
        ],
        "html": "28_lexical_to_semantic.html",
        "iframe_height": 940,
    },
    {
        "id": "internal-compression",
        "kind": "fullbleed",
        "chapter": "CAP. 4 · puente",
        "title": "El modelo se comprime a sí mismo",
        "subtitle": "Antes de que apliquemos SVD, la red ya está reduciendo dimensión",
        "body": [
            ("Aquí está el dato que une los dos hemisferios del TFG "
             "(interpretabilidad y compresión). Tomamos las representaciones "
             "de tokens de contenido en cada capa, las apilamos en una "
             "matriz <code>(N_tokens × 768)</code>, y le hacemos SVD. Esto te "
             "dice <b>cuántas dimensiones está usando realmente</b> el "
             "subespacio de tokens en cada capa."),
            ("El panel superior muestra dos cosas a la vez. Las curvas "
             "azules son la <b>norma euclídea</b> de los vectores: crecen "
             "monótonas (Kobayashi 2021). Las curvas rojas son el "
             "<b>rango efectivo</b> y <b>k95</b> (cuántos valores singulares "
             "explican el 95% de la varianza). Lo asombroso: el rango "
             "efectivo cae de ~130 dimensiones en las capas tempranas a "
             "<b>22</b> en L12. <b>El k95 cae a 35</b>. De 768 dimensiones "
             "posibles, el modelo termina usando ~5%."),
            ("El heatmap inferior es la prueba completa: para cada capa, "
             "qué fracción de la varianza total cubren los k mejores valores "
             "singulares. La línea negra punteada marca el k95 capa por capa. "
             "Verás cómo esa frontera se desplaza brutalmente hacia la "
             "izquierda en L9-L12 — el subespacio se estrecha en la "
             "fase final."),
            ("<b>Por qué importa</b>: si la representación interna en L12 "
             "vive en 22-35 dimensiones, las matrices de pesos que la "
             "producen son matemáticamente low-rank-aproximables. La SVD "
             "del capítulo 4 no introduce compresión donde no la había — "
             "solo <b>materializa la compresión que el modelo ya hace</b>. "
             "Es la motivación geométrica directa de tu §4. Bibliografía: "
             "Kobayashi (EMNLP 2021); Dong et al. (ICML 2021)."),
        ],
        "html": "29_internal_compression.html",
        "iframe_height": 960,
    },
    {
        "id": "spectral",
        "kind": "fullbleed",
        "chapter": "CAP. 4",
        "title": "El acantilado espectral",
        "subtitle": "22 estrategias de compresión, una transición de fase",
        "body": [
            ("Cuando aplicas SVD uniforme a BERT, el F1 cae suavemente entre "
             "rango 768 y 384… y luego se desploma. Entre <b>r=384 (43% F1) "
             "y r=256 (4%)</b> hay un acantilado real."),
            ("El gráfico 3D de la derecha lo hace literal: superficie F1 sobre "
             "el plano (rango × profundidad). Las capas tardías (8-11) caen al "
             "vacío antes que las tempranas. <b>El algoritmo greedy</b> domina "
             "la frontera de Pareto en 8 de 9 puntos óptimos."),
            ("Mueve el ratón sobre el plano para ver la retención exacta en "
             "cada celda. La estrategia <b>greedy_90</b> retiene un 93% del F1 "
             "con un 14% menos de parámetros."),
        ],
        "html": "01_pareto_landscape.html",
        "iframe_height": 600,
    },
    {
        "id": "heads",
        "kind": "fullbleed",
        "chapter": "CAP. 5.3",
        "title": "Las 144 cabezas",
        "subtitle": "Critical Specialist · Generalist · Minor · Dispensable",
        "body": [
            ("Cada cabeza ocupa una celda en una matriz 12×12 (capa × cabeza). "
             "Color = categoría. <b>La capa 11 no contiene NI UNA cabeza "
             "prescindible</b> — sus 12 son críticas."),
            ("Las estrellas blancas marcan las cabezas que <b>cada emoción</b> "
             "necesita específicamente, según Tabla 19. <b>L11-H6</b> aparece "
             "dos veces: la comparten <i>sadness</i> y <i>realization</i>."),
            ("Esto es importante: el 77% de las cabezas en capas 8-11 son "
             "críticas, frente al 25% en capas 0-4. Hay un gradiente monotónico "
             "que no aparece en arquitecturas no fine-tuneadas."),
        ],
        "html": "02_heads_matrix.html",
        "iframe_height": 720,
    },
    {
        "id": "circuit",
        "kind": "scrolly",
        "chapter": "CAP. 5.3+",
        "title": "El circuito compartido",
        "subtitle": "Cuando dos emociones reutilizan la misma maquinaria",
        "body": [
            ("Aquí el grafo se construye desde cero — sin Plotly, en D3.js puro. "
             "Cada emoción a la izquierda. Cabezas críticas en el centro. "
             "Clusters psicológicos a la derecha."),
            ("<b>L11-H6 es el nodo rojo</b>. Es la única cabeza compartida por "
             "más de una emoción: <i>sadness</i> y <i>realization</i> dependen "
             "ambas de ella. Probable circuito de \"expectativa frustrada\" "
             "que el fine-tuning reutiliza."),
            ("Click en cualquier nodo para ver su entorno. El panel derecho "
             "explica la geografía interna de la emoción seleccionada."),
        ],
        "html": "09_circuit_network.html",
        "iframe_height": 740,
    },
    {
        "id": "crystal",
        "kind": "fullbleed",
        "chapter": "CAP. 5.1",
        "title": "Cristalización por capas",
        "subtitle": "Tres regímenes: temprano (léxico), medio, tardío (contextual)",
        "body": [
            ("El probing lineal traza qué emoción puedes leer del [CLS] en "
             "cada capa. <i>Gratitude</i> se detecta ya en L0 (vocabulario "
             "claro: \"thanks\"). <i>Realization</i> tarda hasta L11."),
            ("La frecuencia en el dataset NO predice la profundidad: "
             "<i>annoyance</i> tiene 3× más ejemplos que <i>disgust</i> y aun "
             "así cristaliza 6 capas más tarde. Lo que importa es la "
             "complejidad semántica, no el volumen de datos."),
            ("Los diamantes blancos marcan la capa de cristalización (donde el "
             "F1 alcanza el 80% de su máximo). El ribbon de la izquierda "
             "muestra el cluster psicológico al que pertenece cada emoción."),
        ],
        "html": "03_crystallization.html",
        "iframe_height": 760,
    },
    {
        "id": "lesion-theater",
        "kind": "fullbleed",
        "chapter": "CAP. 5.2 · animado",
        "title": "Lesion theater",
        "subtitle": "Restaurar capa por capa, ver el modelo revivir",
        "body": [
            ("Pulsa <b>▶ Play</b> para ver la secuencia: 12 etapas, una por capa. "
             "Empezamos con un modelo a F1 = 0 (todas las barras a cero). "
             "Cada etapa restaura los pesos originales de UNA capa más."),
            ("La narrativa es brutal: las primeras 8 etapas no mueven nada. "
             "L8 enciende un destello. L9 empuja arriba a las emociones léxicas. "
             "L10 recupera la mitad. <b>L11 hace explotar todas las barras al baseline</b> "
             "simultáneamente. La línea fantasma marca el F1 que estamos persiguiendo."),
            ("Esta animación es la mejor manera de ENTENDER §5.2: la capacidad "
             "emocional del modelo NO está distribuida — vive concentrada en "
             "las capas finales, y particularmente en la capa 11."),
        ],
        "html": "11_lesion_theater.html",
        "iframe_height": 880,
    },
    {
        "id": "clusters",
        "kind": "fullbleed",
        "chapter": "CAP. 5.4",
        "title": "Seis clusters emergentes",
        "subtitle": "El modelo redescubre la psicología sin imponerla",
        "body": [
            ("Sin pedirle nada, el clustering jerárquico sobre vectores de "
             "selectividad neuronal produce <b>seis grupos</b> con coherencia "
             "psicológica reconocible. Positivas energéticas. Negativas "
             "reactivas. Internas. Epistémicas. Orientadas al otro. Baja "
             "especificidad."),
            ("La barra de la derecha mide la <b>norma de selectividad</b>: "
             "el mejor predictor (ρ=0.64, p=0.001) de la caída de F1 bajo SVD. "
             "Las emociones \"escritas en negrita\" en los pesos del modelo "
             "son las más vulnerables a perturbaciones generales."),
            ("Curiosamente: la SVD no ataca selectivamente las neuronas "
             "emocionales (la geometría espectral es ortogonal a la función). "
             "Pero las emociones que requieren más capacidad neuronal son "
             "más frágiles a CUALQUIER perturbación."),
        ],
        "html": "04_sunburst_clusters.html",
        "iframe_height": 660,
    },
    {
        "id": "landscape",
        "kind": "fullbleed",
        "chapter": "CAP. 5",
        "title": "El paisaje emocional",
        "subtitle": "Cada emoción es un punto en (cristalización × intensidad)",
        "body": [
            ("Toma las 23 emociones, posiciónalas en un plano 2D donde "
             "X = capa de cristalización, Y = norma de selectividad. "
             "Sale el paisaje funcional del modelo emocional."),
            ("Cuadrante superior izquierdo: <i>gratitude</i>, <i>love</i> — "
             "tempranas e intensas. Cuadrante inferior derecho: "
             "<i>realization</i>, <i>disappointment</i> — tardías y difusas. "
             "Línea punteada: <i>sadness</i> ↔ <i>realization</i> compartiendo "
             "L11-H6 (también lo viste arriba)."),
            ("Selecciona una emoción en el menú de la derecha y verás su "
             "huella radial: 6 dimensiones funcionales en una sola figura "
             "polar."),
        ],
        "html": "06_emotional_landscape.html",
        "iframe_height": 700,
    },
    {
        "id": "neurons",
        "kind": "fullbleed",
        "chapter": "CAP. 5.4 · neuronas",
        "title": "Neuron portrait gallery",
        "subtitle": "Las 80 neuronas más selectivas con sus frases más activantes",
        "body": [
            ("§5.4 documenta 36.864 neuronas FFN con scores de selectividad "
             "Cohen's d. Pero los números no cuentan la historia completa. "
             "<b>Aquí ves QUÉ <i>significa</i></b> cada neurona — las 5 "
             "frases del test set que más la encienden."),
            ("Cada tarjeta es UNA neurona: su capa, su índice, su emoción "
             "dominante, su selectividad Cohen's d, y las 5 frases que la "
             "activan al máximo. Verás que la top-1 de admiration "
             "(<code>L11 N944</code>) se enciende con frases tipo elogio — "
             "el <i>significado</i> que el modelo asignó a esa neurona."),
            ("Filtra por emoción / capa / dirección (excitatoria vs "
             "inhibitoria). Es una mirada íntima al lenguaje interno — "
             "lo más cerca que llega un TFG a 'hablar el idioma de BERT'."),
        ],
        "html": "27_neuron_gallery.html",
        "iframe_height": 1100,
    },
    {
        "id": "galaxy",
        "kind": "fullbleed",
        "chapter": "CAP. 5 · 3D",
        "title": "Galaxy formation",
        "subtitle": "23 emociones cristalizando en el espacio LDA, capa por capa",
        "body": [
            ("Mismo experimento que los ríos pero en geometría 3D real. "
             "Cogemos el CLS de cada una de las 13 capas, le aplicamos el "
             "<b>pooler</b> (Linear + tanh) que tu modelo usa internamente, "
             "y proyectamos con <b>LDA supervisada</b> ajustada en L12. Los "
             "3 ejes son las direcciones que MEJOR separan las 23 emociones "
             "según el clasificador real."),
            ("Resultado cuantitativo: separation ratio 4.3 (los centroides "
             "están 4× más separados que la dispersión interna), 40% acc "
             "por nearest-centroid (vs 4% random). Los 23 diamantes que ves "
             "no se inventan — son posiciones reales en el espacio del "
             "modelo. <b>Click en la leyenda</b> para aislar emociones."),
            ("Es la prueba geométrica visceral de la cristalización: a L0 "
             "los 23 diamantes están casi superpuestos en el origen; a L11 "
             "ocupan regiones distintas, cada cual cerca de las frases "
             "que pertenecen a su emoción."),
        ],
        "html": "07_galaxy_formation.html",
        "iframe_height": 860,
    },
    {
        "id": "tokens",
        "kind": "fullbleed",
        "chapter": "CAP. 5 · 3D extendido",
        "title": "Token trajectories",
        "subtitle": "No solo CLS — todos los tokens viajando por el residual stream",
        "body": [
            ("La galaxy formation muestra solo el viaje del CLS. Pero <b>cada "
             "token</b> tiene su propia trayectoria a través de las 13 capas. "
             "Aquí los proyectamos todos a la misma LDA-3D que la galaxy."),
            ("Lo que verás: <b>⟨CLS⟩</b> (rojo, diamante grande) viaja muy "
             "lejos del origen hacia el centroide de la emoción gold — es el "
             "<b>agregador</b> en acción. <b>⟨SEP⟩</b> (naranja) también se "
             "mueve mucho. Pero los <b>tokens de contenido</b> (azules) y las "
             "<b>palabras función</b> (grises) apenas se mueven — el residual "
             "stream preserva su información local mientras CLS la integra."),
            ("Esto visualiza directamente §2.2.3 (residual stream con conexiones "
             "skip) y §2.3.2 (CLS como agregador entrenable). El asimétrico "
             "movimiento entre tokens es la prueba geométrica de que CLS está "
             "diseñado para ser especial."),
        ],
        "html": "23_token_trajectories.html",
        "iframe_height": 880,
    },
    {
        "id": "iterative",
        "kind": "fullbleed",
        "chapter": "CAP. 5 · logit lens",
        "title": "Iterative inference",
        "subtitle": "La curva en U que prueba que la decisión emerge tarde",
        "body": [
            ("Aplicamos el pooler + classifier reales del modelo a CADA capa "
             "(técnica conocida como <b>logit lens</b>). Promediamos sobre "
             "2300 frases y trazamos la sigmoid top-1, la sigmoid del gold "
             "label, y la suma de las 23 sigmoides — todas se mueven "
             "siguiendo el mismo patrón en U."),
            ("<b>Capas 0-3 (saturación)</b>: las estadísticas del CLS "
             "saturan el tanh del pooler → muchas emociones disparan a la "
             "vez con magnitud media. <b>Capas 4-9 (valle)</b>: el CLS está "
             "en transición, tanh ≈ 0, el modelo no decide nada. "
             "<b>Capas 10-11 (cristalización)</b>: el CLS llega a su régimen "
             "natural y una emoción pega un salto definitivo."),
            ("<b>Por qué importa para tu memoria</b>: este patrón refuerza "
             "§5.1 (cristalización) y §5.2 (FFN-L11 como cuello de botella). "
             "El valle medio es prueba de que la decisión emocional NO "
             "EXISTE hasta que la pipeline pooler+classifier se reactiva, "
             "lo que ocurre exactamente al llegar a L11. Por eso el "
             "activation patching de L11 recupera el 100% del F1: restauras "
             "exactamente esta calibración. Bibliografía: Nostalgebraist "
             "(2020) Logit Lens; Belrose et al. (NeurIPS 2023) Tuned Lens."),
        ],
        "html": "15_iterative_inference.html",
        "iframe_height": 980,
    },
    {
        "id": "atlas",
        "kind": "fullbleed",
        "chapter": "CAP. 5.3 · atención",
        "title": "Attention atlas",
        "subtitle": "Las 144 cabezas, todas a la vez, sobre una frase",
        "body": [
            ("Hasta ahora la matriz de cabezas (§5.3) te decía qué cabeza es "
             "<b>importante</b>, pero nunca <b>qué hace exactamente</b>. Esta "
             "viz cierra ese hueco. Eliges una frase y ves los 144 patrones "
             "de atención simultáneamente — uno por cabeza. Bordes "
             "coloreados por la categoría funcional real de tu notebook 6."),
            ("Click en cualquier cabeza para ampliar con etiquetas de "
             "tokens. Verás patrones reconocibles: capas tempranas atienden "
             "diagonalmente (token a sí mismo, vecinos), capas tardías "
             "concentran atención sobre [CLS] o [SEP] (rayas verticales — "
             "el patrón típico de los \"agregadores\")."),
            ("La capa 11 (fila inferior) no tiene ni una cabeza dispensable "
             "según tu memoria. Mira que sus 12 celdas tienen bordes rojos "
             "o azules — todas críticas."),
        ],
        "html": "16_attention_atlas.html",
        "iframe_height": 1100,
    },
    {
        "id": "probes",
        "kind": "fullbleed",
        "chapter": "CAP. 5 · superposition",
        "title": "Probe constellations",
        "subtitle": "Las 23 emociones como direcciones radiantes — superposition visualizada",
        "body": [
            ("Los 23 vectores de peso del classifier (uno por emoción, cada uno "
             "768-d) son <b>las direcciones</b> que el modelo usa para detectar "
             "cada emoción. Aquí los proyectamos a PCA-3D y los pintamos como "
             "flechas radiando desde el origen."),
            ("La lectura geométrica: vectores <b>casi-ortogonales</b> = "
             "emociones que el modelo distingue limpiamente. Vectores "
             "<b>casi-paralelos</b> = emociones que el modelo conflate "
             "(gratitude/love apuntan parecido). El heatmap a la derecha es "
             "la similitud coseno completa en 768-d, reordenada por cluster — "
             "verás bloques diagonales (alta similitud dentro de cluster)."),
            ("Esto es <b>§2.7.5 (superposition) hecho visual</b>: el modelo "
             "codifica más conceptos (23 emociones) que dimensiones tendría si "
             "fueran ortogonales puras. Las direcciones convergen en el espacio "
             "del clasificador y la geometría del cono de flechas te dice "
             "qué emociones comparten infraestructura representacional."),
        ],
        "html": "20_probe_constellations.html",
        "iframe_height": 780,
    },
    {
        "id": "confusion",
        "kind": "fullbleed",
        "chapter": "CAP. 5 · confusión",
        "title": "Confusion matrix evolution",
        "subtitle": "23×23 confusiones a través de las 13 capas",
        "body": [
            ("Aplicamos el logit lens en bloque. Para cada (gold, predicho), "
             "calculamos el sigmoid medio del modelo cuando la frase es de "
             "esa emoción. Animado por capa: en L0 mush uniforme, L7 hueco "
             "del valle, L11 diagonal limpia."),
            ("La parte interesante son las celdas <b>off-diagonal</b> que "
             "sobreviven en L11: pares de emociones que el modelo confunde "
             "incluso al final. <b>annoyance↔disapproval</b>, "
             "<b>fear↔sadness</b>, <b>approval↔realization</b>. Justamente "
             "los pares vecinos en los 6 clusters de §5.4.6 — el modelo no "
             "los separa porque psicológicamente están cerca."),
            ("Las líneas discontinuas marcan las fronteras entre los 6 "
             "clusters de §5.4.6. La diagonal limpia DENTRO de un cluster "
             "+ smudges entre clusters próximos = confirmación visual de "
             "tu taxonomía emergente."),
        ],
        "html": "17_confusion_evolution.html",
        "iframe_height": 920,
    },
    {
        "id": "conf-volume",
        "kind": "fullbleed",
        "chapter": "CAP. 5 · 3D",
        "title": "Confusion volume",
        "subtitle": "Las 13 confusion matrices apiladas como un cubo 3D",
        "body": [
            ("La confusion matrix evolution era 2D animada. Esta es la versión "
             "<b>3D estática y rotable</b>: las 13 matrices 23×23 apiladas "
             "verticalmente como un cubo. Solo dibujamos celdas con sigmoid "
             "medio &gt; 0.15 para que el cubo no sea opaco."),
            ("Lo que cuenta esta vista: las celdas que <b>persisten "
             "verticalmente</b> a lo largo de las capas son <b>confusiones "
             "estructurales</b> del modelo — pares de emociones que NUNCA se "
             "separan completamente. La diagonal es la columna verde central; "
             "los smudges off-diagonal son las confusiones residuales que "
             "viven en cada capa."),
            ("Rota el cubo. Verás que algunos pares (annoyance↔disapproval, "
             "fear↔sadness) tienen <b>columnas verticales</b> tan altas como "
             "la diagonal — el modelo confunde estos pares en TODAS las "
             "capas. Justamente los vecinos en los 6 clusters de §5.4.6, lo "
             "que confirma una vez más que la confusión sigue la geometría "
             "psicológica."),
        ],
        "html": "22_confusion_volume.html",
        "iframe_height": 880,
    },
    {
        "id": "greedy",
        "kind": "fullbleed",
        "chapter": "CAP. 6.3 · algoritmo",
        "title": "Greedy algorithm replay",
        "subtitle": "Cómo construye el algoritmo greedy la compresión",
        "body": [
            ("§6.3 dice que el greedy elige movimientos por eficiencia "
             "(parámetros ahorrados / coste F1). Aquí lo VES en acción. "
             "Empezamos con baseline (todo a rango 768) y vamos avanzando: "
             "greedy_95 → greedy_90 → … → greedy_50."),
            ("La matriz 12×6 se va iluminando célula a célula. Verás que "
             "<b>las primeras decisiones son Q y K</b> (gratis, sin coste F1) "
             "— exactamente lo que predice tu §4.3 sobre la inmunidad de Q/K. "
             "Después vienen FFN-output en capas tempranas (también baratos). "
             "Las capas tardías (8-11) se mantienen intactas hasta el final."),
            ("La línea derecha sigue el F1 retention vs ratio paso a paso. "
             "Es la prueba algorítmica de que tu greedy reproduce los "
             "hallazgos de interpretabilidad sin tener acceso a ellos: "
             "se entera SOLO con los datos empíricos de sensibilidad."),
        ],
        "html": "18_greedy_replay.html",
        "iframe_height": 820,
    },
    {
        "id": "trajectory",
        "kind": "fullbleed",
        "chapter": "CAP. 5 · ★ síntesis",
        "title": "Sentence trajectory",
        "subtitle": "Una frase, cuatro vistas síncronas — la pieza síntesis",
        "body": [
            ("Aquí se juntan todas las viz en una sola experiencia. Eliges "
             "una frase y arrastras el slider de capa. Los CUATRO paneles "
             "se mueven sincronizados: trayectoria 3D del CLS, atención de "
             "las 3 cabezas más críticas, sigmoides multi-label, y curva "
             "del gold."),
            ("Esta es la forma más completa de ver lo que tu memoria "
             "describe sección a sección, ahora junto. Pulsa Play y ves: "
             "(1) el CLS arrancar cerca del origen y desplazarse hacia su "
             "centroide, (2) las cabezas críticas activarse según la capa, "
             "(3) los pétalos saltando del valle a la cristalización, (4) "
             "la confianza gold trazando la U que ya conoces."),
            ("Es el experimento que pediste al inicio del proyecto: ver "
             "BERT pensar en una frase, capa a capa, todas las dimensiones "
             "a la vez. Funciona porque tienes los datos correctos: "
             "tokenizer correcto, pooler aplicado, LDA real, atención real."),
        ],
        "html": "19_sentence_trajectory.html",
        "iframe_height": 980,
    },
    {
        "id": "cards",
        "kind": "fullbleed",
        "chapter": "★ DASHBOARD",
        "title": "Emotion trading cards",
        "subtitle": "El dashboard completo · 23 tarjetas, una por emoción",
        "body": [
            ("La síntesis ejecutiva del TFG. Cada emoción reducida a UNA "
             "tarjeta con su perfil completo: F1 baseline, F1 tras "
             "compresión, recuperación tras fine-tuning, capa de "
             "cristalización, cabeza crítica, neuronas significativas, "
             "selectividad neuronal, y un radar de 6 dimensiones funcionales."),
            ("Las tarjetas se pueden <b>reordenar</b>: por cluster "
             "psicológico (default), por F1, por capa de cristalización, "
             "por número de neuronas, por mejora con fine-tuning, "
             "alfabético. Cada vista ofrece una historia distinta — orden "
             "por F1 baseline muestra qué emociones son fáciles vs "
             "difíciles; orden por mejora con fine-tuning muestra cuáles "
             "se benefician más de la regularización."),
            ("Es la pieza que pones en la portada de la memoria. Es la "
             "que enseñas en la primera diapositiva. Cuando el tribunal "
             "pregunte sobre cualquier emoción concreta, click en su "
             "tarjeta y tienes el dossier."),
        ],
        "html": "24_emotion_cards.html",
        "iframe_height": 1300,
    },
    {
        "id": "fingerprint",
        "kind": "fullbleed",
        "chapter": "CAP. 5 · multi-label",
        "title": "Decision fingerprint",
        "subtitle": "La firma multi-label de UNA frase, capa a capa",
        "body": [
            ("Lo que las métricas agregadas (heatmap, ríos) esconden: para "
             "CADA frase concreta, el modelo no produce <i>una</i> emoción — "
             "produce un vector de 23 probabilidades, varias activas a la vez. "
             "Ese vector es la <b>firma multi-label</b>."),
            ("Aplicamos la cabeza clasificadora real de tu modelo al CLS de "
             "cada capa. Cada pétalo del polar es una emoción; su longitud, "
             "el sigmoid output. <b>Pétalo con borde negro</b> = emoción gold "
             "de la frase. <b>Selecciona frase</b> en el menú, dale a Play, "
             "ves el fingerprint emerger desde el ruido (Emb) hasta la "
             "decisión cristalizada (L11)."),
            ("Lo interesante son los <b>pétalos paralelos</b>: cuando varios "
             "crecen juntos, ese es el modelo creyendo que múltiples emociones "
             "co-ocurren. Es exactamente lo que la BCE multi-label produce, "
             "y es lo que ningún heatmap promediado deja ver."),
        ],
        "html": "14_decision_fingerprint.html",
        "iframe_height": 760,
    },
    {
        "id": "decay",
        "kind": "fullbleed",
        "chapter": "CAP. 4.2 · animado",
        "title": "Compression galaxy decay",
        "subtitle": "Misma galaxia, distintos rangos · la transición de fase visualizada",
        "body": [
            ("La galaxy formation muestra cómo BERT construye estructura. "
             "Esta muestra cómo la SVD la <b>destruye</b>. Las mismas 588 "
             "frases en L12, pero con SVD aplicada uniformemente a {768, "
             "512, 384, 256, 128, 64}."),
            ("Arrastra el slider o pulsa Play. A r=512 todo casi igual. "
             "<b>Entre r=384 y r=256 ves el acantilado real</b>: los clusters "
             "se difuminan. A r=128 la geometría desaparece. A r=64 los "
             "embeddings colapsan a un blob."),
            ("La gráfica derecha cuantifica la degradación: silhouette "
             "(separabilidad) en azul, retención de F1 (Tabla 8 de la "
             "memoria) en terra. Las dos curvas caen juntas a partir de "
             "r=384. La transición de fase de §4.2 — vista como geometría."),
        ],
        "html": "13_compression_decay.html",
        "iframe_height": 800,
    },
    {
        "id": "ft-diff",
        "kind": "fullbleed",
        "chapter": "META · pre vs post",
        "title": "¿Qué cambió el fine-tuning?",
        "subtitle": "Diff Frobenius entre bert-base-uncased y tu 23emo-final",
        "body": [
            ("Antes de seguir compresionando, mira QUÉ HIZO el fine-tuning. "
             "Cargamos el modelo pre-entrenado (<code>bert-base-uncased</code>) "
             "y tu checkpoint <code>23emo-final</code>, y para cada una de las "
             "72 matrices calculamos el cambio relativo: <b>‖W_ft − W_pre‖ / "
             "‖W_pre‖</b>."),
            ("La predicción de §5.5 (\"arquitectura de dos fases\") tiene "
             "implicación clara: <b>el gradiente fluye más fuerte hacia las "
             "capas finales</b>, así que esas deberían cambiar más durante "
             "fine-tuning. Las tempranas, ya buenas en BERT pretrained, "
             "deberían cambiar poco."),
            ("Mira el heatmap: las capas tardías (8-11) tienen valores "
             "claramente más altos. Es <b>evidencia empírica directa</b> de "
             "que tu fine-tuning concentró el aprendizaje exactamente "
             "donde tu memoria predice. Otra confirmación cruzada de §5.5."),
        ],
        "html": "25_finetuning_diff.html",
        "iframe_height": 920,
    },
    {
        "id": "spec-landscape",
        "kind": "fullbleed",
        "chapter": "CAP. 4.1 · 3D",
        "title": "Spectral landscape",
        "subtitle": "La asimetría espectral hecha topografía 3D",
        "body": [
            ("Si las spectral flowers eran 72 imágenes, esto es <b>una sola "
             "topografía</b>. Mismas 72 matrices del modelo, ahora apiladas "
             "como filas de un terreno 3D. Eje X = índice del valor singular, "
             "eje Z = magnitud σᵢ/σ₁."),
            ("Verás el contraste más nítido del TFG: <b>las matrices Q y K "
             "forman picos abruptos</b> (decaimiento espectral rápido — pocos "
             "valores singulares dominan), mientras que <b>las FFN forman "
             "mesetas casi planas</b> (espectro distribuido). La asimetría que "
             "motiva toda la compresión informada de §4 hecha relieve."),
            ("Diamantes = k95 (rango efectivo) de cada matriz. Q/K en ~395, "
             "FFN en ~620 — exactamente la Tabla 6 de tu memoria. Rota la "
             "cámara para ver la diferencia frontal vs trasera. Datos: SVD "
             "real computada sobre tu checkpoint."),
        ],
        "html": "21_spectral_landscape.html",
        "iframe_height": 860,
    },
    {
        "id": "flowers",
        "kind": "fullbleed",
        "chapter": "CAP. 4.1 · iconográfico",
        "title": "Spectral flowers",
        "subtitle": "72 matrices, 72 huellas espectrales",
        "body": [
            ("Cada flor es una matriz de pesos. Cada pétalo es un valor "
             "singular σᵢ, normalizado a σ₁. Tomamos 32 pétalos por flor. "
             "El SVD se computó sobre el modelo real."),
            ("Mira las columnas Q y K (azul): flores con 2-3 pétalos enormes "
             "que dominan, el resto diminutos. Espectro <b>concentrado</b> "
             "→ rango efectivo bajo → comprimibles. Mira las columnas FFN "
             "(turquesa): flores rellenas, casi circulares, donde TODOS los "
             "pétalos son comparables. Espectro <b>plano</b> → cada dimensión "
             "aporta → frágiles bajo SVD."),
            ("Una sola imagen, una sola explicación: la asimetría espectral "
             "que motiva §4.1 entera. La forma de la flor predice la "
             "compresibilidad. Pasa el ratón sobre cualquier flor para ver "
             "su k95 exacto."),
        ],
        "html": "12_spectral_flowers.html",
        "iframe_height": 1240,
    },
    {
        "id": "sandbox",
        "kind": "scrolly",
        "chapter": "CAP. 6",
        "title": "Beat the greedy",
        "subtitle": "Diseña tu propia compresión y mira cómo aguanta",
        "body": [
            ("18 sliders. 6 componentes (Q, K, V, AttnOut, FFN-int, FFN-out) "
             "× 3 bandas de profundidad (early, middle, late). Cada slider "
             "es un rango SVD de 32 a 768. Total: 5^18 configuraciones "
             "posibles."),
            ("El simulador estima F1 macro y posición Pareto en tiempo real "
             "usando un damage model fitado a las Tablas 9 y 10 de la "
             "memoria. Los presets te dan puntos de partida (uniform_r256, "
             "greedy_90)."),
            ("Reto: <b>¿puedes vencer al algoritmo greedy</b> que retiene 93% "
             "del F1 con 14% menos de parámetros? La única manera de "
             "intentarlo es entender cuáles componentes son realmente "
             "frágiles. Pista: lo que viste en el activation patching."),
        ],
        "html": "05_compression_sandbox.html",
        "iframe_height": 880,
    },
    {
        "id": "outro",
        "kind": "outro",
        "chapter": "FIN",
        "title": "Cómo está hecho",
        "subtitle": "Stack y créditos",
        "body": [
            ("<b>Datos numéricos.</b> Todas las visualizaciones se alimentan "
             "directamente de los CSVs de los notebooks 2-9 (resultados reales "
             "del fine-tune del autor sobre BERT-base-uncased y 23 emociones "
             "GoEmotions). 61 tablas exportadas: probing por capa, ablación de "
             "144 cabezas, especialización neuronal, activation patching, "
             "frontera de Pareto completa con 21 estrategias evaluadas."),
            ("<b>Activaciones y geometría.</b> Galaxy formation, sentence "
             "divergence, compression decay y spectral flowers se computan "
             "ejecutando el checkpoint <code>bert-goemotions-23emo-final/</code> "
             "(109.5M params, 23 emociones) sobre 690 frases del test set. PCA "
             "fija ajustada en L12, mismas coordenadas para todas las capas."),
            ("<b>Visualización.</b> 9 piezas: 6 estáticas en Plotly + 2 "
             "interactivas con animación + 1 grafo custom en D3.js puro. "
             "HTMLs autocontenidos, sin servidor — todo el contenido vive "
             "embebido como JSON dentro del propio archivo."),
            ("<b>Memoria.</b> Guido Biosca Lasa, director Lluís Padró Cirera, "
             "FIB-UPC, 2026."),
        ],
        "html": None,
    },
]


# ─── HTML template ───────────────────────────────────────────────────────────

def build_index() -> pathlib.Path:
    nav_items = []
    sections_html = []

    for s in SECTIONS:
        nav_items.append(
            f'<a href="#{s["id"]}" data-section="{s["id"]}">'
            f'<span class="ch">{s["chapter"]}</span>'
            f'<span class="t">{s["title"].replace("<br>", " ")}</span></a>'
        )

        if s["kind"] == "hero":
            body_html = "".join(f'<p>{b}</p>' for b in s["body"])
            sections_html.append(f"""
<section id="{s['id']}" class="hero" data-section="{s['id']}">
  <div class="chapter">{s['chapter']}</div>
  <h1>{s['title']}</h1>
  <div class="subtitle">{s['subtitle']}</div>
  <div class="body">{body_html}</div>
  <div class="byline">
    Guido Biosca Lasa · Director: Lluís Padró Cirera · FIB-UPC · 2026
  </div>
  <div class="scroll-cue">↓ Scrolla para empezar el recorrido</div>
</section>""")

        elif s["kind"] == "callout":
            body_html = "".join(f'<p>{b}</p>' for b in s["body"])
            sections_html.append(f"""
<section id="{s['id']}" class="callout" data-section="{s['id']}">
  <div class="chapter">{s['chapter']}</div>
  <h2>{s['title']}</h2>
  <div class="subtitle">{s['subtitle']}</div>
  <div class="body">{body_html}</div>
</section>""")

        elif s["kind"] == "outro":
            body_html = "".join(f'<p>{b}</p>' for b in s["body"])
            sections_html.append(f"""
<section id="{s['id']}" class="outro" data-section="{s['id']}">
  <div class="chapter">{s['chapter']}</div>
  <h2>{s['title']}</h2>
  <div class="subtitle">{s['subtitle']}</div>
  <div class="body">{body_html}</div>
</section>""")

        elif s["kind"] == "fullbleed":
            body_html = "".join(f'<p>{b}</p>' for b in s["body"])
            iframe_h = s.get("iframe_height", 720)
            sections_html.append(f"""
<section id="{s['id']}" class="fullbleed" data-section="{s['id']}">
  <div class="header">
    <div class="chapter">{s['chapter']}</div>
    <h2>{s['title']}</h2>
    <div class="subtitle">{s['subtitle']}</div>
    <div class="body">{body_html}</div>
  </div>
  <div class="frame-wrap" style="--iframe-h: {iframe_h}px">
    <iframe data-src="{s['html']}" loading="lazy"></iframe>
  </div>
</section>""")

        else:  # "scrolly"
            paragraphs_html = "".join(f'<p>{b}</p>' for b in s["body"])
            iframe_h = s.get("iframe_height", 700)
            sections_html.append(f"""
<section id="{s['id']}" class="scrolly" data-section="{s['id']}">
  <div class="text-col">
    <div class="chapter">{s['chapter']}</div>
    <h2>{s['title']}</h2>
    <div class="subtitle">{s['subtitle']}</div>
    {paragraphs_html}
  </div>
  <div class="figure-col">
    <div class="frame-wrap" style="--iframe-h: {iframe_h}px">
      <iframe data-src="{s['html']}" loading="lazy"></iframe>
    </div>
  </div>
</section>""")

    nav_html = "\n".join(nav_items)
    body = "\n".join(sections_html)

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Anatomía Emocional · TFG Guido Biosca</title>
<link rel="preconnect" href="https://cdn.plot.ly">
<style>
  :root {{
    --ink: #1A1A1A; --ink-2: #4A4A4A; --ink-3: #8A8A85;
    --bg: #FAFAF6; --paper: #FFFFFF; --spine: #C8C7C1; --grid: #EBEBEB;
    --terra: #C1553A; --sand: #D4A843; --sage: #5A8F7B; --blue: #3A6EA5; --plum: #7B5E7B;
    --serif: "TeX Gyre Pagella", "Palatino Linotype", "Palatino", "Book Antiqua", "Iowan Old Style", serif;
    --sans: "Inter", "Helvetica Neue", -apple-system, BlinkMacSystemFont, sans-serif;
    --mono: "JetBrains Mono", "SF Mono", Menlo, monospace;
  }}
  * {{ box-sizing: border-box; }}
  html {{ scroll-behavior: smooth; }}
  body {{
    margin: 0; background: var(--bg); color: var(--ink);
    font-family: var(--serif); font-size: 17px; line-height: 1.65;
  }}
  /* Reading progress bar */
  #progress {{
    position: fixed; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(to right, var(--terra) var(--scroll, 0%), transparent 0);
    z-index: 100;
  }}
  /* Side nav */
  nav {{
    position: fixed; left: 0; top: 0; width: 230px; height: 100vh;
    background: var(--paper); border-right: 0.5px solid var(--spine);
    padding: 36px 22px 30px 30px; overflow-y: auto;
    transition: opacity 0.3s;
    z-index: 50;
  }}
  nav h1 {{
    font-family: var(--serif); font-size: 15.5px; font-weight: normal;
    color: var(--ink); margin: 0 0 4px 0; letter-spacing: 0.2px; line-height: 1.3;
  }}
  nav h1 .acc {{ color: var(--terra); }}
  nav .author {{
    font-family: var(--sans); font-size: 10.5px; color: var(--ink-3);
    margin-bottom: 28px; letter-spacing: 0.5px;
  }}
  nav a {{
    display: block; text-decoration: none; padding: 10px 8px;
    border-left: 2px solid transparent; transition: all 0.18s;
    margin-bottom: 1px; border-radius: 2px;
  }}
  nav a:hover, nav a.active {{
    background: rgba(193,85,58,0.07); border-left-color: var(--terra);
  }}
  nav a .ch {{
    display: block; font-family: var(--sans); font-size: 9px; font-weight: 500;
    color: var(--ink-3); letter-spacing: 0.9px; text-transform: uppercase;
  }}
  nav a .t {{
    display: block; font-size: 13px; color: var(--ink); margin-top: 2px; line-height: 1.3;
  }}
  nav a.active .t {{ color: var(--terra); }}

  main {{ margin-left: 230px; }}

  /* HERO */
  section.hero {{
    min-height: 100vh; padding: 12vh 8vw 6vh 8vw; max-width: 1400px;
    display: flex; flex-direction: column; justify-content: center;
  }}
  section.hero .chapter {{
    font-family: var(--sans); font-size: 11px; font-weight: 500; color: var(--terra);
    text-transform: uppercase; letter-spacing: 1.6px; margin-bottom: 18px;
  }}
  section.hero h1 {{
    font-size: 64px; line-height: 1.05; margin: 0 0 18px 0; font-weight: normal;
    letter-spacing: -0.6px; max-width: 920px;
  }}
  section.hero .subtitle {{
    font-style: italic; color: var(--ink-2); font-size: 22px; margin-bottom: 32px;
    max-width: 780px;
  }}
  section.hero .body {{ max-width: 660px; color: var(--ink-2); font-size: 17px; }}
  section.hero .body p {{ margin: 0 0 16px 0; }}
  section.hero .byline {{
    margin-top: 50px; font-family: var(--sans); font-size: 12px;
    color: var(--ink-3); letter-spacing: 0.5px;
  }}
  section.hero .scroll-cue {{
    position: absolute; bottom: 4vh; font-family: var(--sans); font-size: 12px;
    color: var(--ink-3); letter-spacing: 0.4px; animation: pulse 2.5s ease-in-out infinite;
  }}
  @keyframes pulse {{
    0%, 100% {{ opacity: 0.4; transform: translateY(0); }}
    50% {{ opacity: 1; transform: translateY(4px); }}
  }}

  /* SCROLLY: text + sticky figure */
  section.scrolly {{
    display: grid; grid-template-columns: 380px 1fr; gap: 52px;
    padding: 70px 60px 70px 60px;
    max-width: 1500px;
    border-top: 0.5px solid var(--spine);
  }}
  section.scrolly .text-col {{
    padding-top: 12px;
  }}
  section.scrolly .figure-col {{
    position: sticky; top: 30px; height: fit-content;
  }}
  section.scrolly .chapter, section.fullbleed .chapter, section.callout .chapter, section.outro .chapter {{
    font-family: var(--sans); font-size: 10.5px; font-weight: 500; color: var(--terra);
    text-transform: uppercase; letter-spacing: 1.4px; margin-bottom: 10px;
  }}
  section.scrolly h2, section.fullbleed h2, section.callout h2, section.outro h2 {{
    font-size: 36px; font-weight: normal; line-height: 1.1; margin: 0 0 8px 0;
    letter-spacing: -0.3px;
  }}
  section.scrolly .subtitle, section.fullbleed .subtitle,
  section.callout .subtitle, section.outro .subtitle {{
    font-style: italic; color: var(--ink-2); font-size: 18px; margin-bottom: 22px;
  }}
  section.scrolly p {{ color: var(--ink-2); margin: 0 0 18px 0; }}

  /* FULLBLEED — galaxy gets the spotlight */
  section.fullbleed {{
    padding: 80px 60px;
    max-width: 1500px;
    border-top: 0.5px solid var(--spine);
  }}
  section.fullbleed .header {{ max-width: 800px; margin-bottom: 28px; }}
  section.fullbleed .body p {{ color: var(--ink-2); margin: 0 0 12px 0; max-width: 800px; }}

  /* CALLOUT — sentence inspector pitch */
  section.callout {{
    padding: 80px 60px; max-width: 1100px;
    border-top: 0.5px solid var(--spine);
    background: linear-gradient(135deg, rgba(212,168,67,0.08), rgba(193,85,58,0.06));
  }}
  section.callout .body p {{ color: var(--ink-2); max-width: 800px; }}
  section.callout pre {{
    background: var(--ink); color: #FFFFFF; font-family: var(--mono);
    font-size: 13px; padding: 14px 18px; border-radius: 4px;
    max-width: 740px; overflow-x: auto;
  }}
  section.callout code {{
    font-family: var(--mono); font-size: 0.9em;
    background: rgba(0,0,0,0.05); padding: 2px 5px; border-radius: 3px;
  }}

  /* OUTRO */
  section.outro {{
    padding: 80px 60px 120px 60px; max-width: 1100px;
    border-top: 0.5px solid var(--spine);
  }}
  section.outro .body p {{ color: var(--ink-2); max-width: 800px; margin: 0 0 16px 0; }}
  section.outro code {{
    font-family: var(--mono); font-size: 0.85em;
    background: rgba(0,0,0,0.05); padding: 2px 5px; border-radius: 3px;
  }}

  /* IFRAME wrapper */
  .frame-wrap {{
    position: relative; border: 0.5px solid var(--spine); border-radius: 4px;
    overflow: hidden; background: var(--paper);
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    transform: translateY(20px); opacity: 0;
    transition: transform 0.7s ease-out, opacity 0.7s ease-out;
  }}
  .frame-wrap.in-view {{ transform: translateY(0); opacity: 1; }}
  .frame-wrap iframe {{
    border: none; width: 100%; height: var(--iframe-h, 720px); display: block;
    background: var(--paper);
  }}
  .frame-wrap::before {{
    content: "Cargando…"; position: absolute; inset: 0;
    display: flex; align-items: center; justify-content: center;
    color: var(--ink-3); font-style: italic; font-size: 14px;
    opacity: 1; transition: opacity 0.3s;
  }}
  .frame-wrap.loaded::before {{ opacity: 0; pointer-events: none; }}

  @media (max-width: 1100px) {{
    nav {{ position: relative; width: 100%; height: auto;
           border-right: none; border-bottom: 0.5px solid var(--spine);
           padding: 14px 22px; }}
    nav a {{ display: inline-block; margin-right: 14px; padding: 4px 6px; border-left: none; border-bottom: 2px solid transparent; }}
    nav a:hover, nav a.active {{ border-left: none; border-bottom-color: var(--terra); }}
    main {{ margin-left: 0; }}
    section.hero {{ padding: 8vh 6vw; }}
    section.hero h1 {{ font-size: 42px; }}
    section.scrolly {{ grid-template-columns: 1fr; padding: 50px 28px; gap: 24px; }}
    section.scrolly .figure-col {{ position: static; }}
    section.fullbleed, section.callout, section.outro {{ padding: 50px 28px; }}
  }}
</style>
</head>
<body>

<div id="progress"></div>

<nav>
  <h1>Anatomía emocional<br><span class="acc">de un Transformer</span></h1>
  <div class="author">Guido Biosca Lasa<br>TFG · FIB-UPC</div>
  {nav_html}
</nav>

<main>
{body}
</main>

<script>
  // Reading progress bar
  function updateProgress() {{
    const scrolled = window.scrollY;
    const total = document.documentElement.scrollHeight - window.innerHeight;
    const pct = Math.min(100, (scrolled / total) * 100);
    document.documentElement.style.setProperty('--scroll', pct + '%');
  }}
  window.addEventListener('scroll', updateProgress, {{ passive: true }});
  updateProgress();

  // Active section in sidebar nav
  const navLinks = document.querySelectorAll('nav a');
  const sections = document.querySelectorAll('section[data-section]');
  const navObserver = new IntersectionObserver(entries => {{
    entries.forEach(e => {{
      if (e.isIntersecting) {{
        const id = e.target.dataset.section;
        navLinks.forEach(a => a.classList.toggle('active', a.dataset.section === id));
      }}
    }});
  }}, {{ rootMargin: "-30% 0px -60% 0px", threshold: 0 }});
  sections.forEach(s => navObserver.observe(s));

  // Lazy-load iframes when in view, with fade-in
  const frames = document.querySelectorAll('.frame-wrap');
  const frameObserver = new IntersectionObserver(entries => {{
    entries.forEach(e => {{
      if (e.isIntersecting) {{
        const wrap = e.target;
        const iframe = wrap.querySelector('iframe');
        if (iframe && iframe.dataset.src && !iframe.src) {{
          iframe.src = iframe.dataset.src;
          iframe.addEventListener('load', () => wrap.classList.add('loaded'), {{ once: true }});
        }}
        wrap.classList.add('in-view');
      }}
    }});
  }}, {{ rootMargin: "0px 0px -10% 0px", threshold: 0.05 }});
  frames.forEach(f => frameObserver.observe(f));
</script>

</body>
</html>
"""
    out = OUT / "index.html"
    out.write_text(html, encoding="utf-8")
    print(f"✓ wrote {out}")
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("\n=== Building all visualizations ===\n")

    # Cap. 4 — compression results
    spectral_flowers.main(OUT)
    spectral_landscape.main(OUT)
    pareto_3d.main(OUT)
    compression_decay.main(OUT)
    # Cap. 5 — interpretability
    crystallization.main(OUT)
    galaxy_formation.main(OUT)
    token_trajectories.main(OUT)
    iterative_inference.main(OUT)
    heads_matrix.main(OUT)
    attention_atlas.main(OUT)
    probe_constellations.main(OUT)
    circuit_network.main(OUT)
    lesion_theater.main(OUT)
    sunburst.main(OUT)
    fingerprints.main(OUT)
    confusion_evolution.main(OUT)
    confusion_volume.main(OUT)
    decision_fingerprint.main(OUT)
    sentence_trajectory.main(OUT)
    # Cap. 6 — synthesis
    greedy_replay.main(OUT)
    compression_sandbox.main(OUT)
    # Top-tier additions: dashboard / arquitectura / fine-tune diff / neuronas
    bert_architecture.main(OUT)
    finetuning_diff.main(OUT)
    emotion_cards.main(OUT)
    neuron_gallery.main(OUT)
    lexical_to_semantic.main(OUT)
    internal_compression.main(OUT)

    print("\n=== Building scrollytelling index ===\n")
    build_index()

    print(f"\n✓ Done. Open {OUT / 'index.html'} in a browser.\n")


if __name__ == "__main__":
    main()
