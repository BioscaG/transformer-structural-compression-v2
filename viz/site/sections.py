"""Bilingual section content (ES/EN) for the editorial site.

Each text field is a dict ``{"es": ..., "en": ...}`` produced by the helper
``T(es, en)``. List fields like ``body`` are lists of those dicts. The build
renders both versions wrapped with a language attribute and a JS toggle in
the nav swaps which one is visible.

Voice: direct, concrete. Short sentences, real numbers, no AI-isms.
"""

from __future__ import annotations


def T(es: str, en: str) -> dict:
    """Bilingual string helper."""
    return {"es": es, "en": en}


SECTIONS: list[dict] = [
    # ── PART 1: el sujeto ────────────────────────────────────────────────
    {
        "kind": "part",
        "id": "parte-1",
        "num": T("Parte 01", "Part 01"),
        "title": T("El sujeto", "The subject"),
        "intro": [
            T("BERT-base. 12 capas. 144 cabezas de atención. 36 864 "
              "neuronas en los bloques FFN. 109,5 millones de parámetros.",
              "BERT-base. 12 layers. 144 attention heads. 36,864 neurons "
              "across the FFN blocks. 109.5 million parameters."),
            T("Encima, un fine-tune sobre GoEmotions: 23 emociones, "
              "etiquetado multi-label. Cada frase puede llevar varias "
              "emociones a la vez.",
              "On top of it, a fine-tune on GoEmotions: 23 emotions with "
              "multi-label tagging. A single sentence can carry several "
              "emotions at once."),
            T("Antes de operar nada, primero hay que conocer al sujeto.",
              "Before cutting anything open, you have to meet the subject."),
        ],
    },
    {
        "kind": "figure",
        "id": "arquitectura",
        "chapter": T("§ 01.1 · Arquitectura", "§ 01.1 · Architecture"),
        "title": T("BERT entero, en 3D", "All of BERT, in 3D"),
        "subtitle": T(
            "12 capas apiladas, cada una con 12 cabezas de atención y un bloque FFN.",
            "12 stacked layers, each with 12 attention heads and an FFN block."),
        "body": [
            T("Cada esfera es una cabeza de atención. Cada anillo "
              "turquesa es un bloque FFN. Por la columna central viaja "
              "el residual stream con el [CLS] desde abajo hasta el "
              "classifier de arriba.",
              "Each sphere is an attention head. Each turquoise ring is "
              "an FFN block. The dotted central column is the residual "
              "stream that carries [CLS] from the bottom to the "
              "classifier on top."),
            T("El color codifica la categoría funcional según el "
              "ablation study (notebook 6). Roja: critical specialist. "
              "Azul: critical generalist. Amarilla: minor. Gris: "
              "prescindible. El tamaño es el impacto agregado al "
              "ablacionar.",
              "Color encodes the functional category from the ablation "
              "study (notebook 6). Red: critical specialist. Blue: "
              "critical generalist. Yellow: minor. Grey: dispensable. "
              "Size encodes aggregate impact when the head is ablated."),
            T("Compara la capa 11 con la capa 0. Arriba casi todo es "
              "rojo y azul. Abajo casi todo es gris. El trabajo "
              "emocional no está distribuido uniformemente — vive "
              "concentrado al final.",
              "Compare layer 11 with layer 0. Up top almost everything "
              "is red and blue. Down at the bottom it's mostly grey. "
              "Emotional work isn't spread evenly — it's concentrated "
              "at the end of the stack."),
        ],
        "figure": "bert_architecture",
        "caption": T(
            "Esferas: cabezas. Anillos: bloques FFN. Columna punteada: "
            "residual stream. Diamante rojo: token [CLS]. Cuadrado verde: "
            "classifier de 23 emociones. Datos: head_categories.csv.",
            "Spheres: heads. Rings: FFN blocks. Dotted column: residual "
            "stream. Red diamond: [CLS] token. Green square: 23-emotion "
            "classifier. Data: head_categories.csv."),
        "fig_id": "01.1",
    },
    {
        "kind": "figure",
        "id": "lex2sem",
        "chapter": T("§ 01.2 · Base teórica", "§ 01.2 · Theoretical base"),
        "title": T("De diccionario a clasificador",
                   "From dictionary to classifier"),
        "subtitle": T("Cómo BERT olvida palabras y aprende emociones.",
                      "How BERT forgets words and learns emotions."),
        "body": [
            T("Una palabra entra en BERT y, al salir, ya no es esa "
              "palabra. El modelo sustituye \"qué palabra es\" por "
              "\"qué papel juega\" capa a capa. Es un fenómeno conocido "
              "(Tenney 2019, Ethayarajh 2019).",
              "A word enters BERT and on the way out it isn't that word "
              "anymore. The model replaces \"which word is this\" with "
              "\"what role does it play\" layer by layer. The phenomenon "
              "is well documented (Tenney 2019, Ethayarajh 2019)."),
            T("Tres curvas miden la transición. Azul: retención léxica, "
              "cuánto del embedding original sobrevive. Amarilla: "
              "anisotropía, cuán parecidos son los tokens entre sí. "
              "Roja: F1 del probe lineal de emoción, cuánta información "
              "de la etiqueta es linealmente extraíble.",
              "Three curves track the transition. Blue: lexical "
              "retention — how much of the original embedding survives. "
              "Yellow: anisotropy — how similar the tokens look to each "
              "other. Red: linear-probe F1 for emotion — how much of "
              "the label is linearly recoverable."),
            T("Las tres se cruzan en L8–L9. Antes, fase léxica. "
              "Después, fase semántica. Los cuatro mini-heatmaps lo "
              "hacen visceral: filas tempranas con colores variados "
              "(cada token preserva identidad), filas tardías "
              "monocromas (todos los tokens han colapsado al mismo "
              "vector contextual).",
              "The three curves cross around L8–L9. Before that, "
              "lexical phase. After that, semantic phase. The four "
              "mini-heatmaps make it tangible: early rows show varied "
              "colors (each token keeps its identity), late rows go "
              "monochrome (every token has collapsed onto the same "
              "contextual vector)."),
        ],
        "figure": "lexical_to_semantic",
        "caption": T(
            "Tres curvas sobre 46 frases del test set. Heatmaps de cuatro "
            "frases ejemplo: cos(hidden[L,t], hidden[0,t]). Refs: Tenney "
            "et al. ACL 2019; Ethayarajh EMNLP 2019.",
            "Three curves over 46 test-set sentences. Heatmaps of four "
            "example sentences: cos(hidden[L,t], hidden[0,t]). Refs: "
            "Tenney et al. ACL 2019; Ethayarajh EMNLP 2019."),
        "fig_id": "01.2",
    },

    # ── PART 2: compresión ───────────────────────────────────────────────
    {
        "kind": "part",
        "id": "parte-2",
        "num": T("Parte 02", "Part 02"),
        "title": T("El cuerpo en pedazos", "The body in pieces"),
        "intro": [
            T("El plan: aplicar SVD a las matrices de pesos del modelo, "
              "quedarte con los k mayores valores singulares y ver "
              "cuánto cae el F1.",
              "The plan: apply SVD to the model's weight matrices, keep "
              "the top-k singular values and watch how much F1 drops."),
            T("La sorpresa: el modelo ya está comprimiendo por dentro. "
              "Los tokens al final viven en un subespacio de baja "
              "dimensión. La SVD no introduce compresión donde no la "
              "había. Materializa la que ya existe.",
              "The surprise: the model is already compressing itself "
              "internally. Tokens at the end live in a low-dimensional "
              "subspace. SVD doesn't introduce compression where there "
              "wasn't any — it just makes explicit what's already there."),
        ],
    },
    {
        "kind": "figure",
        "id": "internal-compression",
        "chapter": T("§ 02.1 · El puente", "§ 02.1 · The bridge"),
        "title": T("El modelo se comprime <em>a sí mismo</em>",
                   "The model compresses <em>itself</em>"),
        "subtitle": T(
            "Antes de aplicar SVD, la red ya está reduciendo dimensión.",
            "Before SVD touches anything, the network is already "
            "reducing dimensionality."),
        "body": [
            T("Apilas las representaciones de tokens de cada capa en una "
              "matriz 469 × 768 y le aplicas SVD. La pregunta: cuántas "
              "dimensiones está usando realmente el modelo capa a capa.",
              "Stack each layer's token representations into a 469 × 768 "
              "matrix and run SVD on it. The question: how many "
              "dimensions does the model actually use at each layer?"),
            T("Las curvas azules son norma euclídea de los vectores. "
              "Crecen con la profundidad (Kobayashi 2021). Las rojas son "
              "rango efectivo y k95 — cuántos valores singulares cubren "
              "el 95 % de la varianza. Caen en pico: de ~130 dimensiones "
              "en capas tempranas a 22 en L12. El k95 baja hasta 35.",
              "The blue curves are vector L2 norms. They grow with depth "
              "(Kobayashi 2021). The red ones are effective rank and "
              "k95 — how many singular values cover 95 % of the variance. "
              "They collapse: ~130 dimensions in early layers down to "
              "22 by L12. k95 drops to 35."),
            T("De 768 dimensiones disponibles, el modelo termina usando "
              "un 5 %. El heatmap inferior es la prueba completa: "
              "energía espectral acumulada por capa, con la línea negra "
              "del k95 desplazándose a la izquierda en L9–L12.",
              "Out of 768 available dimensions, the model ends up using "
              "around 5 %. The bottom heatmap shows the full picture: "
              "cumulative spectral energy per layer, with the black k95 "
              "line drifting left from L9 to L12."),
        ],
        "pull": T(
            "Si la representación interna en L12 vive en 22 de 768 "
            "dimensiones, las matrices que la producen son aproximables "
            "por low-rank. La SVD no introduce compresión donde no la "
            "había. Materializa la que ya existe.",
            "If the internal representation at L12 lives in 22 of 768 "
            "dimensions, the matrices producing it are amenable to "
            "low-rank approximation. SVD doesn't introduce compression "
            "where there wasn't any — it just makes the existing "
            "compression explicit."),
        "figure": "internal_compression",
        "caption": T(
            "Norma media de hidden states (azul) vs rango efectivo y k95 "
            "(terra). Heatmap de energía espectral acumulada. Datos: 46 "
            "frases del test set sobre 23emo-final. Refs: Kobayashi EMNLP "
            "2021; Dong et al. ICML 2021.",
            "Mean hidden-state norm (blue) vs effective rank and k95 "
            "(terra). Heatmap of cumulative spectral energy. Data: 46 "
            "test-set sentences on 23emo-final. Refs: Kobayashi EMNLP "
            "2021; Dong et al. ICML 2021."),
        "fig_id": "02.1",
    },
    {
        "kind": "figure",
        "id": "pareto",
        "chapter": T("§ 02.2 · El acantilado", "§ 02.2 · The cliff"),
        "title": T("22 estrategias, una transición de fase",
                   "22 strategies, one phase transition"),
        "subtitle": T(
            "SVD uniforme a todas las capas. Entre rango 384 y 256 el F1 se desploma.",
            "Uniform SVD across every layer. Between rank 384 and 256, "
            "F1 falls off the cliff."),
        "body": [
            T("La superficie de la derecha es F1 sobre el plano "
              "(rango × profundidad). Cada celda es una configuración. "
              "El color es retención de F1.",
              "The right-hand surface is F1 over the (rank × depth) "
              "plane. Each cell is a configuration. Color is F1 "
              "retention."),
            T("Las capas tardías (8–11) caen al vacío antes que las "
              "tempranas. Esa asimetría es lo que motiva la compresión "
              "informada del capítulo 6.",
              "Late layers (8–11) plunge into the void before the early "
              "ones. That asymmetry is what motivates the informed "
              "compression of chapter 6."),
            T("El algoritmo greedy domina la frontera de Pareto en 8 de "
              "9 puntos óptimos. La estrategia greedy_90 retiene un 93 % "
              "del F1 con un 14 % menos de parámetros.",
              "The greedy algorithm dominates the Pareto frontier in 8 "
              "out of 9 optimal points. The greedy_90 strategy retains "
              "93 % of F1 with 14 % fewer parameters."),
        ],
        "figure": "pareto_3d",
        "caption": T(
            "Frontera de Pareto: 22 estrategias evaluadas. Datos: "
            "compression_comparison.csv. Eje rango uniforme, eje "
            "profundidad de capa, color F1 macro.",
            "Pareto frontier: 22 evaluated strategies. Data: "
            "compression_comparison.csv. Axes: uniform rank, layer "
            "depth, color F1 macro."),
        "fig_id": "02.2",
    },
    {
        "kind": "figure",
        "id": "spectral-landscape",
        "chapter": T("§ 02.3 · 3D", "§ 02.3 · 3D"),
        "title": T("La asimetría espectral, hecha topografía",
                   "Spectral asymmetry as a landscape"),
        "subtitle": T(
            "Las 72 matrices del modelo apiladas como filas de un terreno.",
            "All 72 weight matrices stacked as rows of a relief map."),
        "body": [
            T("Cada fila es una matriz de pesos. Eje X: índice del valor "
              "singular. Eje Z: magnitud normalizada σᵢ/σ₁.",
              "Each row is one weight matrix. X-axis: singular-value "
              "index. Z-axis: normalised magnitude σᵢ/σ₁."),
            T("Q y K forman picos abruptos: pocos valores singulares "
              "dominan, espectro concentrado, rango efectivo bajo, fácil "
              "de comprimir. Las FFN forman mesetas casi planas: "
              "espectro distribuido, cada dimensión aporta, frágiles "
              "bajo SVD.",
              "Q and K form sharp peaks: a few singular values dominate, "
              "concentrated spectrum, low effective rank, easy to "
              "compress. The FFN matrices form near-flat plateaus: "
              "distributed spectrum, every dimension contributes, "
              "fragile under SVD."),
            T("Los diamantes marcan k95 por matriz. Q/K cerca de 395. "
              "FFN cerca de 620. Es la Tabla 6 de la memoria, en "
              "relieve.",
              "The diamonds mark k95 per matrix. Q/K hover around 395. "
              "FFN around 620. It's Table 6 of the thesis, but in 3D."),
        ],
        "figure": "spectral_landscape",
        "caption": T(
            "SVD computada sobre el checkpoint 23emo-final. 72 matrices "
            "= 12 capas × 6 componentes (Q, K, V, Attn-O, FFN-i, FFN-o).",
            "SVD computed on the 23emo-final checkpoint. 72 matrices = "
            "12 layers × 6 components (Q, K, V, Attn-O, FFN-i, FFN-o)."),
        "fig_id": "02.3",
    },
    {
        "kind": "figure",
        "id": "decay",
        "chapter": T("§ 02.4 · Animación", "§ 02.4 · Animation"),
        "title": T("La galaxia se deshace", "The galaxy dissolves"),
        "subtitle": T(
            "Misma proyección que galaxy formation, ahora con SVD activa.",
            "Same projection as galaxy formation, now with SVD turned on."),
        "body": [
            T("Las mismas 588 frases en L12, proyectadas con LDA. "
              "Seis configuraciones de compresión: rangos 768 (sin "
              "tocar), 512, 384, 256, 128, 64.",
              "The same 588 sentences at L12, projected with LDA. Six "
              "compression configs: ranks 768 (untouched), 512, 384, "
              "256, 128, 64."),
            T("A r=512 todo casi igual. Entre r=384 y r=256 está el "
              "acantilado: los clusters se difuminan. A r=128 la "
              "geometría desaparece. A r=64 todos los embeddings "
              "colapsan en un blob.",
              "At r=512 almost nothing changes. Between r=384 and r=256 "
              "is the cliff: clusters blur. At r=128 the geometry is "
              "gone. At r=64 the embeddings collapse into a single blob."),
            T("La gráfica de la derecha lo cuantifica: silhouette en "
              "azul, retención de F1 en terra. Caen juntas a partir de "
              "r=384. La transición de fase, vista como geometría que "
              "se deshace.",
              "The right-hand chart quantifies the same story: "
              "silhouette in blue, F1 retention in terra. They drop "
              "together from r=384 onwards. Phase transition, seen as "
              "geometry coming apart."),
        ],
        "figure": "compression_decay",
        "caption": T(
            "588 frases del test set, L12, LDA-3D fija. SVD aplicada "
            "uniformemente. Slider o Play.",
            "588 test-set sentences, L12, fixed LDA-3D. SVD applied "
            "uniformly. Use the slider or Play."),
        "fig_id": "02.4",
    },
    {
        "kind": "figure",
        "id": "ft-diff",
        "chapter": T("§ 02.5 · Pre vs post", "§ 02.5 · Pre vs post"),
        "title": T("Qué cambió el fine-tuning",
                   "What the fine-tune actually changed"),
        "subtitle": T(
            "Diff Frobenius entre bert-base-uncased y 23emo-final.",
            "Frobenius diff between bert-base-uncased and 23emo-final."),
        "body": [
            T("Cargo los dos modelos. Para cada una de las 72 matrices "
              "calculo el cambio relativo: ‖W_ft − W_pre‖ / ‖W_pre‖. "
              "Cuánto se ha movido cada matriz durante el fine-tune.",
              "Load both models. For each of the 72 matrices, compute "
              "relative change ‖W_ft − W_pre‖ / ‖W_pre‖ — how far each "
              "matrix moved during the fine-tune."),
            T("La predicción de §5.5 era simple. El gradiente fluye "
              "más fuerte hacia las capas finales. Las tardías deberían "
              "cambiar mucho. Las tempranas, ya buenas en el "
              "pre-entrenamiento, deberían quedarse casi igual.",
              "The prediction in §5.5 was simple. Gradient flows harder "
              "into late layers. Late ones should change a lot; early "
              "ones, already serviceable from pre-training, should "
              "barely move."),
            T("El heatmap lo confirma. Los valores en capas 8–11 son "
              "claramente más altos. Evidencia empírica directa de la "
              "arquitectura de dos fases.",
              "The heatmap confirms it. Values across layers 8–11 are "
              "clearly higher. Direct empirical evidence of the "
              "two-phase architecture."),
        ],
        "figure": "finetuning_diff",
        "caption": T(
            "Norma Frobenius del cambio relativo, 72 matrices del "
            "encoder. bert-base-uncased vs 23emo-final.",
            "Frobenius norm of relative change, 72 encoder matrices. "
            "bert-base-uncased vs 23emo-final."),
        "fig_id": "02.5",
    },

    # ── PART 3: localizando emociones ───────────────────────────────────
    {
        "kind": "part",
        "id": "parte-3",
        "num": T("Parte 03", "Part 03"),
        "title": T("Dónde viven las emociones",
                   "Where the emotions live"),
        "intro": [
            T("Si las capas tardías son las que más cambian con el "
              "fine-tune y las que más rango efectivo pierden, también "
              "deberían ser donde se decide qué emoción detectar.",
              "If late layers are the ones that change most during the "
              "fine-tune and the ones losing the most effective rank, "
              "they should also be where the decision lives."),
            T("Probing por capa, geometría 3D, logit lens. Tres maneras "
              "distintas de decir lo mismo.",
              "Layer-wise probing, 3D geometry, logit lens. Three "
              "different ways of saying the same thing."),
        ],
    },
    {
        "kind": "figure",
        "id": "crystallization",
        "chapter": T("§ 03.1 · Probing", "§ 03.1 · Probing"),
        "title": T("Cristalización por capas", "Layer-wise crystallisation"),
        "subtitle": T(
            "Cuándo aparece cada emoción dentro del modelo.",
            "When each emotion shows up inside the model."),
        "body": [
            T("Para cada capa entrenas un clasificador lineal sobre el "
              "[CLS]. Te dice cuánta información de cada emoción es "
              "linealmente extraíble en ese punto.",
              "For each layer you train a linear classifier on top of "
              "[CLS]. It tells you how much of each emotion is linearly "
              "recoverable at that point."),
            T("Gratitude sale ya en L0. Tiene vocabulario obvio "
              "(\"thanks\", \"thank you\"). Realization aguanta hasta "
              "L11. Necesita contexto entero para diferenciarse.",
              "Gratitude shows up as early as L0. The vocabulary is "
              "obvious (\"thanks\", \"thank you\"). Realization holds "
              "out until L11. It needs the full context to be told apart."),
            T("La frecuencia en el dataset NO predice la profundidad. "
              "Annoyance tiene 3× más ejemplos que disgust. Cristaliza "
              "6 capas más tarde. Lo que importa es la complejidad "
              "semántica, no el volumen de datos.",
              "Frequency in the dataset does NOT predict depth. "
              "Annoyance has 3× more examples than disgust, yet "
              "crystallises 6 layers later. What matters is semantic "
              "complexity, not data volume."),
            T("Los diamantes marcan la capa de cristalización: donde el "
              "F1 alcanza el 80 % de su máximo. La barra de la izquierda "
              "es el cluster psicológico de cada emoción.",
              "The diamonds mark the crystallisation layer — where F1 "
              "reaches 80 % of its maximum. The left ribbon is each "
              "emotion's psychological cluster."),
        ],
        "figure": "crystallization",
        "caption": T(
            "Probing lineal por capa, 23 emociones × 13 capas. Datos: "
            "probe_results.csv del notebook 4.",
            "Layer-wise linear probing, 23 emotions × 13 layers. Data: "
            "probe_results.csv from notebook 4."),
        "fig_id": "03.1",
    },
    {
        "kind": "figure",
        "id": "galaxy",
        "chapter": T("§ 03.2 · Geometría", "§ 03.2 · Geometry"),
        "title": T("Galaxy formation", "Galaxy formation"),
        "subtitle": T(
            "23 emociones cristalizando en el espacio LDA, capa por capa.",
            "23 emotions crystallising in LDA space, layer by layer."),
        "body": [
            T("Tomamos el [CLS] de cada capa, le aplicamos el pooler "
              "real del modelo (Linear + tanh) y proyectamos con LDA "
              "supervisada ajustada en L12. Los tres ejes son las "
              "direcciones que mejor separan las 23 emociones.",
              "Take the [CLS] from each layer, apply the model's real "
              "pooler (Linear + tanh), and project with supervised LDA "
              "fitted at L12. The three axes are the directions that "
              "best separate the 23 emotions."),
            T("En L0 los 23 diamantes (los centroides de cada emoción) "
              "están casi superpuestos en el origen. En L11 ocupan "
              "regiones separadas, cerca de las frases que les "
              "corresponden.",
              "At L0 the 23 diamonds (one centroid per emotion) sit "
              "almost on top of the origin. By L11 they occupy distinct "
              "regions, next to the sentences they belong to."),
            T("Cuantitativo: separation ratio de 4,3 (los centroides "
              "están 4× más separados que la dispersión interna). 40 % "
              "de accuracy por nearest-centroid contra 4 % de un baseline "
              "aleatorio.",
              "Quantitative: separation ratio of 4.3 (centroids are 4× "
              "farther apart than the within-cluster spread). "
              "Nearest-centroid accuracy of 40 % vs a 4 % random "
              "baseline."),
            T("Click en la leyenda para aislar una emoción.",
              "Click an emotion in the legend to isolate it."),
        ],
        "figure": "galaxy_formation",
        "caption": T(
            "2300 frases del test set. Pooler aplicado, LDA-3D fija "
            "ajustada en L12. Mismas coordenadas para todas las capas.",
            "2,300 test-set sentences. Pooler applied, fixed LDA-3D "
            "fitted at L12. Same coordinates across every layer."),
        "fig_id": "03.2",
    },
    {
        "kind": "figure",
        "id": "iterative",
        "chapter": T("§ 03.3 · Logit lens", "§ 03.3 · Logit lens"),
        "title": T("La curva en U", "The U curve"),
        "subtitle": T(
            "Aplicar el classifier real a cada capa, no sólo a la última.",
            "Applying the real classifier to every layer, not just the last."),
        "body": [
            T("Logit lens. Técnica de Nostalgebraist (2020), refinada "
              "por Belrose et al. (NeurIPS 2023). Aplicas el pooler + "
              "classifier reales del modelo a las 13 representaciones "
              "intermedias.",
              "Logit lens. Originally Nostalgebraist (2020), refined by "
              "Belrose et al. (NeurIPS 2023). You take the model's real "
              "pooler + classifier and apply them to all 13 intermediate "
              "representations."),
            T("Promediado sobre 2300 frases, el sigmoid medio traza "
              "una U. Capas 0–3: las estadísticas del [CLS] saturan el "
              "tanh del pooler — muchas emociones se disparan a la vez "
              "con magnitud media. Capas 4–9: transición, tanh cerca de "
              "cero, todas las sigmoides colapsan. Capas 10–11: el [CLS] "
              "alcanza su régimen entrenado y una emoción pega un salto.",
              "Averaged over 2,300 sentences, the mean sigmoid traces a "
              "U. Layers 0–3: [CLS] statistics saturate the pooler's "
              "tanh — several emotions fire at once with medium "
              "magnitude. Layers 4–9: transition zone, tanh near zero, "
              "every sigmoid collapses. Layers 10–11: [CLS] reaches its "
              "trained regime and one emotion jumps."),
            T("Esto explica por qué el activation patching de L11 "
              "recupera el 100 % del F1: lo que restauras es justamente "
              "esta calibración.",
              "This is why activation-patching layer 11 alone recovers "
              "100 % of F1: what you're restoring is exactly this "
              "calibration."),
        ],
        "figure": "iterative_inference",
        "caption": T(
            "Sigmoid promedio del pooler+classifier aplicado por capa. "
            "Top-1, gold, suma de las 23. 2300 frases.",
            "Mean sigmoid of pooler+classifier applied per layer. "
            "Top-1, gold, sum of all 23. 2,300 sentences."),
        "fig_id": "03.3",
    },
    {
        "kind": "figure",
        "id": "lens-vs-probe",
        "chapter": T("§ 03.4 · Comparación", "§ 03.4 · Comparison"),
        "title": T("Lo que sabe vs lo que <em>sabe leer</em>",
                   "What it knows vs what it can <em>read</em>"),
        "subtitle": T(
            "Probing y logit lens contestan preguntas distintas. La "
            "diferencia es informativa.",
            "Probing and logit lens answer different questions. The "
            "gap between them is informative."),
        "body": [
            T("Dos curvas, tres paneles. Las dos miden información de "
              "emoción en cada capa. Mismos datos. Preguntas distintas.",
              "Two curves, three panels. Both measure emotion "
              "information at every layer. Same data. Different "
              "questions."),
            T("El probe lineal (rojo) es un clasificador nuevo "
              "entrenado sobre cada capa. Te dice cuánta información hay "
              "ahí, linealmente extraíble. Sube monótonamente: la "
              "información se acumula con la profundidad.",
              "The linear probe (red) is a fresh classifier trained on "
              "each layer. It tells you how much information is there, "
              "linearly recoverable. It rises monotonically: information "
              "accumulates with depth."),
            T("El logit lens (verde) es la cabeza entrenada en L11, "
              "aplicada en cada capa. Hace U porque sólo sabe leer "
              "activaciones estilo L11. En las capas tempranas y medias "
              "lee mal aunque la información esté ahí.",
              "The logit lens (green) is the head trained at L11, "
              "applied at each layer. It makes a U because it can only "
              "read L11-style activations. Early and middle layers fool "
              "it even though the information is there."),
            T("La banda gris entre las dos curvas es la diferencia: "
              "información que <strong>existe pero el modelo no usa</"
              "strong>. Por eso basta con restaurar L11 en el activation "
              "patching para recuperar el 100 % del F1: no falta "
              "información en el resto del modelo, falta una cabeza que "
              "sepa leerla.",
              "The grey band between the two curves is the gap: "
              "information that <strong>exists but the model doesn't "
              "use</strong>. That's why restoring just L11 via "
              "activation patching recovers 100 % of F1 — the rest of "
              "the model isn't missing information, it's missing a head "
              "that knows how to read it."),
            T("Los tres paneles separan emociones por capa de "
              "cristalización. Tempranas: probe alto en L0 pero el lens "
              "aún no decide, brecha máxima. Tardías: probe y lens "
              "suben a la vez, brecha mínima.",
              "The three panels split emotions by crystallisation "
              "layer. Early: probe is already high at L0 but the lens "
              "hasn't decided — maximum gap. Late: probe and lens rise "
              "together — minimum gap."),
        ],
        "figure": "lens_vs_probe",
        "caption": T(
            "Probe F1 macro de notebook 4 contra logit lens "
            "(pooler+classifier reales aplicados al [CLS] de cada capa) "
            "sobre 2300 frases del test set. Bandas de fase compartidas "
            "con la curva U.",
            "Probe F1 macro from notebook 4 vs logit lens (real "
            "pooler+classifier applied to [CLS] at each layer) over "
            "2,300 test-set sentences. Phase bands shared with the U "
            "curve."),
        "fig_id": "03.4",
    },
    {
        "kind": "figure",
        "id": "fingerprint",
        "chapter": T("§ 03.5 · Multi-label", "§ 03.5 · Multi-label"),
        "title": T("Decision fingerprint", "Decision fingerprint"),
        "subtitle": T(
            "Las métricas agregadas esconden la firma multi-label de cada frase.",
            "Aggregate metrics hide the multi-label signature of each sentence."),
        "body": [
            T("Multi-label significa que el modelo no produce una "
              "emoción. Produce un vector de 23 sigmoides, varias "
              "activas a la vez. Ese vector es la firma de la frase.",
              "Multi-label means the model doesn't produce an emotion. "
              "It produces a vector of 23 sigmoids, several of them "
              "active at once. That vector is the sentence's signature."),
            T("Aplico el classifier real al [CLS] de cada capa. Cada "
              "pétalo del polar es una emoción. Su longitud es el "
              "sigmoid. El pétalo con borde negro grueso es la emoción "
              "gold.",
              "I apply the real classifier to [CLS] at each layer. Each "
              "petal of the polar is one emotion. Its length is the "
              "sigmoid. The petal with the thick black border is the "
              "gold emotion."),
            T("Selecciona una frase y dale a Play. Vas viendo el "
              "fingerprint emerger desde el ruido (Emb) hasta la "
              "decisión cristalizada (L11). Cuando varios pétalos "
              "crecen juntos, el modelo cree que coexisten varias "
              "emociones — eso es lo que la BCE multi-label produce y "
              "ningún heatmap promediado deja ver.",
              "Pick a sentence and hit Play. You watch the fingerprint "
              "emerge from noise (Emb) into a crystallised decision "
              "(L11). When several petals grow together, the model "
              "thinks multiple emotions coexist — that's what multi-"
              "label BCE produces and no averaged heatmap can show you."),
        ],
        "figure": "decision_fingerprint",
        "caption": T(
            "Sigmoid de las 23 emociones por capa, vector polar. "
            "Aplicado al pooler+classifier real.",
            "Sigmoid of all 23 emotions per layer, polar vector. "
            "Applied with the real pooler+classifier."),
        "fig_id": "03.5",
    },

    # ── PART 4: atención ────────────────────────────────────────────────
    {
        "kind": "part",
        "id": "parte-4",
        "num": T("Parte 04", "Part 04"),
        "title": T("144 cabezas", "144 heads"),
        "intro": [
            T("En cada capa hay 12 cabezas de atención. 12 × 12 = 144 "
              "en total. Cada una con su propio patrón y su propia "
              "función.",
              "Each layer has 12 attention heads. 12 × 12 = 144 in "
              "total. Each one with its own pattern and its own role."),
            T("No todas son iguales. No todas son críticas. No todas "
              "hacen lo mismo.",
              "Not all are equal. Not all are critical. Not all do "
              "the same job."),
        ],
    },
    {
        "kind": "figure",
        "id": "heads",
        "chapter": T("§ 04.1 · Categorías", "§ 04.1 · Categories"),
        "title": T("Las 144 cabezas", "The 144 heads"),
        "subtitle": T(
            "Critical specialist, critical generalist, minor, prescindible.",
            "Critical specialist, critical generalist, minor, dispensable."),
        "body": [
            T("Cada cabeza es una celda en una matriz 12 × 12. Color = "
              "categoría según el ablation study. La capa 11 (fila "
              "inferior) no contiene ni una cabeza prescindible. Sus 12 "
              "son críticas.",
              "Each head is a cell in a 12 × 12 matrix. Color = "
              "category from the ablation study. Layer 11 (bottom row) "
              "has zero dispensable heads. All 12 are critical."),
            T("Los puntos negros marcan las cabezas que cada emoción "
              "necesita específicamente, según la Tabla 19. L11-H6 "
              "aparece dos veces — la comparten sadness y realization.",
              "The black dots mark the heads each emotion needs "
              "specifically, from Table 19. L11-H6 appears twice — "
              "shared between sadness and realization."),
            T("El 77 % de las cabezas en capas 8–11 son críticas. En "
              "capas 0–4 es sólo el 25 %. Hay un gradiente que no "
              "aparece en arquitecturas no fine-tuneadas.",
              "77 % of heads in layers 8–11 are critical. In layers 0–4 "
              "it's only 25 %. A gradient that doesn't show up in "
              "non-fine-tuned architectures."),
        ],
        "figure": "heads_matrix",
        "caption": T(
            "12 capas × 12 cabezas = 144 celdas. Datos: "
            "head_categories.csv del notebook 6.",
            "12 layers × 12 heads = 144 cells. Data: "
            "head_categories.csv from notebook 6."),
        "fig_id": "04.1",
    },
    {
        "kind": "figure",
        "id": "atlas",
        "chapter": T("§ 04.2 · Patrones", "§ 04.2 · Patterns"),
        "title": T("Attention atlas", "Attention atlas"),
        "subtitle": T(
            "Las 144 cabezas, todas a la vez, sobre una frase.",
            "All 144 heads, simultaneously, over one sentence."),
        "body": [
            T("Eliges una frase. El sistema te enseña los 144 patrones "
              "de atención simultáneamente. El borde de cada celda "
              "codifica su categoría funcional.",
              "Pick a sentence. The system shows you all 144 attention "
              "patterns at once. Each cell's border encodes its "
              "functional category."),
            T("Patrones reconocibles. Las capas tempranas atienden "
              "diagonalmente: token a sí mismo, vecinos. Las tardías "
              "concentran atención en [CLS] o [SEP] — rayas verticales. "
              "Es el patrón típico de las cabezas agregadoras.",
              "Recognisable shapes. Early layers attend diagonally: "
              "token to itself, neighbours. Late layers concentrate "
              "attention on [CLS] or [SEP] — vertical stripes. The "
              "classic aggregator pattern."),
            T("Click en cualquier cabeza para ampliarla con etiquetas "
              "de tokens.",
              "Click any head to enlarge it with token labels."),
        ],
        "figure": "attention_atlas",
        "caption": T(
            "Pesos de atención reales sobre 46 frases del test set. 12 "
            "capas × 12 cabezas = 144 mini-mapas.",
            "Real attention weights over 46 test-set sentences. 12 "
            "layers × 12 heads = 144 mini-maps."),
        "fig_id": "04.2",
    },
    {
        "kind": "figure",
        "id": "constellations",
        "chapter": T("§ 04.3 · Superposition", "§ 04.3 · Superposition"),
        "title": T("Las 23 emociones como direcciones",
                   "The 23 emotions as directions"),
        "subtitle": T(
            "Los vectores de peso del classifier, proyectados a 3D.",
            "The classifier's weight vectors, projected into 3D."),
        "body": [
            T("El classifier tiene 23 vectores de 768 dimensiones, uno "
              "por emoción. Esos vectores son las direcciones que el "
              "modelo usa para detectar cada emoción. Los proyecto a "
              "PCA-3D y los pinto como flechas radiando del origen.",
              "The classifier has 23 vectors of 768 dimensions, one "
              "per emotion. Those vectors are the directions the model "
              "uses to detect each emotion. I project them into PCA-3D "
              "and draw them as arrows from the origin."),
            T("Vectores casi-ortogonales: emociones que el modelo "
              "distingue limpiamente. Casi-paralelos: emociones que "
              "confunde. Gratitude y love apuntan parecido.",
              "Near-orthogonal vectors: emotions the model "
              "distinguishes cleanly. Near-parallel: emotions it "
              "confuses. Gratitude and love point in similar "
              "directions."),
            T("El heatmap es la similitud coseno completa en 768 "
              "dimensiones, reordenada por cluster. Los bloques "
              "diagonales muestran alta similitud dentro de un cluster.",
              "The heatmap is the full cosine similarity in 768 "
              "dimensions, reordered by cluster. Diagonal blocks show "
              "high similarity within a cluster."),
            T("Esto es §2.7.5 (superposition) hecho geometría. El "
              "modelo codifica más conceptos que dimensiones tendría si "
              "todos fueran ortogonales puros.",
              "This is §2.7.5 (superposition) made geometric. The "
              "model encodes more concepts than dimensions would allow "
              "if they were all strictly orthogonal."),
        ],
        "figure": "probe_constellations",
        "caption": T(
            "23 vectores del classifier proyectados con PCA fija. "
            "Heatmap de cosine similarity en 768 dimensiones, "
            "reordenado por cluster.",
            "23 classifier vectors projected with fixed PCA. Cosine-"
            "similarity heatmap in 768 dimensions, reordered by cluster."),
        "fig_id": "04.3",
    },
    {
        "kind": "figure",
        "id": "circuit",
        "chapter": T("§ 04.4 · Circuitos", "§ 04.4 · Circuits"),
        "title": T("El circuito compartido", "The shared circuit"),
        "subtitle": T(
            "Cuando dos emociones reutilizan la misma maquinaria.",
            "When two emotions reuse the same machinery."),
        "body": [
            T("Aquí el grafo se construye en D3.js. Sin Plotly. Cada "
              "emoción a la izquierda. Cabezas críticas en el centro. "
              "Clusters psicológicos a la derecha.",
              "Here the graph is built in D3.js. No Plotly. Emotions on "
              "the left. Critical heads in the middle. Psychological "
              "clusters on the right."),
            T("L11-H6 es el nodo rojo. Es la única cabeza compartida "
              "por más de una emoción. La usan sadness y realization. "
              "Probable circuito de \"expectativa frustrada\" que el "
              "fine-tune reutiliza.",
              "L11-H6 is the red node. The only head shared by more "
              "than one emotion. Used by sadness and realization. "
              "Likely a \"frustrated expectation\" circuit the "
              "fine-tune is reusing."),
            T("Click en cualquier nodo para ver su entorno.",
              "Click any node to inspect its neighbourhood."),
        ],
        "figure": "circuit_network",
        "caption": T(
            "Grafo construido a partir de top_heads_per_emotion.csv. "
            "Nodos en rojo: cabezas compartidas.",
            "Graph built from top_heads_per_emotion.csv. Red nodes: "
            "shared heads."),
        "fig_id": "04.4",
    },

    # ── PART 5: causalidad y neuronas ───────────────────────────────────
    {
        "kind": "part",
        "id": "parte-5",
        "num": T("Parte 05", "Part 05"),
        "title": T("Causalidad", "Causality"),
        "intro": [
            T("Hasta aquí, correlación. Saber que el F1 sube en una "
              "capa no demuestra que esa capa cause la decisión. Para "
              "eso, lesionar.",
              "Up to here, correlation. Knowing F1 rises at a layer "
              "doesn't prove that layer causes the decision. To prove "
              "it, you have to lesion."),
            T("Apagar una capa. Apagar una neurona. Ver qué se rompe.",
              "Turn off a layer. Turn off a neuron. See what breaks."),
        ],
    },
    {
        "kind": "figure",
        "id": "lesion",
        "chapter": T("§ 05.1 · Lesion", "§ 05.1 · Lesion"),
        "title": T("Lesion theater", "Lesion theatre"),
        "subtitle": T(
            "Restaurar capa por capa, ver el modelo revivir.",
            "Restore the model layer by layer and watch it come back."),
        "body": [
            T("Empezamos con un modelo a F1 = 0. Todas las barras a "
              "cero. Cada etapa restaura los pesos originales de UNA "
              "capa más. 12 etapas en total.",
              "Start with a model at F1 = 0. Every bar at zero. Each "
              "stage restores the original weights of one more layer. "
              "12 stages in total."),
            T("Las primeras 8 etapas no mueven nada. L8 enciende un "
              "destello. L9 empuja arriba a las emociones léxicas. L10 "
              "recupera la mitad. L11 hace explotar todas las barras al "
              "baseline simultáneamente.",
              "The first 8 stages move nothing. L8 sets off a flicker. "
              "L9 lifts the lexical emotions. L10 recovers half. L11 "
              "blows every bar back to baseline simultaneously."),
            T("La capacidad emocional del modelo no está distribuida. "
              "Vive concentrada en las capas finales.",
              "The model's emotional capacity isn't distributed. It "
              "lives concentrated in the final layers."),
        ],
        "figure": "lesion_theater",
        "caption": T(
            "Activation patching secuencial por capa. F1 macro y por "
            "emoción. 12 etapas + estado inicial.",
            "Sequential layer-wise activation patching. F1 macro and "
            "per emotion. 12 stages plus initial state."),
        "fig_id": "05.1",
    },

    # ── PART 6: síntesis ─────────────────────────────────────────────────
    {
        "kind": "part",
        "id": "parte-6",
        "num": T("Parte 06", "Part 06"),
        "title": T("Síntesis", "Synthesis"),
        "intro": [
            T("Una taxonomía emocional emergente, una compresión "
              "informada por interpretabilidad, y una vista que "
              "junta todo.",
              "An emergent emotional taxonomy, a compression informed "
              "by interpretability, and a view that brings it all "
              "together."),
        ],
    },
    {
        "kind": "figure",
        "id": "clusters",
        "chapter": T("§ 06.1 · Taxonomía", "§ 06.1 · Taxonomy"),
        "title": T("Seis clusters que aparecen solos",
                   "Six clusters emerging on their own"),
        "subtitle": T(
            "El modelo redescubre la psicología sin que se la impongan.",
            "The model rediscovers psychology without being told to."),
        "body": [
            T("Sin pedirle nada, un clustering jerárquico sobre los "
              "vectores de selectividad neuronal produce seis grupos "
              "con coherencia psicológica reconocible. Positivas "
              "energéticas. Negativas reactivas. Internas. Epistémicas. "
              "Orientadas al otro. Baja especificidad.",
              "Without asking for anything, hierarchical clustering on "
              "the neural-selectivity vectors yields six groups with "
              "recognisable psychological coherence. High-energy "
              "positives. Reactive negatives. Internal. Epistemic. "
              "Other-oriented. Low specificity."),
            T("La barra de la derecha mide la norma de selectividad. "
              "Es el mejor predictor (ρ = 0.64, p = 0.001) de la caída "
              "de F1 bajo SVD. Las emociones \"escritas en negrita\" "
              "en los pesos del modelo son las más vulnerables.",
              "The right-hand bar shows the selectivity norm. Best "
              "predictor (ρ = 0.64, p = 0.001) of F1 drop under SVD. "
              "Emotions \"written in bold\" in the model's weights are "
              "the most fragile."),
            T("La SVD no ataca selectivamente las neuronas emocionales. "
              "La geometría espectral es ortogonal a la función. Pero "
              "las emociones que requieren más capacidad neuronal son "
              "más frágiles a CUALQUIER perturbación.",
              "SVD doesn't selectively target emotional neurons. The "
              "spectral geometry is orthogonal to function. But the "
              "emotions that demand more neural capacity are more "
              "fragile under ANY perturbation."),
        ],
        "figure": "sunburst",
        "caption": T(
            "Sunburst con 6 clusters, 23 emociones. Áreas proporcionales "
            "a frecuencia en train. Bibliografía: Russell 1980 "
            "(circumplex).",
            "Sunburst with 6 clusters, 23 emotions. Areas proportional "
            "to train-set frequency. Reference: Russell 1980 "
            "(circumplex)."),
        "fig_id": "06.1",
    },
    {
        "kind": "figure",
        "id": "landscape",
        "chapter": T("§ 06.2 · Mapa", "§ 06.2 · Map"),
        "title": T("El paisaje emocional", "The emotional landscape"),
        "subtitle": T(
            "Cada emoción en (cristalización × intensidad).",
            "Each emotion plotted in (crystallisation × intensity)."),
        "body": [
            T("Las 23 emociones, posicionadas en un plano 2D. Eje X: "
              "capa de cristalización. Eje Y: norma de selectividad.",
              "The 23 emotions placed in a 2D plane. X-axis: "
              "crystallisation layer. Y-axis: selectivity norm."),
            T("Cuadrante superior izquierdo: gratitude, love. Tempranas "
              "e intensas. Cuadrante inferior derecho: realization, "
              "disappointment. Tardías y difusas.",
              "Top-left quadrant: gratitude, love. Early and intense. "
              "Bottom-right quadrant: realization, disappointment. Late "
              "and diffuse."),
            T("La línea punteada conecta sadness y realization. "
              "Comparten L11-H6.",
              "The dotted line connects sadness and realization. They "
              "share L11-H6."),
            T("Selecciona una emoción en el menú de la derecha y verás "
              "su huella radial: 6 dimensiones funcionales en una "
              "figura polar.",
              "Pick an emotion in the right-hand menu to see its radial "
              "fingerprint — 6 functional dimensions on a polar plot."),
        ],
        "figure": "emotional_landscape",
        "caption": T(
            "23 emociones sobre el plano (cristalización × norma de "
            "selectividad). Datos: crystallization_layers.csv y "
            "neuron_catalog.csv.",
            "23 emotions on the (crystallisation × selectivity-norm) "
            "plane. Data: crystallization_layers.csv and "
            "neuron_catalog.csv."),
        "fig_id": "06.2",
    },
    {
        "kind": "figure",
        "id": "trajectory",
        "chapter": T("§ 06.3 · Síntesis", "§ 06.3 · Synthesis"),
        "title": T("Una frase, cuatro vistas",
                   "One sentence, four views"),
        "subtitle": T(
            "El experimento que se pide al inicio: ver BERT pensar.",
            "The experiment you ask for at the start: watching BERT "
            "think."),
        "body": [
            T("Aquí se juntan todas las vistas en una sola experiencia. "
              "Eliges una frase. Arrastras el slider de capa. Los "
              "cuatro paneles se mueven sincronizados. Trayectoria 3D "
              "del [CLS]. Atención de las 3 cabezas más críticas. "
              "Sigmoides multi-label. Curva del gold.",
              "All the previous views collapse into a single experience. "
              "Pick a sentence. Drag the layer slider. The four panels "
              "move in sync. 3D trajectory of [CLS]. Attention of the "
              "3 most critical heads. Multi-label sigmoids. Gold curve."),
            T("Pulsa Play. El [CLS] arranca cerca del origen y se "
              "desplaza hacia su centroide. Las cabezas críticas se "
              "activan capa a capa. Los pétalos saltan del valle a la "
              "cristalización. La curva del gold traza la U.",
              "Hit Play. [CLS] starts near the origin and slides toward "
              "its centroid. The critical heads activate layer by "
              "layer. The petals jump from valley to crystallisation. "
              "The gold curve traces the U."),
        ],
        "figure": "sentence_trajectory",
        "caption": T(
            "Cuatro paneles síncronos. Datos reales del modelo "
            "23emo-final aplicado en vivo a la frase elegida.",
            "Four synchronised panels. Real data from the 23emo-final "
            "model applied live to the selected sentence."),
        "fig_id": "06.3",
    },
    {
        "kind": "figure",
        "id": "greedy",
        "chapter": T("§ 06.4 · Algoritmo", "§ 06.4 · Algorithm"),
        "title": T("Greedy en acción", "Greedy in action"),
        "subtitle": T(
            "Cómo el algoritmo construye la compresión paso a paso.",
            "How the algorithm builds the compression step by step."),
        "body": [
            T("El greedy elige movimientos por eficiencia: parámetros "
              "ahorrados / coste F1. Aquí lo ves en acción. Empezamos "
              "con baseline (todo a rango 768) y avanzamos: greedy_95 "
              "→ greedy_90 → … → greedy_50.",
              "Greedy picks moves by efficiency: parameters saved / F1 "
              "cost. Here you watch it work. Start at baseline (every "
              "rank 768) and step through: greedy_95 → greedy_90 → … "
              "→ greedy_50."),
            T("La matriz 12 × 6 se va iluminando célula a célula. Las "
              "primeras decisiones son Q y K — gratis, sin coste F1. "
              "Exactamente lo que predice §4.3 sobre la inmunidad de "
              "Q/K. Después vienen FFN-output en capas tempranas. Las "
              "tardías (8–11) se mantienen intactas hasta el final.",
              "The 12 × 6 matrix lights up cell by cell. The first "
              "decisions are Q and K — free, zero F1 cost. Exactly "
              "what §4.3 predicts about Q/K immunity. Then FFN-output "
              "in early layers. The late layers (8–11) stay untouched "
              "until the very end."),
            T("La línea derecha sigue el F1 vs ratio de compresión "
              "paso a paso. Es la prueba algorítmica de que el greedy "
              "reproduce los hallazgos de interpretabilidad sin acceso "
              "a ellos. Se entera sólo con datos empíricos de "
              "sensibilidad.",
              "The right-hand line tracks F1 vs compression ratio at "
              "each step. Algorithmic proof that greedy reproduces the "
              "interpretability findings without access to them. It "
              "figures it out from empirical sensitivity data alone."),
        ],
        "figure": "greedy_replay",
        "caption": T(
            "Replay del algoritmo greedy_50, _60, …, _95. Datos: "
            "greedy_*_ranks.csv del notebook 9.",
            "Replay of the greedy_50, _60, …, _95 algorithm. Data: "
            "greedy_*_ranks.csv from notebook 9."),
        "fig_id": "06.4",
    },
]


# ── Hero stats ────────────────────────────────────────────────────────────
HERO_STATS = [
    ("12 × 144",  T("capas × cabezas", "layers × heads")),
    ("36 864",    T("neuronas FFN", "FFN neurons")),
    ("23",        T("emociones, multi-label", "emotions, multi-label")),
    ("21",        T("visualizaciones", "visualisations")),
]


# ── Hero text ─────────────────────────────────────────────────────────────
HERO = {
    "tag":      T("Trabajo de fin de grado · 2026",
                  "Bachelor's thesis · 2026"),
    "title_1":  T("Anatomía", "Anatomy"),
    "title_2":  T("emocional", "of an"),
    "title_3":  T("de un transformer", "emotional transformer"),
    "lead":     T(
        "Compresión selectiva e interpretabilidad mecánica de BERT-base "
        "sobre GoEmotions. Un manual visual para desarmar el modelo capa "
        "por capa.",
        "Selective compression and mechanistic interpretability of "
        "BERT-base on GoEmotions. A visual manual that takes the model "
        "apart layer by layer."),
    "scroll":   T("Scroll · Empezar el recorrido",
                  "Scroll · Begin the tour"),
    "left":     T("TFG · 2026<br>Universitat Politècnica<br>de Catalunya · FIB",
                  "Thesis · 2026<br>Universitat Politècnica<br>de Catalunya · FIB"),
    "right":    T("Vol. 01<br>Compresión selectiva<br>Interpretabilidad mecánica",
                  "Vol. 01<br>Selective compression<br>Mechanistic interpretability"),
    "director": T("Director: Lluís Padró Cirera",
                  "Advisor: Lluís Padró Cirera"),
}


# ── Outro / footer / nav text ─────────────────────────────────────────────
OUTRO = {
    "label":    T("Cierre", "Closing"),
    "title":    T("Cómo está hecho", "How it's built"),
    "sub":      T("Stack, datos y créditos.",
                  "Stack, data, credits."),
    "p1": T(
        "Datos numéricos. Las visualizaciones se alimentan directamente "
        "de los CSVs de los notebooks 2 al 9. Resultados reales del "
        "fine-tune sobre BERT-base-uncased y 23 emociones de GoEmotions. "
        "61 tablas exportadas: probing por capa, ablación de 144 "
        "cabezas, especialización neuronal, activation patching, "
        "frontera de Pareto completa con 22 estrategias evaluadas.",
        "Numbers. The visualisations read directly from the CSVs of "
        "notebooks 2–9. Real results from the fine-tune of "
        "bert-base-uncased on the 23 GoEmotions classes. 61 exported "
        "tables: layer-wise probing, 144-head ablation, neural "
        "specialisation, activation patching, full Pareto frontier with "
        "22 evaluated strategies."),
    "p2": T(
        "Activaciones y geometría. Galaxy formation, sentence "
        "trajectory, compression decay y spectral flowers se computan "
        "ejecutando el checkpoint <code>23emo-final</code> (109,5 M "
        "parámetros, 23 emociones) sobre frases del test set. Pooler y "
        "classifier reales aplicados en cada paso. LDA fija ajustada en "
        "L12 para coordenadas consistentes capa a capa.",
        "Activations and geometry. Galaxy formation, sentence "
        "trajectory, compression decay and spectral flowers are "
        "computed by running the <code>23emo-final</code> checkpoint "
        "(109.5 M parameters, 23 emotions) on test-set sentences. The "
        "real pooler and classifier are applied at every step. Fixed "
        "LDA fitted at L12 for consistent layer-by-layer coordinates."),
    "p3": T(
        "Visualización. 21 piezas: 14 estáticas en Plotly, 6 "
        "interactivas con HTML+JS custom, 1 grafo en D3.js puro. Cada "
        "figura es un HTML autocontenido. La página que las une es "
        "static HTML+CSS+JS. Sin frameworks, sin servidor, sin backend.",
        "Visualisation. 21 pieces: 14 static Plotly figures, 6 custom "
        "HTML+JS, 1 pure D3.js graph. Each figure is a self-contained "
        "HTML. The page that ties them together is static HTML+CSS+JS. "
        "No frameworks, no server, no backend."),
    "p4": T(
        "Memoria. Guido Biosca Lasa. Director: Lluís Padró Cirera. "
        "FIB-UPC, 2026. <a href=\"sobre.html\">Más sobre el proyecto</a>.",
        "Thesis. Guido Biosca Lasa. Advisor: Lluís Padró Cirera. "
        "FIB-UPC, 2026. <a href=\"sobre.html\">More about the project</a>."),
}


NAV = {
    "brand":    T("Anatomía Emocional", "Emotional Anatomy"),
    "chapters": T("Capítulos", "Chapters"),
    "data":     T("Datos", "Data"),
    "about":    T("Sobre", "About"),
    "index":    T("Índice", "Index"),
}


FOOTER = {
    "brand_title":    T("Anatomía emocional<br><em>de un transformer.</em>",
                        "Anatomy of an<br><em>emotional transformer.</em>"),
    "brand_sub":      T("Trabajo de fin de grado.", "Bachelor's thesis."),
    "brand_school":   T("Facultat d'Informàtica de Barcelona · 2026.",
                        "Facultat d'Informàtica de Barcelona · 2026."),
    "project":        T("Proyecto", "Project"),
    "project_repo":   T("Repositorio", "Repository"),
    "project_pdf":    T("Memoria · PDF", "Thesis · PDF"),
    "project_about":  T("Cómo está hecho", "How it's built"),
    "author":         T("Autor", "Author"),
    "tribunal":       T("Tribunal", "Committee"),
    "director":       T("Director: Lluís Padró Cirera",
                        "Advisor: Lluís Padró Cirera"),
    "stack":          T("Built with Python · Plotly · D3 · no frameworks",
                        "Built with Python · Plotly · D3 · no frameworks"),
}


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
    "compression_decay":     820,
    "finetuning_diff":       920,
    "crystallization":       780,
    "galaxy_formation":      880,
    "iterative_inference":   560,
    "decision_fingerprint":  680,
    "heads_matrix":          740,
    "attention_atlas":      1320,
    "probe_constellations":  800,
    "circuit_network":       820,
    "lesion_theater":        900,
    "sunburst":              680,
    "emotional_landscape":   720,
    "sentence_trajectory":  1020,
    "greedy_replay":         820,
}
