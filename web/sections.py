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
        "title": T("El sujeto y las preguntas",
                   "The subject and the questions"),
        "intro": [
            T("BERT-base. 12 capas. 144 cabezas de atención. 36 864 "
              "neuronas en los bloques FFN. 109,5 millones de "
              "parámetros. Encima, un fine-tune sobre GoEmotions: "
              "23 emociones, etiquetado multi-label. F1 macro de "
              "0,577 sobre el conjunto de test — el baseline contra "
              "el que se mide todo.",
              "BERT-base. 12 layers. 144 attention heads. 36,864 "
              "neurons in the FFN blocks. 109.5 million parameters. "
              "On top of it, a fine-tune on GoEmotions: 23 emotions, "
              "multi-label tagging. F1 macro of 0.577 on the test "
              "set — the baseline everything is measured against."),
            T("Hay tres preguntas en juego. La primera es descriptiva: "
              "qué le pasa al modelo cuando le aplicas SVD a las "
              "matrices que lo componen y cuánto puedes quitar antes "
              "de romperlo. La segunda es explicativa: qué dice la "
              "interpretabilidad mecánica sobre dónde y cómo vive la "
              "información emocional dentro de la red. La tercera es "
              "prescriptiva: si juntas las dos respuestas, ¿se puede "
              "comprimir mejor que tirando rangos al tuntún?",
              "There are three questions at play. The first is "
              "descriptive: what happens when you apply SVD to the "
              "matrices that make up the model, and how much you can "
              "take away before it breaks. The second is explanatory: "
              "what mechanistic interpretability says about where and "
              "how emotional information lives inside the network. "
              "The third is prescriptive: if you combine the two "
              "answers, can you compress better than by picking ranks "
              "at random?"),
            T("La compresión funciona aquí como un bisturí "
              "experimental. Cada configuración comprimida es una "
              "intervención que mide la importancia funcional del "
              "componente afectado por la magnitud de lo que se "
              "rompe. Quitar partes del modelo y mirar qué se "
              "sobrevive es una manera empírica de hacer el mapa "
              "de cómo está organizado por dentro.",
              "Compression here works as an experimental scalpel. "
              "Every compressed configuration is an intervention that "
              "measures the functional importance of the affected "
              "component by the magnitude of what breaks. Removing "
              "parts of the model and watching what survives is an "
              "empirical way to map how the information is laid out "
              "inside it."),
            T("Antes de operar nada, primero hay que conocer al "
              "sujeto.",
              "Before cutting anything open, you have to meet the "
              "subject."),
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
        "id": "metodologia",
        "chapter": T("§ 01.2 · Pipeline", "§ 01.2 · Pipeline"),
        "title": T("El experimento <em>entero</em>",
                   "The <em>full</em> experiment"),
        "subtitle": T(
            "Mapa interactivo: dónde está cada cosa.",
            "Interactive map: where everything lives."),
        "body": [
            T("Antes de bajar a las figuras una a una, una vista de "
              "pájaro de qué se ha hecho. Tres bloques.",
              "Before diving into the figures one by one, a "
              "bird's-eye view of what was done. Three blocks."),
            T("<strong>Setup</strong>: GoEmotions filtrado a 23 "
              "emociones, BERT-base, fine-tune AdamW (LR 2e-5, 4 "
              "epochs, batch 32) → checkpoint <em>23emo-final</em>, "
              "F1 macro 0,577 sobre el test set. Es el baseline "
              "contra el que se mide TODO lo demás.",
              "<strong>Setup</strong>: GoEmotions filtered to 23 "
              "emotions, BERT-base, AdamW fine-tune (LR 2e-5, 4 "
              "epochs, batch 32) → checkpoint <em>23emo-final</em>, "
              "F1 macro 0.577 on the test set. It's the baseline "
              "EVERYTHING else is measured against."),
            T("<strong>Dos brazos paralelos</strong>: en uno, "
              "compresión SVD con cuatro familias de estrategias (22 "
              "estrategias evaluadas en total). En el otro, cinco "
              "técnicas de interpretabilidad mecánica con "
              "granularidad creciente: capa, componente, cabeza, "
              "neurona.",
              "<strong>Two parallel arms</strong>: on one side, SVD "
              "compression with four strategy families (22 evaluated "
              "strategies total). On the other, five mechanistic "
              "interpretability techniques at increasing granularity: "
              "layer, component, head, neuron."),
            T("<strong>Síntesis</strong>: los dos brazos confluyen en "
              "una compresión informada por datos empíricos de "
              "sensibilidad — el algoritmo greedy. Y un ciclo final "
              "de fine-tuning que recupera (y supera) el baseline "
              "con menos parámetros.",
              "<strong>Synthesis</strong>: the two arms converge into "
              "compression informed by empirical sensitivity data — "
              "the greedy algorithm. And one final fine-tuning loop "
              "that recovers (and surpasses) the baseline with fewer "
              "parameters."),
            T("Click en cualquier bloque del pipeline para saltar "
              "directamente a la figura correspondiente.",
              "Click any block in the pipeline to jump directly to "
              "the corresponding figure."),
        ],
        "figure": "methodology",
        "caption": T(
            "Pipeline experimental interactivo. Equivalente a la "
            "Figura 1 del Capítulo 3 de la memoria, con enlaces a "
            "cada sección.",
            "Interactive experimental pipeline. Equivalent to Figure "
            "1 of Chapter 3 of the thesis, with links to each "
            "section."),
        "fig_id": "01.2",
    },
    {
        "kind": "figure",
        "id": "lex2sem",
        "chapter": T("§ 01.3 · Base teórica", "§ 01.3 · Theoretical base"),
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
        "fig_id": "01.3",
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
        "id": "component-sensitivity",
        "chapter": T("§ 02.3 · Asimetría", "§ 02.3 · Asymmetry"),
        "title": T("14× de diferencia entre Q y FFN",
                   "14× gap between Q and FFN"),
        "subtitle": T(
            "Aislando qué tipo de componente aguanta y cuál se rompe.",
            "Isolating which component type survives and which breaks."),
        "body": [
            T("Para ver el efecto de cada tipo por separado se "
              "comprimen las 12 matrices de un solo tipo a un rango "
              "fijo y se deja todo lo demás intacto. Se repite el "
              "experimento para los seis tipos y tres rangos.",
              "To see each type's effect on its own, the 12 matrices "
              "of a single type get compressed to a fixed rank while "
              "everything else stays intact. The experiment is "
              "repeated for the six types and three ranks."),
            T("A rango 128 las diferencias son enormes. Q retiene el "
              "99,4 % del F1, K el 98,4 %. La FFN Intermediate, al "
              "mismo rango, se queda en 6,9 %. Eso son 14× de "
              "retención absoluta entre uno y otro, 72× si lo "
              "normalizas por porcentaje de parámetros eliminados.",
              "At rank 128 the differences are huge. Q keeps 99.4 % "
              "of F1, K 98.4 %. FFN Intermediate, at the same rank, "
              "drops to 6.9 %. That's 14× the absolute retention "
              "between one and the other, 72× once you normalise by "
              "parameters eliminated."),
            T("Aparecen tres regímenes. Q y K son inmunes: el "
              "espectro está concentrado, la dimensionalidad efectiva "
              "es baja, y la degradación con el rango es casi lineal. "
              "V y Attn-O hacen un acantilado: aguantan rango 256 "
              "pero colapsan en cuanto bajas a 64. Las dos FFN son "
              "frágiles desde el principio: el espectro es plano, "
              "cada dimensión aporta lo suyo, y a rango 256 ya están "
              "rotas.",
              "Three regimes show up. Q and K are immune: the "
              "spectrum is concentrated, effective dimensionality is "
              "low, and degradation with rank is roughly linear. V "
              "and Attn-O hit a cliff: they survive rank 256 but "
              "collapse the moment you drop to 64. The two FFNs are "
              "fragile from the start: a flat spectrum, every "
              "dimension contributing something, and at rank 256 "
              "they're already broken."),
        ],
        "pull": T(
            "La estructura del espectro predice bastante bien el "
            "comportamiento bajo compresión. Tratar igual a Q y a "
            "FFN es subóptimo por un factor de hasta 72×.",
            "The spectral structure predicts compression behaviour "
            "fairly accurately. Treating Q the same as FFN is "
            "suboptimal by up to 72×."),
        "figure": "component_sensitivity",
        "caption": T(
            "Datos: notebook 3, component_sensitivity.csv. F1 retención "
            "al comprimir uniformemente sólo las 12 matrices de un tipo.",
            "Data: notebook 3, component_sensitivity.csv. F1 retention "
            "from uniformly compressing only the 12 matrices of one type."),
        "fig_id": "02.3",
    },
    {
        "kind": "figure",
        "id": "spectral-landscape",
        "chapter": T("§ 02.4 · 3D", "§ 02.4 · 3D"),
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
        "fig_id": "02.4",
    },
    {
        "kind": "figure",
        "id": "decay",
        "chapter": T("§ 02.5 · Animación", "§ 02.5 · Animation"),
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
        "fig_id": "02.5",
    },
    {
        "kind": "figure",
        "id": "ft-diff",
        "chapter": T("§ 02.6 · Pre vs post", "§ 02.6 · Pre vs post"),
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
        "fig_id": "02.6",
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
        "id": "info-gain",
        "chapter": T("§ 03.2 · Ganancia", "§ 03.2 · Gain"),
        "title": T("L0 absorbe el <em>61 %</em>",
                   "L0 absorbs <em>61 %</em>"),
        "subtitle": T(
            "Cuánto F1 añade cada capa respecto a la anterior.",
            "How much F1 each layer adds over the previous one."),
        "body": [
            T("La curva de probing acumulada del panel anterior sube "
              "de forma engañosamente suave. Si miras la derivada — "
              "cuánto añade cada capa respecto a la anterior — el "
              "perfil cambia bastante: hay un salto enorme al "
              "principio y casi nada después.",
              "The cumulative probing curve in the previous panel "
              "rises deceptively smoothly. Look at the derivative "
              "instead — how much each layer adds over the previous "
              "one — and the picture shifts: there's a giant jump at "
              "the start and almost nothing after."),
            T("Del embedding (F1 = 0) a L0, la media salta a 0,349. "
              "Eso es el 61 % de la separabilidad final del modelo "
              "(F1 medio en L11 = 0,569). En una sola capa.",
              "From the embedding (F1 = 0) to L0, mean F1 jumps to "
              "0.349. That's 61 % of the model's final separability "
              "(mean F1 at L11 = 0.569). In a single layer."),
            T("Lo que viene después es desarrollo, no creación. Las "
              "capas L1–L7 aportan ganancias modestas (+0,010 a "
              "+0,048 cada una). En L8 hay un pequeño rebote — "
              "desambiguación contextual de emociones tardías como "
              "joy, desire, approval. Y después, L10 y L11 aportan "
              "prácticamente nada (<0,005).",
              "What comes after is refinement, not creation. Layers "
              "L1–L7 add modest gains (+0.010 to +0.048 each). At L8 "
              "there's a small bump — contextual disambiguation of "
              "late emotions like joy, desire, approval. And after "
              "that, L10 and L11 contribute almost nothing (<0.005)."),
        ],
        "pull": T(
            "El probing dice que las capas tardías no aportan "
            "información nueva. El activation patching dice que son "
            "las únicas suficientes para revivir el modelo. La "
            "paradoja se resuelve más abajo: lo que hacen no es "
            "crear señal, es rotarla hacia la base del clasificador.",
            "Probing says the late layers don't add new information. "
            "Activation patching says they're the only ones sufficient "
            "to revive the model. The paradox resolves further down: "
            "what they do isn't create signal, it's rotate it onto "
            "the classifier's basis."),
        "figure": "info_gain",
        "caption": T(
            "Δ F1 macro por capa. Media sobre 23 emociones. Datos: "
            "probe_results.csv (notebook 4).",
            "Δ F1 macro per layer. Mean over 23 emotions. Data: "
            "probe_results.csv (notebook 4)."),
        "fig_id": "03.2",
    },
    {
        "kind": "figure",
        "id": "galaxy",
        "chapter": T("§ 03.3 · Geometría", "§ 03.3 · Geometry"),
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
        "fig_id": "03.3",
    },
    {
        "kind": "figure",
        "id": "iterative",
        "chapter": T("§ 03.4 · Logit lens", "§ 03.4 · Logit lens"),
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
        "fig_id": "03.4",
    },
    {
        "kind": "figure",
        "id": "lens-vs-probe",
        "chapter": T("§ 03.5 · Comparación", "§ 03.5 · Comparison"),
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
              "capas 0–4 es sólo el 27 %. La capa 11 entera (12 cabezas) "
              "no contiene NINGUNA cabeza prescindible. Y hay 21 cabezas "
              "interferentes — su ablación MEJORA el F1.",
              "77 % of heads in layers 8–11 are critical. In layers 0–4 "
              "it's only 27 %. Layer 11 as a whole (12 heads) has ZERO "
              "dispensable heads. And there are 21 interfering heads — "
              "ablating them IMPROVES F1."),
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
              "destello (0,1 % de restauración). L9 empuja arriba a las "
              "emociones léxicas (4,1 %). L10 recupera el 24,1 %. L11 "
              "hace explotar todas las barras al baseline "
              "simultáneamente: <strong>100 % de restauración con una "
              "sola capa</strong>.",
              "The first 8 stages move nothing. L8 sets off a flicker "
              "(0.1 % restoration). L9 lifts the lexical emotions "
              "(4.1 %). L10 recovers 24.1 %. L11 blows every bar back to "
              "baseline simultaneously: <strong>100 % restoration with a "
              "single layer</strong>."),
            T("Y hay más. Restaurar SOLO la FFN de L11 (no la atención) "
              "ya recupera el 100 %. La atención de L11 sola, sólo el "
              "63,3 %. La capacidad emocional del modelo no está "
              "distribuida. Vive concentrada en un sub-componente "
              "específico de la última capa.",
              "And it goes further. Restoring ONLY the FFN of L11 (not "
              "attention) already gets 100 %. L11 attention alone, only "
              "63.3 %. The model's emotional capacity isn't distributed. "
              "It lives concentrated in one specific sub-component of "
              "the last layer."),
        ],
        "figure": "lesion_theater",
        "caption": T(
            "Activation patching secuencial por capa. F1 macro y por "
            "emoción. 12 etapas + estado inicial.",
            "Sequential layer-wise activation patching. F1 macro and "
            "per emotion. 12 stages plus initial state."),
        "fig_id": "05.1",
    },
    {
        "kind": "figure",
        "id": "patching-components",
        "chapter": T("§ 05.2 · Por componente",
                     "§ 05.2 · By component"),
        "title": T("Solo la <em>FFN de L11</em> basta",
                   "<em>L11 FFN</em> alone is enough"),
        "subtitle": T(
            "Patching no por capa entera sino por sub-componente.",
            "Patching not by full layer but by sub-component."),
        "body": [
            T("La figura anterior restaura una capa entera (sus 6 "
              "matrices). ¿Y si restauramos solo una mitad — solo el "
              "bloque de atención, o solo el FFN? Es la versión más "
              "fina del experimento.",
              "The previous figure restores a whole layer (all 6 "
              "matrices). What happens if we restore only half — just "
              "the attention block, or just the FFN? The finer version "
              "of the experiment."),
            T("De L8 a L10 las dos columnas crecen lentamente y a la "
              "par. Pero en L11 se separan: la atención sola devuelve "
              "el 63 %, mientras que <strong>la FFN sola devuelve el "
              "100 %</strong>. Un sub-componente — una tercera parte "
              "de los parámetros de L11 — basta para revivir un modelo "
              "completamente colapsado.",
              "From L8 to L10 both columns grow slowly and in lockstep. "
              "But in L11 they split: attention alone gets 63 %, "
              "<strong>FFN alone gets 100 %</strong>. One "
              "sub-component — a third of L11's parameters — is enough "
              "to revive a totally collapsed model."),
            T("Esto va contra la narrativa de \"Attention Is All You "
              "Need\". Para clasificación emocional sobre este "
              "fine-tune, lo crítico es la FFN tardía. La atención "
              "ayuda; la FFN decide.",
              "This goes against the \"Attention Is All You Need\" "
              "narrative. For emotion classification on this "
              "fine-tune, what's critical is the late FFN. Attention "
              "helps; FFN decides."),
        ],
        "pull": T(
            "La FFN de L11 ejecuta una rotación geométrica que lleva "
            "la representación a la base sobre la que opera el "
            "classifier. Por eso es suficiente — y por eso es la "
            "única matriz del modelo que NUNCA conviene comprimir.",
            "L11's FFN executes a geometric rotation that takes the "
            "representation onto the basis where the classifier "
            "operates. That's why it's sufficient — and why it's the "
            "single matrix in the model that should NEVER be "
            "compressed."),
        "figure": "patching_components",
        "caption": T(
            "Datos: notebook 5, activation_patching_per_component.csv. "
            "Restauración media sobre 23 emociones. Capas 0–7 omitidas "
            "(restauran 0 % al ser patcheadas individualmente).",
            "Data: notebook 5, activation_patching_per_component.csv. "
            "Mean restoration over 23 emotions. Layers 0–7 omitted "
            "(each restores 0 % when patched individually)."),
        "fig_id": "05.2",
    },

    # ── PART 6: el mapa emocional ────────────────────────────────────────
    {
        "kind": "part",
        "id": "parte-6",
        "num": T("Parte 06", "Part 06"),
        "title": T("El mapa emocional", "The emotional map"),
        "intro": [
            T("Cinco técnicas de interpretabilidad mecánica han "
              "documentado lo mismo desde ángulos distintos: la "
              "información emocional vive concentrada al final del "
              "modelo. Probing, logit lens, activation patching, "
              "ablación de cabezas y selectividad neuronal convergen "
              "en una arquitectura funcional de tres niveles.",
              "Five mechanistic interpretability techniques have "
              "documented the same thing from different angles: "
              "emotional information lives concentrated at the end of "
              "the model. Probing, logit lens, activation patching, "
              "head ablation and neural selectivity converge on a "
              "three-level functional architecture."),
            T("Aquí se pone todo junto: dónde viven las neuronas "
              "emocionales, qué taxonomía emerge sin pedírselo al "
              "modelo, y cómo se ve una sola frase atravesando todas "
              "las capas a la vez.",
              "This is where everything fits together: where the "
              "emotional neurons live, what taxonomy emerges without "
              "asking the model for it, and what a single sentence "
              "looks like as it crosses all the layers at once."),
        ],
    },
    {
        "kind": "figure",
        "id": "neurons",
        "chapter": T("§ 06.1 · Neuronas", "§ 06.1 · Neurons"),
        "title": T("Dónde viven las <em>neuronas emocionales</em>",
                   "Where the <em>emotional neurons</em> live"),
        "subtitle": T(
            "Selectividad por neurona: 84 % de las significativas en L8–L11.",
            "Per-neuron selectivity: 84 % of significant ones in L8–L11."),
        "body": [
            T("Para cada una de las 36 864 neuronas intermedias del "
              "modelo (12 capas × 3 072) calculamos un score tipo "
              "Cohen's d: ¿cuánto se diferencia su activación cuando "
              "una emoción está presente respecto a cuando no? "
              "Llamamos significativas a las que tienen |d| > 2,0.",
              "For each of the 36,864 intermediate neurons in the "
              "model (12 layers × 3,072) we compute a Cohen's-d–style "
              "score: how different is its activation when an "
              "emotion is present versus absent? We call significant "
              "those with |d| > 2.0."),
            T("Hay 3 642 en total. La distribución por profundidad es "
              "extrema: 11 en capas 0–3 (0,3 %), 570 en capas 4–7 "
              "(16 %), <strong>3 061 en capas 8–11 (84 %)</strong>. "
              "L11 sola contiene 1 127 — más que toda la mitad "
              "inferior del modelo combinada.",
              "There are 3,642 in total. The depth distribution is "
              "extreme: 11 in layers 0–3 (0.3 %), 570 in layers 4–7 "
              "(16 %), <strong>3,061 in layers 8–11 (84 %)</strong>. "
              "Layer 11 alone contains 1,127 — more than the entire "
              "bottom half of the model combined."),
            T("Por emoción, el desequilibrio también es brutal. "
              "Gratitude tiene 818 neuronas dedicadas, max selectivity "
              "6,88. Remorse 442. Love 399. En el extremo opuesto, "
              "annoyance, disappointment y realization tienen CERO "
              "neuronas significativas. Su representación distribuida "
              "explica por qué son las más vulnerables a CUALQUIER "
              "perturbación del modelo.",
              "By emotion the imbalance is also brutal. Gratitude has "
              "818 dedicated neurons, max selectivity 6.88. Remorse "
              "442. Love 399. At the other extreme, annoyance, "
              "disappointment and realization have ZERO significant "
              "neurons. Their distributed representation explains why "
              "they're the most fragile under ANY model "
              "perturbation."),
        ],
        "pull": T(
            "La norma del vector de selectividad es el mejor predictor "
            "de la caída de F1 bajo SVD (Spearman ρ = 0,64, "
            "p = 0,001). Las emociones \"escritas en negrita\" en los "
            "pesos son las que más sufren cualquier compresión.",
            "The selectivity-vector norm is the best predictor of F1 "
            "drop under SVD (Spearman ρ = 0.64, p = 0.001). The "
            "emotions written \"in bold\" inside the weights are the "
            "ones that suffer most under any compression."),
        "figure": "neurons",
        "caption": T(
            "Datos: notebook 7. neuron_significant_counts.csv y "
            "neuron_catalog.csv. Conteos reales sobre el conjunto de "
            "test del checkpoint 23emo-final.",
            "Data: notebook 7. neuron_significant_counts.csv and "
            "neuron_catalog.csv. Real counts on the test set of the "
            "23emo-final checkpoint."),
        "fig_id": "06.1",
    },
    {
        "kind": "figure",
        "id": "clusters",
        "chapter": T("§ 06.2 · Taxonomía", "§ 06.2 · Taxonomy"),
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
        "fig_id": "06.2",
    },
    {
        "kind": "figure",
        "id": "landscape",
        "chapter": T("§ 06.3 · Mapa", "§ 06.3 · Map"),
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
        "fig_id": "06.3",
    },
    {
        "kind": "figure",
        "id": "trajectory",
        "chapter": T("§ 06.4 · Síntesis", "§ 06.4 · Synthesis"),
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
        "fig_id": "06.4",
    },

    # ── PART 7: compresión informada ─────────────────────────────────────
    {
        "kind": "part",
        "id": "parte-7",
        "num": T("Parte 07", "Part 07"),
        "title": T("Compresión informada", "Informed compression"),
        "intro": [
            T("La pregunta prescriptiva del proyecto: ¿se puede usar lo "
              "que ahora sabemos del modelo para comprimirlo mejor?",
              "The project's prescriptive question: can we use what we "
              "now know about the model to compress it better?"),
            T("Antes de saltar al algoritmo final hay que contar el "
              "intento que NO funcionó. Tres heurísticas escritas a "
              "mano a partir de los hallazgos de interpretabilidad. "
              "Resultado: convergen exactamente sobre los baselines "
              "ciegos. Saber QUÉ medir no basta — hay que medir CUÁNTO.",
              "Before jumping to the final algorithm we have to tell "
              "the attempt that did NOT work. Three hand-written "
              "heuristics built from the interpretability findings. "
              "Result: they collapse onto blind baselines. Knowing "
              "WHAT to measure isn't enough — you have to measure HOW "
              "MUCH."),
        ],
    },
    {
        "kind": "figure",
        "id": "heuristic-negative",
        "chapter": T("§ 07.1 · Resultado negativo",
                     "§ 07.1 · Negative result"),
        "title": T("La heurística <em>colapsa</em> sobre lo ciego",
                   "Heuristic <em>collapses</em> onto blind"),
        "subtitle": T(
            "Tres reglas informadas, exactamente sobre uniform_r256 y r512.",
            "Three informed rules, exactly on top of uniform_r256 and r512."),
        "body": [
            T("Las tres heurísticas se escribieron antes que el "
              "greedy. La idea era directa: si las capas tardías son "
              "críticas, protégerlas; si Q y K son inmunes, "
              "comprímelos primero. Tres niveles de agresividad. "
              "Resultado experimental:",
              "The three heuristics were written before the greedy "
              "one. The idea was direct: if late layers are critical, "
              "protect them; if Q and K are immune, compress them "
              "first. Three aggressiveness levels. Experimental "
              "result:"),
            T("informed_aggressive coincide exactamente con "
              "uniform_r256 (mismo ratio 0,612, mismo F1 0,025). "
              "informed_moderate coincide con uniform_r512 (ratio "
              "1,000, F1 0,464). Y informed_light necesita ratio "
              "1,285, es decir, más parámetros que el modelo "
              "original. Tres reglas escritas a mano y los tres "
              "puntos caen literalmente sobre los baselines ciegos.",
              "informed_aggressive matches uniform_r256 exactly "
              "(same ratio 0.612, same F1 0.025). informed_moderate "
              "matches uniform_r512 (ratio 1.000, F1 0.464). And "
              "informed_light needs ratio 1.285, that is, more "
              "parameters than the original model. Three hand-"
              "written rules and the three points fall literally on "
              "top of the blind baselines."),
            T("Ninguna está sobre la frontera de Pareto. Saber "
              "cualitativamente qué es importante no se convierte por "
              "sí solo en una asignación óptima de rangos. La "
              "interpretabilidad cualitativa identifica las variables "
              "que importan; los valores numéricos los tienen que "
              "fijar los datos empíricos de sensibilidad.",
              "None of them lands on the Pareto frontier. Knowing "
              "qualitatively what's important doesn't translate by "
              "itself into an optimal rank assignment. Qualitative "
              "interpretability identifies the variables that matter; "
              "the actual numeric values have to come from empirical "
              "sensitivity data."),
        ],
        "pull": T(
            "Es un resultado negativo, pero informativo. Habría sido "
            "fácil contar la historia como éxito si las heurísticas "
            "hubiesen funcionado. La narrativa entera — la heurística "
            "no aporta, pivot a data-driven, el greedy domina "
            "Pareto — es en sí misma una observación metodológica "
            "que se puede llevar a otros sitios.",
            "It's a negative result, but an informative one. It "
            "would have been easy to spin the story as a success if "
            "the heuristics had worked. The whole arc — heuristics "
            "don't add anything, pivot to data-driven, greedy "
            "dominates Pareto — is in itself a methodological "
            "observation that travels well."),
        "figure": "heuristic_negative",
        "caption": T(
            "Datos: notebook 9, compression_comparison.csv. 21 "
            "estrategias evaluadas: 6 uniformes, 4 adaptativas, 3 "
            "heurísticas, 8 greedy.",
            "Data: notebook 9, compression_comparison.csv. 21 "
            "evaluated strategies: 6 uniform, 4 adaptive, 3 "
            "heuristic, 8 greedy."),
        "fig_id": "07.1",
    },
    {
        "kind": "figure",
        "id": "greedy",
        "chapter": T("§ 07.2 · Algoritmo", "§ 07.2 · Algorithm"),
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
        "fig_id": "07.2",
    },
    {
        "kind": "figure",
        "id": "recovery",
        "chapter": T("§ 07.3 · Recuperación", "§ 07.3 · Recovery"),
        "title": T("El comprimido <em>vuelve</em>",
                   "The compressed model <em>comes back</em>"),
        "subtitle": T(
            "Greedy_90 + 3 épocas de fine-tuning supera al baseline.",
            "Greedy_90 + 3 epochs of fine-tuning beat the baseline."),
        "body": [
            T("Punto de partida: greedy_90, 86,4 % de los parámetros "
              "del baseline. F1 macro 0,539. Habíamos perdido un "
              "6,7 % de rendimiento, que sería el coste razonable de "
              "una compresión.",
              "Starting point: greedy_90, 86.4 % of baseline "
              "parameters. F1 macro 0.539. We'd lost 6.7 % of "
              "performance, which would be the reasonable cost of "
              "compression."),
            T("Tres épocas de fine-tuning después, F1 macro 0,591. "
              "Por encima del baseline original (0,577) con un 13,6 % "
              "menos de parámetros. La compresión no es un coste; "
              "está actuando como regularización implícita.",
              "Three epochs of fine-tuning later, F1 macro 0.591. "
              "Above the original baseline (0.577) with 13.6 % fewer "
              "parameters. Compression isn't a cost; it's behaving "
              "like implicit regularisation."),
            T("La ganancia se concentra donde más hace falta. "
              "Embarrassment pasa de F1 0,267 a 0,509 — un 90 % "
              "relativo más, en una emoción con sólo 303 ejemplos en "
              "entrenamiento. Desire sube un 16 %, excitement un "
              "9,5 %, realization un 10 %. La explicación más "
              "plausible es que la SVD ha eliminado direcciones "
              "ruidosas en las que el baseline había memorizado "
              "patrones espurios para emociones con poca masa de "
              "datos.",
              "The gain concentrates where it's needed most. "
              "Embarrassment goes from F1 0.267 to 0.509 — a 90 % "
              "relative jump, on an emotion with only 303 training "
              "examples. Desire goes up 16 %, excitement 9.5 %, "
              "realization 10 %. The most plausible reading is that "
              "SVD removed noisy directions where the baseline had "
              "memorised spurious patterns for emotions with little "
              "data behind them."),
        ],
        "pull": T(
            "Es un resultado preliminar — un modelo, una tarea, sin "
            "grupo de control con epochs adicionales. Pero la dirección "
            "es la opuesta a lo que se asume sobre compresión.",
            "It's a preliminary result — one model, one task, no "
            "control group with extra epochs. But the direction is the "
            "opposite of what's typically assumed about compression."),
        "figure": "finetuning_recovery",
        "caption": T(
            "Datos: notebook 9, finetuning_recovery.csv. F1 baseline / "
            "comprimido (greedy_90) / fine-tuneado por emoción.",
            "Data: notebook 9, finetuning_recovery.csv. F1 baseline / "
            "compressed (greedy_90) / fine-tuned per emotion."),
        "fig_id": "07.3",
    },
]


# ── Hero stats ────────────────────────────────────────────────────────────
HERO_STATS = [
    ("12 × 144",  T("capas × cabezas", "layers × heads")),
    ("36 864",    T("neuronas FFN", "FFN neurons")),
    ("23",        T("emociones, multi-label", "emotions, multi-label")),
    ("26",        T("visualizaciones", "visualisations")),
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
    "title":    T("Lo que <em>emergió</em>", "What <em>emerged</em>"),
    "sub":      T(
        "Recapitulación honesta: lo que se ha encontrado, lo que no se ha "
        "podido demostrar, y por dónde seguir.",
        "An honest recap: what was found, what couldn't be demonstrated, "
        "and where to go next."),
    "blocks": [
        # ── Contribuciones ─────────────────────────────────────────────
        {
            "label": T("01 · Contribuciones",
                       "01 · Contributions"),
            "title": T("Tres hallazgos sustantivos",
                       "Three substantive findings"),
            "paragraphs": [
                T("Arquitectura funcional de tres niveles. Las capas "
                  "tempranas hacen señal léxica cruda; L0 sola absorbe "
                  "el 61 % de la separabilidad final del modelo. Las "
                  "medias computan una transición que el probing ve "
                  "subir pero la cabeza de clasificación no sabe "
                  "todavía leer. Las tardías hacen la rotación final, "
                  "y restaurar sólo la FFN de la capa 11 ya recupera "
                  "el 100 % del F1 desde un colapso total. No hace "
                  "falta toda la capa: hace falta ese sub-componente "
                  "concreto.",
                  "A three-level functional architecture. The early "
                  "layers do raw lexical work, with L0 alone absorbing "
                  "61 % of the model's final separability. The middle "
                  "layers compute a transition that probing sees rise "
                  "but the classifier head can't yet read. The late "
                  "ones do the final rotation, and restoring just the "
                  "FFN of layer 11 already recovers 100 % of F1 from "
                  "total collapse. You don't need the whole layer; you "
                  "need that specific sub-component."),
                T("Sensibilidad muy desigual entre componentes. Q a "
                  "rango 128 conserva el 99,4 % del F1; la FFN "
                  "Intermediate al mismo rango cae al 6,9 %. Catorce "
                  "veces más de retención absoluta, setenta y dos "
                  "veces si se normaliza por parámetros eliminados. La "
                  "compresión uniforme cae por un acantilado entre "
                  "rango 384 y 256, y por debajo de 128 el F1 es "
                  "exactamente cero. Que yo sepa, este factor de "
                  "14×–72× no estaba cuantificado en la literatura "
                  "previa de SVD sobre Transformers.",
                  "Wildly uneven sensitivity across components. Q at "
                  "rank 128 keeps 99.4 % of F1; FFN Intermediate at "
                  "the same rank drops to 6.9 %. Fourteen times the "
                  "absolute retention, seventy-two times once you "
                  "normalise by parameters eliminated. Uniform "
                  "compression falls off a cliff between rank 384 and "
                  "256, and below 128 F1 is exactly zero. As far as "
                  "I'm aware, this 14×–72× factor wasn't quantified "
                  "in the prior literature on SVD for Transformers."),
                T("El algoritmo greedy se queda con 8 de los 9 puntos "
                  "Pareto-óptimos. Al 80 % de parámetros retiene el "
                  "87 % del F1, frente al 43 % de la compresión "
                  "uniforme al mismo ratio. Y descubre la jerarquía "
                  "que la interpretabilidad había encontrado por otra "
                  "vía: comprime Q y K primero, no toca la FFN "
                  "Intermediate de las capas tardías. Tras tres "
                  "épocas de fine-tuning, el modelo comprimido al "
                  "86,4 % de parámetros llega a F1 0,591 — por encima "
                  "del baseline (0,577). La ganancia se concentra en "
                  "emociones infrarrepresentadas: embarrassment pasa "
                  "de 0,267 a 0,509.",
                  "The greedy algorithm takes 8 of the 9 "
                  "Pareto-optimal points. At 80 % parameters it keeps "
                  "87 % of F1, versus 43 % for uniform at the same "
                  "ratio. And it rediscovers the hierarchy that "
                  "interpretability had found from a different angle: "
                  "compress Q and K first, never touch the FFN "
                  "Intermediate of the late layers. After three "
                  "epochs of fine-tuning, the compressed model at "
                  "86.4 % of parameters reaches F1 0.591 — above the "
                  "baseline (0.577). The gain concentrates on "
                  "underrepresented emotions: embarrassment goes "
                  "from 0.267 to 0.509."),
            ],
        },
        # ── Limitaciones ────────────────────────────────────────────────
        {
            "label": T("02 · Limitaciones",
                       "02 · Limitations"),
            "title": T("Lo que <em>no</em> se ha podido demostrar",
                       "What <em>couldn't</em> be demonstrated"),
            "paragraphs": [
                T("Un modelo, una tarea. Todo lo que aparece aquí se "
                  "ejecuta sobre BERT-base y GoEmotions. Que la "
                  "arquitectura sea encoder-only puede estar "
                  "favoreciendo la concentración tardía que "
                  "documentamos; con BERT-large, RoBERTa, GPT-2 o "
                  "LLaMA podría no replicar igual. Es la prioridad "
                  "número uno del trabajo futuro, pero hoy no está "
                  "verificada.",
                  "One model, one task. Everything in here runs on "
                  "BERT-base and GoEmotions. The encoder-only "
                  "architecture might be helping the late-layer "
                  "concentration we documented; with BERT-large, "
                  "RoBERTa, GPT-2 or LLaMA it might not replicate the "
                  "same. It's the top item on the future-work list, "
                  "but as of today it's not verified."),
                T("Compresión post-hoc, no durante el entrenamiento. "
                  "La SVD entra cuando el modelo ya está fine-tuneado. "
                  "Otros caminos (pruning estructurado, cuantización, "
                  "destilación) interactúan de forma distinta y no se "
                  "evalúan en combinación. Lo razonable sería un "
                  "pipeline que los apile, pero queda fuera del "
                  "alcance.",
                  "Post-hoc compression, not during training. SVD "
                  "comes in once the model is already fine-tuned. "
                  "Other paths (structured pruning, quantisation, "
                  "distillation) interact differently and aren't "
                  "evaluated in combination. The reasonable next step "
                  "is a pipeline that stacks them, but it's out of "
                  "scope here."),
                T("Potencia estadística limitada. Las correlaciones "
                  "se calculan sobre n = 23 emociones; eso da "
                  "confianza para detectar efectos grandes (ρ > 0,556) "
                  "pero no efectos moderados. La regularización que "
                  "se observa al fine-tunear el modelo comprimido "
                  "(+90 % en embarrassment) carece de grupo de control "
                  "con épocas extra sobre el baseline sin comprimir, "
                  "así que se reporta como observación consistente "
                  "con la hipótesis y no como causalidad establecida.",
                  "Limited statistical power. Correlations are "
                  "computed over n = 23 emotions; that gives confidence "
                  "for detecting large effects (ρ > 0.556) but not "
                  "moderate ones. The regularisation observed when "
                  "fine-tuning the compressed model (+90 % on "
                  "embarrassment) doesn't have a control with extra "
                  "epochs on the uncompressed baseline, so it's "
                  "reported as an observation consistent with the "
                  "hypothesis rather than established causation."),
                T("Distorsión espectral, no ruido neutral. El "
                  "activation patching parte de una corrupción "
                  "estructurada (SVD a rango 64), no de ruido "
                  "gaussiano como en el causal tracing original de "
                  "Meng et al. Las conclusiones que se sacan son "
                  "funcionales —qué componentes bastan para revivir "
                  "el modelo desde el colapso— más que estrictamente "
                  "causales en el sentido de Pearl. Es una distinción "
                  "que conviene tener clara.",
                  "Spectral distortion, not neutral noise. Activation "
                  "patching starts from a structured corruption (SVD "
                  "to rank 64), not Gaussian noise as in Meng et al.'s "
                  "original causal tracing. What you can conclude is "
                  "functional — which components are enough to "
                  "revive the model from collapse — rather than "
                  "strictly causal in Pearl's sense. Worth keeping "
                  "that distinction in mind."),
            ],
        },
        # ── Trabajo futuro ──────────────────────────────────────────────
        {
            "label": T("03 · Trabajo futuro",
                       "03 · Future work"),
            "title": T("Predicciones <em>falsables</em>",
                       "<em>Falsifiable</em> predictions"),
            "paragraphs": [
                T("Generalización a otros modelos y tareas. La "
                  "predicción concreta: el ratio de compresibilidad "
                  "espectral k₉₅(Q)/k₉₅(FFN) en BERT-large debería "
                  "caer en el rango [0,55, 0,75]. En BERT-base es "
                  "0,64. En decoder-only la restauración por "
                  "activation patching debería repartirse entre varias "
                  "capas tardías en lugar de concentrarse tanto en "
                  "L11. Si no replica, hay que matizar la hipótesis "
                  "de la jerarquía funcional.",
                  "Generalisation to other models and tasks. The "
                  "concrete prediction: the spectral compressibility "
                  "ratio k₉₅(Q)/k₉₅(FFN) in BERT-large should land in "
                  "[0.55, 0.75]. BERT-base sits at 0.64. In "
                  "decoder-only models, activation-patching "
                  "restoration should spread across several late "
                  "layers instead of concentrating so much on L11. If "
                  "it doesn't replicate, the functional-hierarchy "
                  "claim needs softening."),
                T("Verificar causalmente la regularización por "
                  "compresión. Tres condiciones: (i) baseline con 3 "
                  "épocas adicionales, (ii) baseline + greedy + 3 "
                  "épocas (lo de aquí), (iii) baseline + 3 épocas con "
                  "dropout y weight-decay subidos. Si (ii) supera a "
                  "(i) y (iii) en F1 macro y, sobre todo, en "
                  "emociones infrarrepresentadas, la hipótesis de "
                  "regularización implícita queda bien apoyada.",
                  "Causally verifying the regularisation-from-"
                  "compression effect. Three conditions: (i) baseline "
                  "with 3 extra epochs, (ii) baseline + greedy + 3 "
                  "epochs (this work), (iii) baseline + 3 epochs with "
                  "raised dropout and weight decay. If (ii) beats (i) "
                  "and (iii) on macro F1 and especially on "
                  "underrepresented emotions, the implicit-"
                  "regularisation hypothesis stands on firmer ground."),
                T("Compresión por cabeza individual. Tenemos 38 "
                  "cabezas prescindibles más 21 interferentes "
                  "identificadas; son candidatas directas a "
                  "eliminación. Apilar pruning de cabezas + greedy "
                  "SVD + cuantización post-hoc + fine-tuning recovery "
                  "podría componer reducciones multiplicativas sin "
                  "perder F1. Es la línea más práctica.",
                  "Per-head compression granularity. We have 38 "
                  "dispensable plus 21 interfering heads identified; "
                  "those are direct elimination candidates. Stacking "
                  "head pruning + greedy SVD + post-hoc quantisation "
                  "+ fine-tuning recovery could compose multiplicative "
                  "reductions without losing F1. It's the most "
                  "practical line of work."),
                T("Tuned lens y dinámica de entrenamiento. Aprender "
                  "una transformación T_ℓ : ℝ^d → ℝ^d por capa que "
                  "minimice la divergencia KL contra L11 y ver si el "
                  "patrón en U sobrevive a esa calibración. Y "
                  "monitorizar cristalización y especialización "
                  "neuronal durante el fine-tuning, para saber en qué "
                  "momento de la optimización aparece cada propiedad "
                  "estructural.",
                  "Tuned lens and training dynamics. Learn a "
                  "per-layer transformation T_ℓ : ℝ^d → ℝ^d that "
                  "minimises KL divergence against L11 and check "
                  "whether the U pattern survives that calibration. "
                  "And track crystallisation and neural specialisation "
                  "during fine-tuning itself, to find out at what "
                  "point in optimisation each structural property "
                  "shows up."),
            ],
        },
    ],
    "coda": T(
        "BERT no fue diseñado para clasificar emociones. La "
        "arquitectura funcional que aparece aquí — cristalización "
        "progresiva, dominio de la FFN tardía, la U del logit lens, "
        "los seis clusters con coherencia psicológica — no se "
        "programó. Salió sola. Lo que hace la interpretabilidad "
        "mecánica es documentar lo que el gradiente decidió, no lo "
        "que nadie prescribió.<br><br>"
        "<a href=\"sobre.html\">Cómo está hecho · Stack y créditos</a>",
        "BERT wasn't designed for emotion classification. The "
        "functional architecture that shows up here — progressive "
        "crystallisation, late-FFN dominance, the logit-lens U, six "
        "clusters with psychological coherence — wasn't programmed. "
        "It came out on its own. What mechanistic interpretability "
        "does is document what gradient descent decided, not what "
        "anyone prescribed.<br><br>"
        "<a href=\"sobre.html\">How it's built · Stack and credits</a>"),
}


NAV = {
    "brand":    T("Anatomía Emocional", "Emotional Anatomy"),
    "chapters": T("Capítulos", "Chapters"),
    "data":     T("Datos", "Data"),
    "about":    T("Sobre", "About"),
    "index":    T("Índice", "Index"),
}


COMMENTS = {
    "label":   T("Discusión", "Discussion"),
    "title":   T("Comentarios", "Comments"),
    "sub":     T(
        "Si algo te ha llamado la atención, o discrepas, o quieres "
        "preguntar — abajo. Comentar requiere cuenta de GitHub.",
        "If something caught your eye, or you disagree, or you'd like "
        "to ask — below. Commenting requires a GitHub account."),
    # Giscus config — replace data-category-id once you create the
    # "Comments" category in GitHub Discussions (see web/COMMENTS.md).
    "giscus": {
        "repo":         "BioscaG/transformer-structural-compression-v2",
        "repo_id":      "R_kgDORWTOKA",
        "category":     "Announcements",
        "category_id":  "DIC_kwDORWTOKM4C8Tfx",
        "mapping":      "pathname",
        "reactions":    "1",
        "input_pos":    "top",
    },
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
    "heads_matrix":          740,
    "attention_atlas":      1320,
    "probe_constellations":  800,
    "lesion_theater":        900,
    "patching_components":   600,
    "sunburst":              680,
    "emotional_landscape":   720,
    "sentence_trajectory":  1020,
    "greedy_replay":         820,
    "component_sensitivity": 560,
    "info_gain":             560,
    "finetuning_recovery":   600,
    "neurons":               600,
    "heuristic_negative":    620,
    "methodology":           780,
}
