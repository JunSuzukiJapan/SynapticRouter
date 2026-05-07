[English](../../README.md) | [日本語](../../README_JP.md)

# Synaptic Routing Architecture (SRA)

## 🎮 Demos Interactivas (Jupyter Notebooks)

Hemos preparado Jupyter Notebooks donde puedes experimentar interactivamente el "uso del cerebro específico para cada tarea" y la "robustez" de SRA directamente en tu navegador. Puedes ejecutarlos en segundos en Google Colab, ¡así que pruébalos!

- [01 SRA Quickstart](../../notebooks/01_sra_quickstart_es.ipynb)
- [02 Learning and Routing Demo](../../notebooks/02_learning_and_routing_demo_es.ipynb)
- [03 Multitask Routing Demo](../../notebooks/03_multitask_routing_demo_es.ipynb)
- [04 Decision Transformer Routing Demo](../../notebooks/04_decision_transformer_routing_demo_es.ipynb)
- [05 Lesion Experiment Demo](../../notebooks/05_lesion_experiment_demo_es.ipynb)


Synaptic Routing Architecture (SRA) es una novedosa arquitectura de red neuronal dinámica, dispersa y modular inspirada en el cerebro biológico (sinapsis).
En lugar de un Transformer masivo y estático, SRA enruta dinámicamente las entradas a "sinapsis" (pequeños módulos) apropiadas para lograr un aprendizaje más eficiente e inteligencia estructural.

## 🎯 Motivación

En los últimos años, mientras que los modelos de IA se han vuelto cada vez más masivos, las redes monolíticas enfrentan desafíos significativos, como "la escalada de recursos computacionales" y el "olvido catastrófico durante el aprendizaje multitarea".
SRA intenta resolver estos problemas utilizando un **enfoque disperso: "llamar y combinar dinámicamente solo los módulos minúsculos necesarios (sinapsis) dependiendo de la entrada"**. Esto permite aprender múltiples tareas con diferentes características dentro de la misma red sin interferencia, con el objetivo de lograr tanto la escalabilidad como una alta eficiencia de aprendizaje.

## 💡 Idea Básica

Los modelos de IA típicos (como los Transformers) intentan procesar todo usando un solo "cerebro" gigante. Sin embargo, con este enfoque, la carga computacional se vuelve demasiado pesada cada vez que el modelo se hace más inteligente o más grande. Por lo tanto, SRA adopta un sistema donde **se preparan muchos "pequeños cerebros expertos (que SRA llama 'sinapsis')", y solo se llama a los expertos necesarios dependiendo del problema en cuestión**.

La clave aquí es el mecanismo que decide "a qué experto llamar". SRA tiene un "enrutador (guía)" que selecciona instantáneamente al experto que parece más capaz al observar los datos de entrada. A medida que cada experto se vuelve más inteligente (aprende), este enrutador aprende simultáneamente "a quién es el correcto elegir", creciendo para poder hacer asignaciones óptimas automáticamente.

## 🧠 Descripción de la Arquitectura

SRA consta principalmente de los siguientes componentes:

1. **Sinapsis (Módulo Sináptico)**
   - Unidades computacionales diminutas e independientes (ej. Transformers en miniatura o MLPs).
   - Se especializan en funciones específicas o procesamiento de patrones a través del aprendizaje.
2. **Router (Enrutador)**
   - Selecciona dinámicamente solo las `Top-k` sinapsis óptimas de todas las disponibles según los tokens de entrada, lo que permite un cálculo disperso.
3. **Synapse Space (Espacio de Sinapsis)**
   - Cada sinapsis se coloca en un espacio de incrustación (embedding), organizándose a sí misma para que la distancia entre las sinapsis represente "similitud funcional".
4. **Regla de Aprendizaje Local**
   - Además de la retropropagación estándar, combina reglas de aprendizaje local utilizando una regla de 3 factores (como `trace × routing × reward` basado en Hebbian, STDP y recompensa), promoviendo el equilibrio de carga y la especialización.

## 📁 Estructura del Directorio

- `src/` : Implementaciones principales del modelo SRA y scripts.
- `docs/` : Registros de decisiones arquitectónicas y reportes.
- `data/` : Conjuntos de datos de prueba (Código, Matemáticas, Texto, etc.).
- `tests/` : Códigos de prueba para varios componentes.

## 🚀 Uso

```bash
pip install torch
python src/sra_experiment.py --task reverse --steps 2000
```

## 🧪 Experimentos y Análisis

- [Aprendizaje Multitarea y Análisis de Enrutamiento en Razonamiento Algorítmico](./routing_analysis_algorithmic.md)
- [Análisis de Enrutamiento en Modelado de Lenguaje de Dominio Cruzado (Código / Matemáticas / Texto)](./routing_analysis_language.md)
- [Análisis de Enrutamiento en Traducción Multilingüe (Ing / Fra / Jap) y Generalización Zero-Shot](../dev/multilingual_translation_routing_analysis.md)
  - Un informe fascinante que muestra cómo SRA asigna automáticamente diferentes módulos de traducción en función de la estructura gramatical (SVO vs SOV). Aún más sorprendente, al traducir un par de idiomas no aprendido, ¡utiliza inconscientemente el inglés como "idioma pivote" para resolver el problema!
- [Separación Completa de Percepción y Política en Decision Transformer (Aprendizaje por Refuerzo)](../dev/decision_transformer_routing_analysis.md)
  - Le dimos a SRA la capacidad de jugar un juego. Descubrió una estructura modular asombrosa por sí solo: utiliza exactamente el mismo módulo de "visión" para percibir el entorno en todas las tareas, pero cambia a módulos de "cerebro" completamente diferentes dependiendo de si necesita encontrar un tesoro o huir.
- [Verificación de Traducción Multilingüe Práctica usando SRA Encoder-Decoder](../dev/sra_seq2seq_translation_analysis.md)
  - Un informe que demuestra que al extender SRA a una arquitectura Encoder-Decoder y entrenar durante 30,000 pasos en un corpus real (opus100), puede traducir expresiones prácticas como "Merci beaucoup." y "Good morning." con BLEU=1.0. La introducción de Cross-Attention causó un salto de Decoder-only (BLEU=0) a un BLEU promedio general de 0.27, y logró una precisión cercana a la práctica de BLEU=0.56 en la dirección FR→EN.



---

### 🔌 6. Experimento de Hot-Swap Dinámico de Sinapsis
**Archivo:** [`06_hotswap_experiment_demo_es.ipynb`](./06_hotswap_experiment_demo_es.ipynb)

Demuestra el verdadero poder de SRA: "agregar y reemplazar sinapsis como complementos".
Realizamos un experimento en el que los pesos de un modelo de traducción al español entrenado independientemente (Modelo 2) se fusionan en un modelo de traducción al francés/alemán en ejecución (Modelo 1). Obtendrá conocimientos profundos sobre la modularidad de la arquitectura y por qué es crucial compartir las representaciones base (Atención/Incrustación).

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/06_hotswap_experiment_demo_es.ipynb)

---

### 👑 7. Integración de Modelos mediante Super Router y Gumbel-Softmax
**Archivo:** [`07_super_router_gumbel_demo_es.ipynb`](./07_super_router_gumbel_demo_es.ipynb)

Construimos un "Super Router" que agrupa múltiples modelos especializados (un modelo FR/DE y un modelo ES) y enruta el procesamiento dinámicamente según la entrada.
Esto demuestra el problema de "Enrutamiento perezoso" (Lazy Routing) del Enrutamiento Suave simple (Soft Routing) y muestra cómo el uso de Gumbel-Softmax logra un **Enrutamiento Duro perfecto**, reduciendo el cálculo innecesario del modelo en un 100%.

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/07_super_router_gumbel_demo_es.ipynb)

## 🤝 Contribución y Licencia

Licencia: [MIT License](../../LICENSE).
