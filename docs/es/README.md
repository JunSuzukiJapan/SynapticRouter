[English](../../README.md) | [日本語](../../README_JP.md)

# Synaptic Routing Architecture (SRA)

## 🎮 Demos Interactivas (Jupyter Notebooks)

Hemos preparado Jupyter Notebooks donde puedes experimentar interactivamente el "uso del cerebro específico para cada tarea" y la "robustez" de SRA directamente en tu navegador. Puedes ejecutarlos en segundos en Google Colab, ¡así que pruébalos!

| # | Demo | Descripción | Colab |
|---|------|-------------|-------|
| 🟢 1 | [SRA Quickstart](../../notebooks/01_sra_quickstart_es.ipynb) | Estructura básica de SRA y visualización de routing | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/01_sra_quickstart_es.ipynb) |
| 🔵 2 | [Aprendizaje y Routing](../../notebooks/02_learning_and_routing_demo_es.ipynb) | Aprendizaje de tarea única y especialización de routing | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/02_learning_and_routing_demo_es.ipynb) |
| 🔴 3 | [Routing Multitarea](../../notebooks/03_multitask_routing_demo_es.ipynb) ✨ | Aprendizaje multitarea y conmutación de sinapsis | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/03_multitask_routing_demo_es.ipynb) |
| 🕹️ 4 | [Decision Transformer](../../notebooks/04_decision_transformer_routing_demo_es.ipynb) | Separación de percepción y acción en RL | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/04_decision_transformer_routing_demo_es.ipynb) |
| 🧠 5 | [Experimento Lesión](../../notebooks/05_lesion_experiment_demo_es.ipynb) ✨ | Prueba de modularidad funcional destruyendo sinapsis | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/05_lesion_experiment_demo_es.ipynb) |
| 🔌 6 | [Experimento Hot-Swap](../../notebooks/06_hotswap_experiment_demo_es.ipynb) | Hot-swap sináptico dinámico y límites del router | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/06_hotswap_experiment_demo_es.ipynb) |
| 👑 7 | [Super Router (Gumbel)](../../notebooks/07_super_router_gumbel_demo_es.ipynb) | Integración de modelos via Gumbel-Softmax | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/07_super_router_gumbel_demo_es.ipynb) |
| 📖 8 | [SRA LLM (Shakespeare)](../../notebooks/08_sra_llm_demo_shakespeare_es.ipynb) | Construir y entrenar un Tiny LLM con SRA | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/08_sra_llm_demo_shakespeare_es.ipynb) |
| 📚 9 | [LLM Multidominio](../../notebooks/09_sra_llm_demo_multidomain_es.ipynb) | Aprendizaje simultáneo multidominio (Code/Math/Text) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/09_sra_llm_demo_multidomain_es.ipynb) |
| 💻 10 | [Plugin Hot-Swap](../../notebooks/10_hotswap_plugins_demo_es.ipynb) | Hot-swap zero-shot (sin olvido catastrófico) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/10_hotswap_plugins_demo_es.ipynb) |
| 🗑️ 11 | [Eliminación de Sinapsis](../../notebooks/11_synapse_deletion_demo_es.ipynb) | Eliminación dinámica de sinapsis (pop & clear) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/11_synapse_deletion_demo_es.ipynb) |


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

### 🔌 6. Experimento de Hot-Swap Sináptico Dinámico y Límites de Aprendizaje del Enrutador
**Archivo:** [`06_hotswap_experiment_demo_es.ipynb`](./06_hotswap_experiment_demo_es.ipynb)

Demuestra el verdadero poder de SRA: "adición y eliminación dinámica de sinapsis como complementos (Hot-Swap)".
Realizamos un experimento donde una sinapsis específica del español se fusiona en un modelo de traducción francés/alemán en ejecución.
En este cuaderno, aprenderá la **importancia crucial de compartir y congelar el espacio de conocimiento del modelo base (capas de incrustación/atención, etc.)** para establecer un intercambio en caliente. Al mismo tiempo, se enfrentará a la **mayor barrera de SRA (el problema del gradiente desvaneciente)**: el enrutamiento duro estándar (Top-k) no puede aprender (diferenciar) retroactivamente el enrutamiento de las sinapsis agregadas. Esta limitación sirve como un presagio crítico para la siguiente sección "Gumbel-Softmax (Super Router)".

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/06_hotswap_experiment_demo_es.ipynb)

---

### 👑 7. Integración de Modelos mediante Super Router y Gumbel-Softmax
**Archivo:** [`07_super_router_gumbel_demo_es.ipynb`](./07_super_router_gumbel_demo_es.ipynb)

Construimos un "Super Router" que agrupa múltiples modelos especializados (un modelo FR/DE y un modelo ES) y enruta el procesamiento dinámicamente según la entrada.
Esto demuestra el problema de "Enrutamiento perezoso" (Lazy Routing) del Enrutamiento Suave simple (Soft Routing) y muestra cómo el uso de Gumbel-Softmax logra un **Enrutamiento Duro perfecto**, reduciendo el cálculo innecesario del modelo en un 100%.

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/07_super_router_gumbel_demo_es.ipynb)

---

### 📖 8. Demostración de SRA LLM (Shakespeare)
**File:** [`08_sra_llm_demo_shakespeare_es.ipynb`](../../notebooks/08_sra_llm_demo_shakespeare_es.ipynb)

Este es un tutorial que utiliza datos de Shakespeare a pequeña escala para entrenar SRA como un modelo generativo específico de decodificador (LLM). Después del aprendizaje, se utiliza un mapa de calor para visualizar por qué sinapsis pasó cada token del texto generado.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/08_sra_llm_demo_shakespeare_es.ipynb)

---

### 🌐 9. Demostración de LLM multidominio de SRA (código, matemáticas, texto)
**File:** [`09_sra_llm_demo_multidomain_es.ipynb`](../../notebooks/09_sra_llm_demo_multidomain_es.ipynb)

Experimente la especialidad de SRA de "aprendizaje simultáneo de múltiples dominios (código, matemáticas, texto)" en un LLM a pequeña escala. Puede verificar cómo el modelo divide (especializa) automáticamente las sinapsis en función de los datos.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/09_sra_llm_demo_multidomain_es.ipynb)

---

### 💻 10. Práctico intercambio en caliente de complementos (Zero-Shot)
**File:** [`10_hotswap_plugins_demo_es.ipynb`](../../notebooks/10_hotswap_plugins_demo_es.ipynb)

Demostraremos un flujo de trabajo en el que varios equipos de desarrollo aprenden de forma independiente complementos para "código" y "matemáticas" y los "fusionan físicamente (intercambian en caliente)" en el modelo base del entorno de producción después del hecho. Se ha demostrado que incluso después de la fusión, las pérdidas de todos los dominios son exactamente las mismas que durante el aprendizaje independiente (Zero Forgetting).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/10_hotswap_plugins_demo_es.ipynb)

---

### 🗑️ 11. Eliminación sináptica dinámica
**File:** [`11_synapse_deletion_demo_es.ipynb`](../../notebooks/11_synapse_deletion_demo_es.ipynb)

Demostramos la función de SRA, "deleción de sinapsis". Puede experimentar tanto la ``eliminación de complementos (pop_synapses)'', que elimina físicamente las sinapsis agregadas más tarde desde el final, como la ``purga de un dominio específico (clear_synapses)'', que borra y deshabilita de forma segura las sinapsis que no se comparten.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/11_synapse_deletion_demo_es.ipynb)



## 🤝 Contribución y Licencia

Licencia: [MIT License](../../LICENSE).
