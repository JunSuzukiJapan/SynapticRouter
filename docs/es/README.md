[English](../../README.md) | [日本語](../../README_JP.md)

# Synaptic Routing Architecture (SRA)

Synaptic Routing Architecture (SRA) es una novedosa arquitectura de red neuronal dinámica, dispersa y modular inspirada en el cerebro biológico (sinapsis).
En lugar de un Transformer masivo y estático, SRA enruta dinámicamente las entradas a "sinapsis" (pequeños módulos) apropiadas para lograr un aprendizaje más eficiente e inteligencia estructural.

## 🎯 Motivación

En los últimos años, mientras que los modelos de IA se han vuelto cada vez más masivos, las redes monolíticas enfrentan desafíos significativos, como "la escalada de recursos computacionales" y el "olvido catastrófico durante el aprendizaje multitarea".
SRA intenta resolver estos problemas utilizando un **enfoque disperso: "llamar y combinar dinámicamente solo los módulos minúsculos necesarios (sinapsis) dependiendo de la entrada"**. Esto permite aprender múltiples tareas con diferentes características dentro de la misma red sin interferencia, con el objetivo de lograr tanto la escalabilidad como una alta eficiencia de aprendizaje.

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

## 🤝 Contribución y Licencia

Licencia: [MIT License](../../LICENSE).
