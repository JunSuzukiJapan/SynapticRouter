# Análisis de Enrutamiento en Modelado de Lenguaje de Dominio Cruzado (Código / Matemáticas / Texto)

## Descripción General
Llevamos a cabo un experimento de "modelado de lenguaje" usando SRA para aprender tres dominios (Código, Matemáticas, Texto) con diferentes vocabularios simultáneamente.

## Evaluación de Calidad de Inferencia
Después de 1000 pasos de entrenamiento, se lograron predicciones precisas en los tres dominios sin interferencia.

## Análisis de Especialización de Sinapsis
1. **Homogeneización en el estado inicial**: Al principio, cada tarea utilizaba las 16 sinapsis de manera uniforme.
2. **Congelación y Especialización**: Después de cierta etapa, el enrutamiento se separó.
   - `Code` utilizó principalmente la **Sinapsis 8**.
   - `Math` utilizó principalmente las **Sinapsis 10 y 13**.
   - `Text` confió en las **Sinapsis 0 y 15**.

## Conclusión
En el modelado de múltiples dominios, el modelo SRA descubrió de forma autónoma el "uso de las sinapsis correctas para cada tarea", logrando una alta precisión y previniendo el olvido catastrófico.
