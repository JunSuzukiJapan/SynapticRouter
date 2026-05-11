# Enrutamiento Sináptico Bioinspirado: Superando el Olvido Catastrófico mediante Vías Modulares Dinámicas

**Jun Suzuki**, Independent Researcher

## Resumen (Abstract)
El cerebro humano puede aprender y ejecutar tareas fundamentalmente diferentes (como caminar, hablar y calcular) sin interferencia mutua. Esto se debe a que los circuitos neuronales (sinapsis) del cerebro se enrutan dinámicamente según la tarea, manteniendo una "localización funcional" espacialmente aislada. Por el contrario, cuando las Redes Neuronales Artificiales (ANNs) aprenden múltiples tareas dentro de una red monolítica única, sufren de "Olvido Catastrófico", donde las memorias pasadas son destruidas.

En este artículo, proponemos la "Arquitectura de Enrutamiento Sináptico (SRA)", un modelo de aprendizaje continuo inspirado en los mecanismos biológicos de formación dinámica de sinapsis y aislamiento espacial. SRA consta de un "Enrutador (Router)" de una sola capa extremadamente simple y múltiples micromódulos independientes (Sinapsis). A través de nuestros experimentos, demostramos que SRA puede identificar de forma autónoma la naturaleza de una tarea a partir de la entrada, sin que se le proporcione un ID de tarea externo durante la inferencia, y **aprender tanto el enrutamiento (selección de vías) como las representaciones de la tarea de forma completamente simultánea de extremo a extremo (End-to-End).** Mostramos que, sin congelación artificial de pesos ni algoritmos evolutivos complejos, surge una localización funcional autónoma dentro del modelo, evitando por completo el olvido catastrófico.

## 1. Introducción (Introduction)
En el campo del aprendizaje profundo, el "Aprendizaje Continuo" (donde un modelo adquiere conocimiento nuevo continuamente) es una de las mayores barreras para lograr la Inteligencia Artificial General (AGI). Las redes monolíticas, como los actuales modelos masivos de Transformer, inevitablemente olvidan el conocimiento aprendido previamente cuando se ajustan con datos de nuevos dominios.

Para abordar este problema, nuestra investigación se centra en la "Localización Funcional" del cerebro. Al igual que las áreas del lenguaje y motoras del cerebro usan circuitos físicamente distintos para evitar interferencias, SRA está diseñado como una arquitectura que puede encender/apagar dinámicamente (conectar/desconectar) redes independientes diminutas (sinapsis) a través de un mecanismo de enrutamiento dinámico.

## 2. Trabajo Relacionado y la Novedad de SRA
Los enfoques existentes para prevenir el olvido catastrófico incluyen métodos de regularización como EWC (Consolidación de Peso Elástico), que penalizan las actualizaciones de los pesos que fueron importantes para tareas anteriores. Sin embargo, estos métodos están limitados por la capacidad del modelo y finalmente alcanzan sus límites a medida que aumenta el número de tareas.

Un enfoque más "estructural y modular" similar a SRA es **PathNet (2017)** de Google DeepMind. PathNet proporciona numerosos módulos y utiliza algoritmos genéticos para descubrir "rutas" para cada tarea, congelando los pesos después de aprender para evitar el olvido.

### La Ventaja Abrumadora de SRA (Aprendizaje Simultáneo)
En comparación con los enfoques convencionales como PathNet, la novedad fundamental de SRA radica en su capacidad de **"aprender simultáneamente el descubrimiento de vías (enrutamiento) y las representaciones de los módulos de manera diferenciable y de extremo a extremo".**

1. **Enrutamiento Autónomo (Agnóstico a la Tarea):** PathNet requiere que al modelo se le diga explícitamente "qué tarea está realizando (ID de tarea)" durante la inferencia. Por el contrario, el enrutador lineal único de SRA determina autónomamente el dominio (p. ej., "esto es una tarea matemática", "esto es una tarea de lenguaje") basándose en la similitud del coseno de las características de entrada y lo enruta a la sinapsis apropiada.
2. **Eliminación de la Congelación de Pesos y Algoritmos Evolutivos:** En lugar de requerir costos computacionales masivos como los algoritmos genéticos, SRA permite que el enrutador y las sinapsis aprendan cooperativamente usando solo Retropropagación estándar.
3. **Surgimiento de Localización Funcional Dinámica:** Debido a que el enrutador aprende espontáneamente vías tales que "tareas similares usan la misma sinapsis" y "tareas diferentes usan diferentes sinapsis", el aislamiento espacial (localización funcional) surge naturalmente a través de la activación dispersa sin la necesidad de congelación artificial de pesos.

## 3. Arquitectura (Neuro-inspired Design)
SRA es una arquitectura dinámica y dispersa (sparse) que imita la formación de sinapsis del cerebro biológico.

### 3.1 El Enrutador (Dynamic Synaptic Formation)
El enrutador, el corazón de SRA, determina "qué circuitos neuronales deben dispararse" en función de la información de entrada. El enrutador es una capa lineal simple que calcula la similitud del coseno entre las características de entrada y el "vector de incrustación (embedding)" de cada sinapsis, determinando las Top-k sinapsis que mejor coinciden (disparan).

### 3.2 Sinapsis Diminutas (Functional Modules)
Cada sinapsis consta de una Multi-Head Attention y un MLP independientes y extremadamente pequeños. Solo las sinapsis instruidas a "disparar" por el enrutador ejecutan cálculos; los parámetros de otras sinapsis no sufren interferencia. Esto proporciona resistencia al olvido similar al aislamiento espacial del cerebro.

### 3.3 Diagrama de Arquitectura

```mermaid
graph TD
    X[Input Token] --> Base[Residual Base]
    X --> Norm[LayerNorm]
    
    Norm --> Router["Router (Synaptic Routing)"]
    Norm --> SynapseSpace
    
    subgraph Synapse Space (Functional Modules)
        SynapseSpace((Select Top-k))
        S1["Synapse 0<br/>(Mini-Transformer)"]
        S2["Synapse 1<br/>(Mini-Transformer)"]
        S3["Synapse ..."]
        Sn["Synapse 15<br/>(Mini-Transformer)"]
    end
    
    Router -- "Routing Weights" --> SynapseSpace
    SynapseSpace --> S1
    SynapseSpace --> S2
    SynapseSpace -.-> Sn
    
    S1 --> Combine((Weighted Sum))
    S2 --> Combine
    Sn -.-> Combine
    
    Base --> Combine
    Combine --> Out[Output Representation]
```

## 4. Experimento 1: Razonamiento Algorítmico
Para verificar si el aprendizaje simultáneo del enrutador y los módulos puede discernir las propiedades de la tarea, entrenamos el modelo concurrentemente en cuatro tareas de razonamiento algorítmico completamente diferentes (`copy`, `reverse`, `paren`, `addmod`) sin proporcionar ningún ID de tarea.

### Resultados y Surgimiento de la Localización Funcional
Después de 10,000 pasos de aprendizaje simultáneo, el modelo alcanzó un 100% de precisión en todas las tareas. Además, el análisis de la distribución de las vías del enrutador reveló un resultado sorprendente.

**Agrupación de Tareas por el Enrutador:**
- **Área de Manipulación de Secuencias**: `COPY` y `REVERSE` (Compartieron la misma sinapsis con una similitud de 0.969)
- **Área de Cálculo/Lógica**: `PAREN` y `ADDMOD` (Compartieron la misma sinapsis con una similitud de 0.858)
- La similitud entre los dos grupos anteriores fue extremadamente baja (0.029 - 0.336), mostrando una clara separación funcional.

Sin que los humanos proporcionaran ninguna instrucción (IDs de tarea), el aprendizaje simultáneo del enrutador agrupó autónomamente las "tareas de reordenamiento de secuencias" y las "tareas que requieren lógica", **resultando en el surgimiento de segregación modular similar a la localización funcional del cerebro.**

## 5. Experimento 2: Modelado de Lenguaje entre Dominios
A continuación, realizamos un "modelado de lenguaje cruzado entre dominios" más desafiante. Entrenamos el modelo simultáneamente en tres dominios con gramáticas y vocabularios completamente diferentes: `Code` (Python), `Math` (LaTeX) y `Text` (Lenguaje Natural).

### Resultados
A pesar de solo 1,000 pasos de entrenamiento, el enrutador completó la formación de "circuitos especializados por dominio":
- Área `Code`: Dominada por la **Sinapsis 8**
- Área `Math`: Manejada por las **Sinapsis 10 y 13**
- Área `Text`: Manejada por las **Sinapsis 0 y 15**

En una situación donde un modelo monolítico experimentaría un olvido catastrófico, SRA minimizó con éxito la interferencia mutua asignando sinapsis especializadas a cada dominio a través del enrutador.

## 6. Experimento 3: Traducción Automática Multilingüe
Realizamos traducción automática multilingüe usando tres idiomas con diferentes estructuras sintácticas (Inglés: SVO, Francés: SVO, Japonés: SOV).

### Resultados y Discusión
El análisis de la tasa de utilización de sinapsis reveló la formación autónoma de una "sinapsis compartida SVO" que se activaba frecuentemente durante la traducción entre inglés y francés, y una "sinapsis especializada SOV" cuyo uso se disparaba solo durante la traducción al japonés (SOV). Esto indica que el enrutador no solo usó etiquetas de idioma, sino que adquirió las "reglas de orden de palabras y sintácticas" subyacentes esenciales, cambiando dinámicamente los circuitos neuronales en función de ellas.

## 7. Experimento 4: Decision Transformer (RL Fuera de Línea)
Finalmente, para demostrar su adaptabilidad a dominios no relacionados con el lenguaje natural, validamos SRA como un Decision Transformer aprendiendo datos de trayectoria de aprendizaje por refuerzo (RL) fuera de línea. Entrenamos el modelo en registros de dos entornos con reglas completamente diferentes (la tarea del "Tesoro" y la tarea de "Escape").

### Resultados y Discusión
La visualización del enrutamiento para cada token confirmó un fenómeno sorprendente: **completa separación funcional de "Percepción" y "Política".**
- **Tokens de Percepción (Estado)**: Para los tokens de estado, el enrutador los dirigió a una **sinapsis común (Experto 1) sin excepción**, independientemente del tipo de tarea. Esto significa que se formó un "área de percepción espacial" compartida.
- **Tokens de Acción (Decisión de Política)**: Por otro lado, para los tokens de acción, las vías se ramificaron claramente en diferentes sinapsis para la tarea del Tesoro y la de Escape.

La estructura modular ideal de "percibir el entorno con los mismos ojos, pero tomar decisiones con diferentes cerebros" se adquirió sin diseño humano a través del aprendizaje de enrutamiento simultáneo.

## 8. Conclusión (Conclusion)
En este artículo, a través de SRA, demostramos el potencial para un cambio de paradigma de las "redes neuronales estáticas tradicionales" que comparten todos los parámetros en todas las tareas, a una "red modular equipada con aislamiento espacial biológico y enrutamiento dinámico".

El mayor avance es que **al realizar un aprendizaje simultáneo de extremo a extremo de la selección de vías del enrutador y el aprendizaje de representación de los módulos, ha sido posible el aprendizaje continuo independiente de la tarea (Task-Agnostic).** SRA supera el olvido catastrófico y permite insertar un número infinito de nuevas tareas (sinapsis), representando un paso crucial hacia una Inteligencia Artificial General (AGI) escalable.

## Referencias (References)

- Suzuki, J. (2026). [Neuro-inspired Synaptic Routing: Overcoming Catastrophic Forgetting via Dynamic Modular Pathways. *Technical Report*.](https://github.com/JunSuzukiJapan/SynapticRouter/blob/main/docs/paper_draft_en.md)
- Vaswani, A. et al. (2017). Attention Is All You Need. *NeurIPS*.
- Shazeer, N. et al. (2017). Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer. *ICLR*.
- Jiang, A.Q. et al. (2024). Mixtral of Experts. *arXiv:2401.04088*.
- Jang, E. et al. (2017). Categorical Reparameterization with Gumbel-Softmax. *ICLR*.
