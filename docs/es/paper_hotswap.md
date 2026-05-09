# Hotswappable LLM: Composición de Módulos Zero-Shot y Eliminación Quirúrgica de Conocimiento mediante Synaptic Routing Architecture

**Jun Suzuki**, Investigador Independiente

## Abstract
Los grandes modelos de lenguaje (LLMs) almacenan todo su conocimiento de forma densa en un único espacio de parámetros monolítico, haciendo extremadamente difícil agregar o eliminar conocimiento específico e imponiendo severas restricciones a la flexibilidad operativa. En este artículo, aprovecho la modularidad de la Synaptic Routing Architecture (SRA) para proponer y validar el **Hot-Swap** — una operación bidireccional de intercambio de módulos para redes neuronales. Hot-Swap permite **Plug-In**: insertar físicamente sinapsis especializadas entrenadas independientemente en un modelo base preentrenado **sin ningún reentrenamiento**, y **Unplug**: eliminar quirúrgicamente el conocimiento que ya no se necesita. Los experimentos demuestran que un mecanismo de enmascaramiento duro inspirado en técnicas de pre-filtrado de bases de datos vectoriales logra **Zero Forgetting**, donde la pérdida de salida del modelo base coincide exactamente hasta el punto decimal antes y después de la inserción. Además, descubro y resuelvo el "Problema del Agujero Negro" de los vectores cero en la similitud coseno encontrado durante la operación Unplug, estableciendo el ciclo de vida completo de la IA modular: Entrenar → Plug-In → Unplug → Reutilizar.

## 1. Introduction

### 1.1 Limitaciones operativas de los modelos monolíticos
Desde la introducción de "Attention Is All You Need", la arquitectura Transformer ha establecido una posición dominante en muchos dominios, incluido el procesamiento del lenguaje natural. Sin embargo, los LLMs monolíticos con cientos de miles de millones de parámetros enfrentan los siguientes desafíos operativos críticos:

1. **Olvido catastrófico (Catastrophic Forgetting)**: El ajuste fino de un modelo de propósito general en un dominio específico (ej. regulaciones internas, código especializado) destruye o degrada sus capacidades generales originales.
2. **Escalada de costos de entrenamiento**: Cada adición de conocimiento requiere reentrenar el modelo completo (o adaptadores como LoRA), haciendo impracticable el desarrollo completamente paralelo por múltiples equipos.
3. **Imposibilidad de eliminación de conocimiento**: El "Machine Unlearning" — olvidar selectivamente conocimiento específico — es extremadamente difícil en modelos monolíticos donde los parámetros están profundamente entrelazados, y los intentos de reentrenamiento a menudo destruyen capacidades no relacionadas.

### 1.2 Contribuciones
Previamente propuse la Synaptic Routing Architecture (SRA) [Suzuki, 2026], una arquitectura sparse compuesta de módulos independientes diminutos (sinapsis) y un enrutador ligero. Mientras que trabajos anteriores demostraron la capacidad de separación autónoma de tareas del enrutador, este artículo se enfoca en las **innovaciones operativas** habilitadas por la modularidad de SRA — el **Hot-Swap** bidireccional de módulos (inserción y eliminación) — reportando tres contribuciones:

1. **Hot-Swap: Plug-In (Inserción)**: Implementación y validación de un método donde el despliegue de modelos especializados entrenados independientemente requiere solo una copia física de tensores en los slots vacíos del modelo base.
2. **Hot-Swap: Unplug (Eliminación)**: Diseño de dos APIs de eliminación — separación física (`pop_synapses`) y purga por limpieza a cero (`clear_synapses`) — y el descubrimiento y resolución del "Problema del Agujero Negro" de los vectores cero en la similitud coseno.
3. **Prueba experimental de Zero Forgetting**: Demostración de que un mecanismo de enmascaramiento duro inspirado en el pre-filtrado de bases de datos vectoriales garantiza que la pérdida de salida del modelo base permanezca idéntica hasta el punto decimal antes y después tanto de la inserción como de la eliminación.

## 2. Background: SRA Architecture

SRA (Synaptic Routing Architecture) es una arquitectura dinámica y sparse inspirada en el cerebro biológico. Esta sección describe los componentes esenciales para comprender Hot-Swap (ver [Suzuki, 2026] para detalles).

### 2.1 Router
El enrutador, corazón de SRA, es una **única capa lineal** sin ningún mecanismo de Attention. Calcula la similitud coseno entre el estado oculto de entrada $h$ y el vector de características (embedding) $e_i$ de cada sinapsis, seleccionando las Top-k sinapsis.

$$\text{logits}_i = \frac{h \cdot e_i}{\|h\| \cdot \|e_i\|} \cdot \alpha$$

donde $\alpha$ es un factor de escala.

### 2.2 Tiny Synapses
Cada sinapsis es un módulo diminuto independiente compuesto por una pequeña capa Multi-Head Attention y un MLP. Solo las sinapsis seleccionadas por el enrutador ejecutan cálculos, logrando alta eficiencia computacional.

### 2.3 Tronco compartido (Shared Trunk)
Un prerrequisito crítico para Hot-Swap es el enfoque de **Tronco Compartido**. Todas las sinapsis especializadas se derivan del mismo modelo base preentrenado (capas de Embedding, capas de Attention), entrenando independientemente solo los componentes sinápticos. Esto previene la divergencia en las representaciones vectoriales internas (Representation Divergence) entre modelos y permite el trasplante de sinapsis mediante copia física.

## 3. Hot-Swap: Plug-In (Inserción de módulo)

La primera operación de Hot-Swap es **Plug-In** — insertar módulos especializados entrenados independientemente en el modelo base. Esta sección demuestra que el proceso de inserción consiste enteramente en operaciones de tensores PyTorch y es extremadamente simple.

### 3.1 Método

```python
# hotswap_model: Modelo base de producción (con slots vacíos agregados)
# plugin_math: LLM especializado en matemáticas entrenado independientemente por el equipo de matemáticas

with torch.no_grad():
    for l in range(layers):
        target_block = hotswap_model.blocks[l]
        src_block = plugin_math.blocks[l]
        
        # Copiar vectores de embedding del enrutador
        target_block.router.synapse_emb.data[4:8] = src_block.router.synapse_emb.data
        
        # Copiar pesos Expert (TinySynapse) (w1, w2)
        target_block.w1.data[4:8] = src_block.w1.data
        target_block.w2.data[4:8] = src_block.w2.data
```

Los tensores del modelo especializado entrenado se asignan simplemente de forma directa a índices específicos (slots vacíos) de los tensores del modelo base. Dado que SRA entrena solo las sinapsis manteniendo el conocimiento compartido del modelo base (capas de Attention, etc.) completamente congelado mediante el enfoque de Tronco Compartido, esta operación de copia física es válida.

### 3.2 Habilitación del desarrollo paralelo independiente
Este enfoque permite a un "equipo de Código" y un "equipo de Matemáticas" entrenar independientemente sus sinapsis especializadas basándose en el mismo modelo base con cero interferencia mutua. Después del entrenamiento, el despliegue se completa simplemente copiando en memoria los tensores de pesos en los slots vacíos del modelo base de producción.

## 4. Zero Forgetting: Enmascaramiento duro inspirado en el Pre-filtrado de bases de datos vectoriales

### 4.1 Desafío: Confusión del enrutador
La simple copia física de tensores conlleva el riesgo de que el enrutador confunda sinapsis antiguas y nuevas, alterando potencialmente la salida del modelo base.

### 4.2 Filtrado por metadatos en bases de datos vectoriales
Las bases de datos vectoriales modernas como Pinecone y Weaviate emplean filtrado por metadatos junto con la búsqueda semántica basada en similitud coseno.

- **Post-filtrado**: Excluye resultados no coincidentes después de la búsqueda Top-K. Propenso al "agotamiento K-NN" donde quedan resultados insuficientes después del filtrado.
- **Pre-filtrado**: Restringe el espacio de búsqueda mediante máscaras de metadatos **antes** de la búsqueda, realizando Top-K solo entre candidatos calificados. El ruido se elimina completamente.

### 4.3 Máscara dura de pre-ejecución de SRA
El enrutador SRA es esencialmente un **motor de búsqueda vectorial en memoria (Maximum Inner Product Search: MIPS)** que calcula productos escalares entre vectores de entrada y vectores de embedding de sinapsis.

Incorporé el Pre-filtrado de bases de datos vectoriales directamente en la pasada hacia adelante del enrutador. En el momento de la inferencia, se proporciona al modelo una máscara de metadatos que especifica el "conjunto de sinapsis permitidas para la tarea actual".

```python
# Pasada hacia adelante del enrutador
logits = torch.einsum("btd,nd->btn", h_norm, emb_norm) * self.scale

# Pre-filtrado: Establecer logits de sinapsis no autorizadas a -infinito
logits = logits.masked_fill(~allowed_mask, float('-inf'))

# Enrutamiento Top-K
vals, idx = torch.topk(logits, k, dim=-1)
```

Este Pre-filtrado por `masked_fill` asegura que el enrutador seleccione expertos solo entre las sinapsis permitidas. Sin importar cuántos pesos de otros modelos coexistan, son completamente ignorados en el grafo de computación, garantizando que **la pérdida del modelo base coincide exactamente hasta el punto decimal antes y después de la composición (interferencia matemáticamente cero)**.

### 4.4 Resultados experimentales
Comparé la Validation Loss del modelo base (modelo de lenguaje de 3 dominios Code/Math/Text) antes y después del Plug-In de sinapsis especializadas entrenadas independientemente. La pérdida coincidió exactamente hasta el punto decimal, demostrando empíricamente Zero Forgetting.

## 5. Hot-Swap: Unplug (Eliminación de módulo)

La segunda operación de Hot-Swap es **Unplug** — eliminar del modelo base los módulos que ya no son necesarios. Si el conocimiento puede ser "conectado", la capacidad de "desconectarlo" es igualmente esencial. El Machine Unlearning en modelos monolíticos es extremadamente difícil debido al entrelazamiento complejo de parámetros, pero la estructura modular de SRA resuelve este problema mediante operaciones físicas.

### 5.2 Enfoque 1: Eliminación física (pop_synapses)
Cuando las sinapsis añadidas por Hot-Swap ya no son necesarias, se recortan físicamente desde el final del tensor.

```python
def pop_synapses(self, num_drop: int):
    self.synapse_emb = nn.Parameter(self.synapse_emb.data[:-num_drop])
```

**Ventaja**: El uso de VRAM se reduce físicamente, y el modelo puede restaurarse completamente a su estado previo a la adición — como desinstalar un controlador del SO, una parte física del cerebro de la IA puede ser removida.

### 5.3 Enfoque 2: Purga por limpieza a cero (clear_synapses)
Cuando se desconectan sinapsis en índices intermedios en lugar del final, la eliminación física desplazaría todos los índices de sinapsis subsiguientes, rompiendo el sistema de control por máscara de metadatos. En su lugar, el contenido de las sinapsis se limpia a cero para crear un "slot vacío".

```python
def clear_synapses(self, indices_to_clear: list[int]):
    for idx in indices_to_clear:
        self.synapse_emb.data[idx].zero_()
        self.w1.data[idx].zero_()
        self.w2.data[idx].zero_()
```

Al invalidar solo el contenido del slot sin cambiar el tamaño del tensor, la integridad de los índices se preserva perfectamente. Los slots vacíos pueden reutilizarse posteriormente sobrescribiéndolos con nuevas sinapsis vía Hot-Swap.

## 6. The Cosine Similarity Trap: El problema del agujero negro del vector cero

### 6.1 Descubrimiento
Al implementar la purga por limpieza a cero para la operación Unplug, encontré un bug crítico donde **la salida colapsó completamente**.

### 6.2 Análisis de la causa raíz
El enrutador SRA realiza el enrutamiento usando similitud coseno. El vector de embedding de una sinapsis limpiada a cero se convierte en $\mathbf{0}$, que permanece como $\mathbf{0}$ incluso después de la normalización. La similitud coseno entre cualquier vector de entrada $h$ y el vector cero es $0.0$.

El problema surge porque la similitud coseno tiene un rango de $[-1.0, 1.0]$. Si la similitud coseno de una sinapsis válida es negativa (ej. $-0.5$), **la sinapsis vacía ($0.0$) obtiene una puntuación matemáticamente más alta, causando que el enrutador seleccione preferentemente la sinapsis vacía**.

$$\text{similarity}(h, \mathbf{0}) = 0.0 > -0.5 = \text{similarity}(h, e_{\text{valid}})$$

Los datos son "absorbidos y desaparecen en" lo que debería ser un slot vacío inexistente — un comportamiento similar a un agujero negro.

### 6.3 Solución: Bloqueo completo vía enmascaramiento -∞
Agregué procesamiento de máscara para detectar y excluir sinapsis limpiadas a cero en la pasada hacia adelante del enrutador.

```python
logits = torch.einsum("btd,nd->btn", h_norm, emb_norm) * self.scale

# Detectar sinapsis limpiadas a cero
is_cleared = (full_emb == 0).all(dim=-1)
if is_cleared.any():
    logits = logits.masked_fill(is_cleared.view(1, 1, -1), float('-inf'))
```

La máscara $-\infty$ hace matemáticamente imposible que las sinapsis vacías sean seleccionadas, sin importar cuán bajas sean las puntuaciones de otras sinapsis.

## 7. The Complete Lifecycle of Modular AI

Los mecanismos descritos anteriormente permiten a SRA realizar el ciclo de vida completo de la IA modular:

```
Entrenar → Hot-Swap (Componer) → Servir
   ↓                                ↓
Desarrollo                    Purgar (Eliminar)
paralelo independiente              ↓
                            Reutilización de slot
                                    ↓
                            Nuevo Hot-Swap
```

1. **Entrenar**: Múltiples equipos comparten un modelo base y desarrollan independientemente sus sinapsis especializadas en paralelo.
2. **Componer**: Los tensores entrenados se copian físicamente en el modelo base de producción para el despliegue.
3. **Servir**: La inferencia se ejecuta con Zero Forgetting garantizado por el pre-filtrado con máscara dura.
4. **Eliminar**: Las sinapsis innecesarias se eliminan físicamente o se limpian a cero para purga.
5. **Reutilizar**: Los slots vacíos se reutilizan insertando nuevas sinapsis especializadas por Hot-Swap.

## 8. Discussion

### 8.1 Divergencia de representación (Representation Divergence)
El **prerrequisito absoluto** para Hot-Swap es que todas las sinapsis especializadas se deriven del mismo modelo base preentrenado (Tronco Compartido). Trasplantar sinapsis entre modelos entrenados de forma completamente independiente causa el colapso del enrutamiento debido a la divergencia en las representaciones vectoriales internas.

### 8.2 Super Enrutador como alternativa
Para relajar la restricción del Tronco Compartido, se ha validado un enfoque donde modelos completos independientes son encapsulados y orquestados por un Super Enrutador usando Gumbel-Softmax. Este enfoque logra un enrutamiento duro perfecto $1.0$ vs $0.0$, permitiendo la conmutación dinámica completa de recursos computacionales incluso entre modelos con diferentes arquitecturas.

### 8.3 Riesgos de seguridad
La capacidad Hot-Swap introduce nuevos vectores de amenazas de seguridad debido a su propiedad de cargar dinámicamente archivos de pesos desde fuera de un sistema en ejecución. Los riesgos clave incluyen: (1) ejecución arbitraria de código vía exploits Pickle, (2) inyección maliciosa de pesos (Backdoor Injection), (3) secuestro de enrutamiento vía falsificación de claves del enrutador, y (4) ataques DoS vía swap thrashing. Se recomiendan mitigaciones como el formato obligatorio `safetensors`, firma criptográfica de sinapsis y limitación de tasa.

### 8.4 Limitaciones actuales y trabajo futuro
Esta investigación está en etapa experimental con modelos de pequeña escala ($d_\text{model}=128$, $n_\text{layers}=4$). La validación en LLMs de clase 10B sigue siendo un desafío futuro importante. Además, el problema de sincronización del enrutador — la necesidad potencial de aprendizaje adaptativo de las claves del enrutador al agregar sinapsis con capacidades completamente nuevas — requiere investigación adicional.

## 9. Conclusion

En este artículo, propuse y validé métodos para hacer los LLMs "Hotswappable" (dinámicamente conectables) aprovechando la modularidad de SRA (Synaptic Routing Architecture). La operación Plug-In de Hot-Swap completa el despliegue solo mediante copia física de tensores, mientras que la operación Unplug establece dos enfoques de eliminación: separación física y purga por limpieza a cero. A través de un mecanismo de enmascaramiento duro inspirado en el Pre-filtrado de bases de datos vectoriales, Zero Forgetting se garantiza matemáticamente. Al descubrir y resolver el "Problema del Agujero Negro" de los vectores cero en la similitud coseno encontrado durante Unplug, se logra la reutilización segura de slots.

En una era donde los modelos continúan creciendo y volviéndose más opacos, el enfoque "Hotswappable LLM" — que permite el control físico de conexión y desconexión de componentes de inteligencia — representa una dirección extremadamente prometedora para la mantenibilidad, seguridad y eficiencia operativa de los modelos.

## References

- Suzuki, J. (2026). [All You Need Is Router: Dynamic Sparse Modularity in Neural Networks. *Technical Report*.](https://github.com/JunSuzukiJapan/SynapticRouter/blob/main/docs/paper_draft_en.md)
- Vaswani, A. et al. (2017). Attention Is All You Need. *NeurIPS*.
- Shazeer, N. et al. (2017). Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer. *ICLR*.
- Jiang, A.Q. et al. (2024). Mixtral of Experts. *arXiv:2401.04088*.
- Jang, E. et al. (2017). Categorical Reparameterization with Gumbel-Softmax. *ICLR*.

## Appendix: Interactive Demos

Los procesos completos de Hot-Swap y eliminación de sinapsis descritos en este artículo pueden experimentarse interactivamente en los siguientes notebooks de Google Colab.

- **Demo de composición de sinapsis Hot-Swap (Entrenamiento base → Entrenamiento independiente → Composición → Prueba de Zero Forgetting)**<br>
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/10_hotswap_plugins_demo.ipynb)
- **Demo de eliminación de sinapsis (Eliminación física → Limpieza a cero → Resolución del problema del agujero negro)**<br>
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/11_synapse_deletion_demo.ipynb)

## Appendix: Related Technical Documents

- **[El futuro de SRA: Hot-Swap dinámico y extensibilidad](./sra_future_hotswap_ja.md)** — Discusión sobre operación de sinapsis en modo casete, personalización y aprendizaje distribuido.
- **[Riesgos de seguridad en el Hot-Swap de SRA](./sra_security_risks_hotswap_ja.md)** — Vectores de amenazas incluyendo Pickle Exploit, Backdoor Injection, ataques DoS y estrategias de mitigación.
- **[Divergencia de representación y enrutamiento jerárquico](./sra_representation_divergence_ja.md)** — Enfoque de Tronco Compartido y soluciones de Super Enrutador.
- **[Comparación de enrutamiento duro para el enrutador jerárquico SRA](./sra_hierarchical_hard_routing_ja.md)** — Experimentos comparativos de Soft / STE / Gumbel-Softmax.
