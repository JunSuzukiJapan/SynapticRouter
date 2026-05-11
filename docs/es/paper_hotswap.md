# LLM Intercambiable en Caliente (Hotswappable LLM): Composición de Módulos Zero-Shot y Eliminación Quirúrgica de Conocimiento mediante Arquitectura de Enrutamiento Sináptico

**Jun Suzuki**, Independent Researcher

## Resumen (Abstract)
Los seres humanos pueden aprender posnatalmente nuevas habilidades (p. ej., andar en bicicleta o un nuevo idioma) y hacer que funcionen como circuitos independientes en el cerebro sin destruir el conocimiento existente. Sin embargo, los Modelos de Lenguaje Masivos (LLMs) actuales poseen una estructura monolítica que almacena densamente todo el conocimiento en un único espacio de parámetros, lo que hace que la adición independiente de conocimiento o la eliminación de recuerdos específicos sea extremadamente difícil.

En este artículo, aprovechamos el mecanismo de localización funcional de la Arquitectura de Enrutamiento Sináptico (SRA) para proponer un método para la inserción y extracción dinámica de circuitos neuronales (módulos), a saber, **Hot-Swap**. Hot-Swap es una operación bidireccional donde las sinapsis especializadas, entrenadas independientemente del modelo base, pueden ser **trasplantadas quirúrgicamente (Plug-In) en el modelo de producción sin ningún reentrenamiento**, y cuando ya no se necesitan, **memorias específicas pueden ser desconectadas de manera segura (Unplug)**. Los resultados experimentales muestran que, usando un mecanismo de máscara dura inspirado en las bases de datos de vectores, logramos **Cero Olvido (Zero Forgetting)**. Además, descubrimos y resolvimos el "problema del agujero negro" de los vectores cero peculiar de la similitud del coseno, realizando un ciclo de vida completo de IA modular: "Aprender → Trasplantar → Eliminar → Reutilizar".

## 1. Introducción (Introduction)

### 1.1 Limitaciones de los Modelos Monolíticos y Control de la Memoria
Desde "Attention Is All You Need", la arquitectura Transformer ha establecido una posición dominante. Sin embargo, los LLMs monolíticos enfrentan graves problemas con el control del conocimiento:

1. **Olvido Catastrófico (Catastrophic Forgetting)**: Cuando un modelo de propósito general se ajusta a un dominio específico, sus capacidades originales se destruyen.
2. **Costos de Entrenamiento Inflados**: Agregar conocimiento requiere reentrenar o fusionar todo el modelo, lo que dificulta el desarrollo paralelo.
3. **Imposibilidad de Eliminación de Conocimiento**: El "Machine Unlearning", olvidar solo conocimientos específicos, es una tarea abrumadora en modelos monolíticos.

### 1.2 Contribuciones de este Artículo
La Arquitectura de Enrutamiento Sináptico (SRA) [Suzuki, 2026] que propongo es una arquitectura donde la "localización funcional" surge autónomamente. Este artículo se centra en la **innovación operativa** de SRA—**Hot-Swap**—y reporta las siguientes contribuciones:

1. **Hot-Swap: Plug-In (Trasplante Quirúrgico)**: Implementación de un método donde la implementación se completa simplemente copiando físicamente los tensores de peso de un módulo especializado en los espacios vacíos de un modelo base.
2. **Hot-Swap: Unplug (Eliminación e Inactivación de Memoria)**: El diseño de dos APIs de eliminación: desconexión física (`pop_synapses`) y purga por inactivación mediante borrado a cero (`clear_synapses`).
3. **Prueba Experimental de Cero Olvido (Zero Forgetting)**: Demostración de que la pérdida (Loss) del modelo base coincide perfectamente después del trasplante/eliminación, gracias a un mecanismo de máscara dura (Pre-filtering).

## 2. Antecedentes: Arquitectura SRA

SRA es una arquitectura de aprendizaje continuo que imita el aislamiento espacial y el enrutamiento dinámico del cerebro.

### 2.1 Enrutador (Router)
El enrutador, el corazón de SRA, es una capa lineal única que determina "qué circuitos neuronales deben dispararse". Calcula la similitud del coseno entre el estado oculto $h$ y el vector de características $e_i$ de cada sinapsis.

### 2.2 Sinapsis Diminutas (Functional Modules)
Cada sinapsis consta de una Multi-Head Attention y un MLP extremadamente pequeños e independientes. Solo ejecutan cálculos las sinapsis seleccionadas por el enrutador.

### 2.3 Tronco Compartido (Shared Trunk)
Una condición crítica para que Hot-Swap funcione es el **enfoque del Tronco Compartido**. Todas las sinapsis especializadas comparten el mismo modelo base preentrenado y solo entrenan las porciones de la sinapsis de forma independiente. Esto previene la Divergencia de Representación entre modelos.

## 3. Hot-Swap: Plug-In (Trasplante de Módulos)

La primera operación es **Trasplantar (Plug-In)** un módulo especializado entrenado independientemente en el modelo base. Esta operación se completa únicamente mediante operaciones de tensores de PyTorch.

### 3.1 Método

```python
# hotswap_model: Modelo base en producción (con espacios vacíos añadidos)
# plugin_math: LLM especializado entrenado independientemente (nuevo circuito matemático)

with torch.no_grad():
    for l in range(layers):
        target_block = hotswap_model.blocks[l]
        src_block = plugin_math.blocks[l]
        
        # Copiar el vector de incrustación del enrutador (condición de disparo)
        target_block.router.synapse_emb.data[4:8] = src_block.router.synapse_emb.data
        
        # Copiar los pesos de la sinapsis
        target_block.w1.data[4:8] = src_block.w1.data
        target_block.w2.data[4:8] = src_block.w2.data
```

Simplemente asignamos los tensores a los espacios vacíos del modelo base. Dado que el modelo base está congelado, esta copia física funciona a la perfección.

## 4. Cero Olvido: Mecanismo de Máscara Dura

### 4.1 El Problema: Confusión del Enrutador
Simplemente trasplantar tensores corre el riesgo de que el enrutador confunda las sinapsis antiguas y nuevas.

### 4.2 Filtrado Previo (Pre-filtering)
Incorporé la técnica de **Pre-filtrado** de las bases de datos de vectores en la propagación hacia adelante del enrutador, diseñando un mecanismo para cerrar físicamente (máscara dura) "rutas a circuitos neuronales innecesarios para la tarea actual".

```python
# Propagación hacia adelante de la capa del Enrutador
logits = torch.einsum("btd,nd->btn", h_norm, emb_norm) * self.scale

# Pre-filtering: Cerrar rutas de sinapsis no autorizadas con -∞
logits = logits.masked_fill(~allowed_mask, float('-inf'))

# Top-K Routing
vals, idx = torch.topk(logits, k, dim=-1)
```

Gracias a esta máscara `-inf`, los pesos trasplantados son ignorados en la gráfica computacional cuando no son necesarios, garantizando matemáticamente **Cero Olvido**.

## 5. Hot-Swap: Unplug (Eliminación/Inactivación)

La segunda operación es **Desconectar (Unplug)** módulos. SRA resuelve esto con "desconexión física de conexiones sinápticas".

### 5.1 Desconexión Física (`pop_synapses`)
Si una sinapsis se vuelve innecesaria, se descarta físicamente cortando desde el final del tensor.

### 5.2 Purga por Inactivación (`clear_synapses`)
Para eliminar una sinapsis en el medio, borramos su contenido a cero (la convertimos en un espacio vacío). Al invalidar solo el contenido sin cambiar el tamaño del tensor, se mantiene la coherencia perfecta de los índices.

## 6. La Trampa de la Similitud del Coseno: El Problema del Agujero Negro

Al implementar la purga de inactivación, encontré un problema donde el vector de una sinapsis borrada a cero es $\mathbf{0}$, dando una similitud del coseno de $0.0$.
Si el puntaje de una sinapsis normal es negativo (p. ej., $-0.5$), **la sinapsis vacía ($0.0$) tendrá un puntaje más alto, causando que el enrutador seleccione la sinapsis vacía preferentemente.**

Para resolver esto, agregué una máscara de $-\infty$ para detectar sinapsis borradas a cero y bloquear completamente sus rutas, logrando la reutilización segura de los espacios.

## 7. El Ciclo de Vida Completo de la IA Modular
SRA permite la expansión infinita y el mantenimiento continuo de modelos, al igual que el cerebro biológico, a través de: `Aprender → Trasplantar → Operar → Eliminar → Reutilizar → Nuevo Trasplante`.

## 8. Discusión y 9. Conclusión
En este artículo, establecimos un método de "LLM Intercambiable en Caliente" que puede trasplantar y eliminar dinámicamente circuitos neuronales específicos aprovechando el mecanismo de localización funcional de SRA. Este enfoque representa una dirección prometedora para evolucionar los modelos de IA de "cajas negras monolíticas" a "sistemas modulares vitales donde partes de la inteligencia pueden ser insertadas y eliminadas físicamente de forma segura."

## Referencias
- Suzuki, J. (2026). [Neuro-inspired Synaptic Routing: Overcoming Catastrophic Forgetting via Dynamic Modular Pathways. *Technical Report*.](https://github.com/JunSuzukiJapan/SynapticRouter/blob/main/docs/paper_draft_en.md)
