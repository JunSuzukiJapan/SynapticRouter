# Synaptic Routing Architecture (SRA) – Conversation Summary

## ■ Overview
This document summarizes the design discussion of a novel AI architecture inspired by biological synapses.

---

## ■ Core Idea
- Replace dense Transformer computation with modular synapse units
- Use a Router to dynamically select which synapses to activate
- Introduce stateful, dynamic computation instead of static weights

---

## ■ Architecture Components

### ● Synapse
- Small neural module (MLP or small Transformer)
- Acts as a local computation unit
- Can specialize over time

### ● Router
- Selects top-k synapses for each input
- Outputs routing weights

### ● Synapse Space
- Each synapse has an embedding
- Distance defines similarity

---

## ■ Learning

### ● Learning Rule
Δw = trace × routing × reward

---

## ■ Reward Design
R =
- task accuracy
- curiosity
- efficiency
- specialization

---

## ■ VRAM-Based Model Sizing (8GB GPU)

- Total params: 100M – 200M
- Synapses: 256 – 1024
- Per synapse: 100K – 500K params
- Router: 5M – 30M params

---

## ■ Conclusion

This architecture shifts from:
→ large static function

to:
→ dynamic modular system
