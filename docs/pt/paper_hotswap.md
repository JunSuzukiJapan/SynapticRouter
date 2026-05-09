# Hotswappable LLM: Composição de Módulos Zero-Shot e Exclusão Cirúrgica de Conhecimento via Synaptic Routing Architecture

**Jun Suzuki**, Pesquisador Independente

## Abstract
Os grandes modelos de linguagem (LLMs) armazenam todo o conhecimento de forma densa em um único espaço de parâmetros monolítico, tornando extremamente difícil adicionar ou remover conhecimento específico e impondo severas restrições à flexibilidade operacional. Neste artigo, aproveito a modularidade da Synaptic Routing Architecture (SRA) para propor e validar o **Hot-Swap** — uma operação bidirecional de troca de módulos para redes neurais. O Hot-Swap permite o **Plug-In**: inserir fisicamente sinapses especializadas treinadas independentemente em um modelo base pré-treinado **sem qualquer retreinamento**, e o **Unplug**: remover cirurgicamente o conhecimento que não é mais necessário. Os experimentos demonstram que um mecanismo de mascaramento rígido inspirado em técnicas de pré-filtragem de bancos de dados vetoriais atinge o **Zero Forgetting**, onde a perda de saída do modelo base corresponde exatamente até o ponto decimal antes e depois da inserção. Além disso, descubro e resolvo o "Problema do Buraco Negro" dos vetores zero na similaridade do cosseno encontrado durante a operação Unplug, estabelecendo o ciclo de vida completo da IA modular: Treinar → Plug-In → Unplug → Reutilizar.

## 1. Introduction

### 1.1 Limitações operacionais dos modelos monolíticos
Desde a introdução do "Attention Is All You Need", a arquitetura Transformer estabeleceu uma posição dominante em muitos domínios, incluindo o processamento de linguagem natural. No entanto, LLMs monolíticos com centenas de bilhões de parâmetros enfrentam os seguintes desafios operacionais críticos:

1. **Esquecimento catastrófico (Catastrophic Forgetting)**: O ajuste fino de um modelo de propósito geral em um domínio específico (ex. regulamentos internos, código especializado) destrói ou degrada suas capacidades gerais originais.
2. **Escalada dos custos de treinamento**: Cada adição de conhecimento requer o retreinamento de todo o modelo (ou adaptadores como LoRA), tornando impraticável o desenvolvimento totalmente paralelo por múltiplas equipes.
3. **Impossibilidade de exclusão de conhecimento**: O "Machine Unlearning" — esquecer seletivamente conhecimento específico — é extremamente difícil em modelos monolíticos onde os parâmetros estão profundamente entrelaçados, e tentativas de retreinamento frequentemente destroem capacidades não relacionadas.

### 1.2 Contribuições
Previamente propus a Synaptic Routing Architecture (SRA) [Suzuki, 2026], uma arquitetura esparsa composta de módulos independentes diminutos (sinapses) e um roteador leve. Enquanto trabalhos anteriores demonstraram a capacidade de separação autônoma de tarefas do roteador, este artigo foca nas **inovações operacionais** habilitadas pela modularidade do SRA — o **Hot-Swap** bidirecional de módulos (inserção e remoção) — reportando três contribuições:

1. **Hot-Swap: Plug-In (Inserção)**: Implementação e validação de um método onde o deploy de modelos especializados treinados independentemente requer apenas uma cópia física de tensores nos slots vazios do modelo base.
2. **Hot-Swap: Unplug (Remoção)**: Design de duas APIs de remoção — separação física (`pop_synapses`) e purga por limpeza a zero (`clear_synapses`) — e a descoberta e resolução do "Problema do Buraco Negro" dos vetores zero na similaridade do cosseno.
3. **Prova experimental do Zero Forgetting**: Demonstração de que um mecanismo de mascaramento rígido inspirado na pré-filtragem de bancos de dados vetoriais garante que a perda de saída do modelo base permaneça idêntica até o ponto decimal antes e depois tanto da inserção quanto da remoção.

## 2. Background: SRA Architecture

SRA (Synaptic Routing Architecture) é uma arquitetura dinâmica e esparsa inspirada no cérebro biológico. Esta seção descreve os componentes essenciais para entender o Hot-Swap (ver [Suzuki, 2026] para detalhes).

### 2.1 Router
O roteador, coração do SRA, é uma **única camada linear** sem qualquer mecanismo de Attention. Ele calcula a similaridade do cosseno entre o estado oculto de entrada $h$ e o vetor de características (embedding) $e_i$ de cada sinapse, selecionando as Top-k sinapses.

$$\text{logits}_i = \frac{h \cdot e_i}{\|h\| \cdot \|e_i\|} \cdot \alpha$$

onde $\alpha$ é um fator de escala.

### 2.2 Tiny Synapses
Cada sinapse é um módulo diminuto independente composto por uma pequena camada Multi-Head Attention e um MLP. Apenas as sinapses selecionadas pelo roteador executam cálculos, atingindo alta eficiência computacional.

### 2.3 Tronco compartilhado (Shared Trunk)
Um pré-requisito crítico para o Hot-Swap é a abordagem do **Tronco Compartilhado**. Todas as sinapses especializadas são derivadas do mesmo modelo base pré-treinado (camadas de Embedding, camadas de Attention), treinando independentemente apenas os componentes sinápticos. Isso previne a divergência nas representações vetoriais internas (Representation Divergence) entre modelos e permite o transplante de sinapses via cópia física.

## 3. Hot-Swap: Plug-In (Inserção de módulo)

A primeira operação do Hot-Swap é o **Plug-In** — inserir módulos especializados treinados independentemente no modelo base. Esta seção demonstra que o processo de inserção consiste inteiramente em operações tensoriais PyTorch e é extremamente simples.

### 3.1 Método

```python
# hotswap_model: Modelo base de produção (com slots vazios adicionados)
# plugin_math: LLM especializado em matemática treinado independentemente pela equipe de matemática

with torch.no_grad():
    for l in range(layers):
        target_block = hotswap_model.blocks[l]
        src_block = plugin_math.blocks[l]
        
        # Copiar vetores de embedding do roteador
        target_block.router.synapse_emb.data[4:8] = src_block.router.synapse_emb.data
        
        # Copiar pesos Expert (TinySynapse) (w1, w2)
        target_block.w1.data[4:8] = src_block.w1.data
        target_block.w2.data[4:8] = src_block.w2.data
```

Os tensores do modelo especializado treinado são simplesmente atribuídos diretamente a índices específicos (slots vazios) dos tensores do modelo base. Como o SRA treina apenas sinapses mantendo o conhecimento compartilhado do modelo base (camadas de Attention, etc.) completamente congelado via a abordagem do Tronco Compartilhado, essa operação de cópia física é válida.

### 3.2 Habilitando o desenvolvimento paralelo independente
Esta abordagem permite que uma "equipe de Código" e uma "equipe de Matemática" treinem independentemente suas sinapses especializadas com base no mesmo modelo base com zero interferência mútua. Após o treinamento, o deploy é completado simplesmente copiando em memória os tensores de pesos nos slots vazios do modelo base de produção.

## 4. Zero Forgetting: Mascaramento rígido inspirado na Pré-filtragem de bancos de dados vetoriais

### 4.1 Desafio: Confusão do roteador
A simples cópia física de tensores arrisca que o roteador confunda sinapses antigas e novas, potencialmente alterando a saída do modelo base.

### 4.2 Filtragem por metadados em bancos de dados vetoriais
Bancos de dados vetoriais modernos como Pinecone e Weaviate empregam filtragem por metadados junto com busca semântica baseada em similaridade do cosseno.

- **Pós-filtragem**: Exclui resultados não correspondentes após a busca Top-K. Propenso ao "esgotamento K-NN" onde resultados insuficientes permanecem após a filtragem.
- **Pré-filtragem**: Restringe o espaço de busca via máscaras de metadados **antes** da busca, realizando Top-K apenas entre candidatos qualificados. O ruído é completamente eliminado.

### 4.3 Máscara rígida de pré-execução do SRA
O roteador SRA é essencialmente um **motor de busca vetorial em memória (Maximum Inner Product Search: MIPS)** que calcula produtos escalares entre vetores de entrada e vetores de embedding de sinapses.

Incorporei a Pré-filtragem de bancos de dados vetoriais diretamente na passagem direta do roteador. No momento da inferência, uma máscara de metadados especificando o "conjunto de sinapses permitidas para a tarefa atual" é fornecida ao modelo.

```python
# Passagem direta do roteador
logits = torch.einsum("btd,nd->btn", h_norm, emb_norm) * self.scale

# Pré-filtragem: Definir logits de sinapses não autorizadas para -infinito
logits = logits.masked_fill(~allowed_mask, float('-inf'))

# Roteamento Top-K
vals, idx = torch.topk(logits, k, dim=-1)
```

Esta Pré-filtragem por `masked_fill` garante que o roteador selecione especialistas apenas entre as sinapses permitidas. Independentemente de quantos pesos de outros modelos coexistam, eles são completamente ignorados no grafo computacional, garantindo que **a perda do modelo base corresponda exatamente até o ponto decimal antes e depois da composição (interferência matematicamente zero)**.

### 4.4 Resultados experimentais
Comparei a Validation Loss do modelo base (modelo de linguagem de 3 domínios Code/Math/Text) antes e depois do Plug-In de sinapses especializadas treinadas independentemente. A perda correspondeu exatamente até o ponto decimal, demonstrando empiricamente o Zero Forgetting.

## 5. Hot-Swap: Unplug (Remoção de módulo)

A segunda operação do Hot-Swap é o **Unplug** — remover do modelo base os módulos que não são mais necessários. Se o conhecimento pode ser "plugado", a capacidade de "desplugá-lo" é igualmente essencial. O Machine Unlearning em modelos monolíticos é extremamente difícil devido ao entrelaçamento complexo de parâmetros, mas a estrutura modular do SRA resolve este problema através de operações físicas.

### 5.2 Abordagem 1: Remoção física (pop_synapses)
Quando sinapses adicionadas por Hot-Swap não são mais necessárias, são fisicamente cortadas do final do tensor.

```python
def pop_synapses(self, num_drop: int):
    self.synapse_emb = nn.Parameter(self.synapse_emb.data[:-num_drop])
```

**Vantagem**: O uso de VRAM é fisicamente reduzido, e o modelo pode ser totalmente restaurado ao seu estado pré-adição — como desinstalar um driver do SO, uma parte física do cérebro da IA pode ser removida.

### 5.3 Abordagem 2: Purga por limpeza a zero (clear_synapses)
Quando se desplugam sinapses em índices intermediários em vez do final, a exclusão física deslocaria todos os índices de sinapses subsequentes, quebrando o sistema de controle por máscara de metadados. Em vez disso, o conteúdo das sinapses é limpo a zero para criar um "slot vazio".

```python
def clear_synapses(self, indices_to_clear: list[int]):
    for idx in indices_to_clear:
        self.synapse_emb.data[idx].zero_()
        self.w1.data[idx].zero_()
        self.w2.data[idx].zero_()
```

Ao invalidar apenas o conteúdo do slot sem alterar o tamanho do tensor, a integridade dos índices é perfeitamente preservada. Os slots vazios podem ser posteriormente reutilizados sobrescrevendo-os com novas sinapses via Hot-Swap.

## 6. The Cosine Similarity Trap: O problema do buraco negro do vetor zero

### 6.1 Descoberta
Ao implementar a purga por limpeza a zero para a operação Unplug, encontrei um bug crítico onde **a saída colapsou completamente**.

### 6.2 Análise da causa raiz
O roteador SRA realiza o roteamento usando similaridade do cosseno. O vetor de embedding de uma sinapse limpa a zero torna-se $\mathbf{0}$, que permanece $\mathbf{0}$ mesmo após normalização. A similaridade do cosseno entre qualquer vetor de entrada $h$ e o vetor zero é $0.0$.

O problema surge porque a similaridade do cosseno varia em $[-1.0, 1.0]$. Se a similaridade do cosseno de uma sinapse válida é negativa (ex. $-0.5$), **a sinapse vazia ($0.0$) obtém uma pontuação matematicamente mais alta, fazendo com que o roteador selecione preferencialmente a sinapse vazia**.

$$\text{similarity}(h, \mathbf{0}) = 0.0 > -0.5 = \text{similarity}(h, e_{\text{valid}})$$

Os dados são "sugados e desaparecem" no que deveria ser um slot vazio inexistente — um comportamento semelhante a um buraco negro.

### 6.3 Solução: Bloqueio completo via mascaramento -∞
Adicionei processamento de máscara para detectar e excluir sinapses limpas a zero na passagem direta do roteador.

```python
logits = torch.einsum("btd,nd->btn", h_norm, emb_norm) * self.scale

# Detectar sinapses limpas a zero
is_cleared = (full_emb == 0).all(dim=-1)
if is_cleared.any():
    logits = logits.masked_fill(is_cleared.view(1, 1, -1), float('-inf'))
```

A máscara $-\infty$ torna matematicamente impossível a seleção de sinapses vazias, independentemente de quão baixas sejam as pontuações de outras sinapses.

## 7. The Complete Lifecycle of Modular AI

Os mecanismos descritos acima permitem ao SRA realizar o ciclo de vida completo da IA modular:

```
Treinar → Hot-Swap (Compor) → Servir
   ↓                             ↓
Desenvolvimento              Purgar (Excluir)
paralelo independente             ↓
                          Reutilização de slot
                                  ↓
                          Novo Hot-Swap
```

1. **Treinar**: Múltiplas equipes compartilham um modelo base e desenvolvem independentemente suas sinapses especializadas em paralelo.
2. **Compor**: Tensores treinados são fisicamente copiados no modelo base de produção para deploy.
3. **Servir**: A inferência é executada com Zero Forgetting garantido pela pré-filtragem com máscara rígida.
4. **Excluir**: Sinapses desnecessárias são fisicamente removidas ou limpas a zero para purga.
5. **Reutilizar**: Slots vazios são reutilizados inserindo novas sinapses especializadas via Hot-Swap.

## 8. Discussion

### 8.1 Divergência de representação (Representation Divergence)
O **pré-requisito absoluto** para o Hot-Swap é que todas as sinapses especializadas sejam derivadas do mesmo modelo base pré-treinado (Tronco Compartilhado). O transplante de sinapses entre modelos treinados de forma completamente independente causa o colapso do roteamento devido à divergência nas representações vetoriais internas.

### 8.2 Super Roteador como alternativa
Para relaxar a restrição do Tronco Compartilhado, uma abordagem foi validada onde modelos inteiros independentes são encapsulados e orquestrados por um Super Roteador usando Gumbel-Softmax. Esta abordagem atinge roteamento rígido perfeito $1.0$ vs $0.0$, permitindo a comutação dinâmica completa de recursos computacionais mesmo entre modelos com arquiteturas diferentes.

### 8.3 Riscos de segurança
A capacidade Hot-Swap introduz novos vetores de ameaças de segurança devido à sua propriedade de carregar dinamicamente arquivos de pesos de fora de um sistema em execução. Os riscos principais incluem: (1) execução arbitrária de código via exploits Pickle, (2) injeção maliciosa de pesos (Backdoor Injection), (3) sequestro de roteamento via falsificação de chaves do roteador, e (4) ataques DoS via swap thrashing. Mitigações como o formato obrigatório `safetensors`, assinatura criptográfica de sinapses e limitação de taxa são recomendadas.

### 8.4 Limitações atuais e trabalhos futuros
Esta pesquisa está em estágio experimental com modelos de pequena escala ($d_\text{model}=128$, $n_\text{layers}=4$). A validação em LLMs de classe 10B permanece um importante desafio futuro. Além disso, o problema de sincronização do roteador — a potencial necessidade de aprendizado adaptativo das chaves do roteador ao adicionar sinapses com capacidades completamente novas — requer investigação adicional.

## 9. Conclusion

Neste artigo, propus e validei métodos para tornar os LLMs "Hotswappable" (dinamicamente plugáveis) aproveitando a modularidade do SRA (Synaptic Routing Architecture). A operação Plug-In do Hot-Swap completa o deploy apenas através da cópia física de tensores, enquanto a operação Unplug estabelece duas abordagens de remoção: separação física e purga por limpeza a zero. Através de um mecanismo de mascaramento rígido inspirado na Pré-filtragem de bancos de dados vetoriais, o Zero Forgetting é matematicamente garantido. Ao descobrir e resolver o "Problema do Buraco Negro" dos vetores zero na similaridade do cosseno encontrado durante o Unplug, a reutilização segura de slots é alcançada.

Em uma era onde os modelos continuam crescendo e se tornando mais opacos, a abordagem "Hotswappable LLM" — que permite o controle físico de plugar e desplugar componentes de inteligência — representa uma direção extremamente promissora para a manutenibilidade, segurança e eficiência operacional dos modelos.

## References

- Suzuki, J. (2026). [All You Need Is Router: Dynamic Sparse Modularity in Neural Networks. *Technical Report*.](https://github.com/JunSuzukiJapan/SynapticRouter/blob/main/docs/paper_draft_en.md)
- Vaswani, A. et al. (2017). Attention Is All You Need. *NeurIPS*.
- Shazeer, N. et al. (2017). Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer. *ICLR*.
- Jiang, A.Q. et al. (2024). Mixtral of Experts. *arXiv:2401.04088*.
- Jang, E. et al. (2017). Categorical Reparameterization with Gumbel-Softmax. *ICLR*.

## Appendix: Interactive Demos

Os processos completos de Hot-Swap e exclusão de sinapses descritos neste artigo podem ser experimentados interativamente nos seguintes notebooks Google Colab.

- **Demo de composição de sinapses Hot-Swap (Treinamento base → Treinamento independente → Composição → Prova do Zero Forgetting)**<br>
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/10_hotswap_plugins_demo.ipynb)
- **Demo de exclusão de sinapses (Remoção física → Limpeza a zero → Resolução do problema do buraco negro)**<br>
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/11_synapse_deletion_demo.ipynb)

## Appendix: Related Technical Documents

- **[O futuro do SRA: Hot-Swap dinâmico e extensibilidade](./sra_future_hotswap_ja.md)** — Discussão sobre operação de sinapses em modo cassete, personalização e aprendizado distribuído.
- **[Riscos de segurança no Hot-Swap do SRA](./sra_security_risks_hotswap_ja.md)** — Vetores de ameaças incluindo Pickle Exploit, Backdoor Injection, ataques DoS e estratégias de mitigação.
- **[Divergência de representação e roteamento hierárquico](./sra_representation_divergence_ja.md)** — Abordagem do Tronco Compartilhado e soluções Super Roteador.
- **[Comparação de roteamento rígido para o roteador hierárquico SRA](./sra_hierarchical_hard_routing_ja.md)** — Experimentos comparativos de Soft / STE / Gumbel-Softmax.
