# LLM Substituível a Quente (Hotswappable LLM): Composição de Módulos Zero-Shot e Exclusão Cirúrgica de Conhecimento via Arquitetura de Roteamento Sinaptico

**Jun Suzuki**, Independent Researcher

## Resumo (Abstract)
Os humanos podem aprender novas habilidades (por exemplo, andar de bicicleta ou um novo idioma) e fazê-las funcionar como circuitos independentes no cérebro sem destruir o conhecimento existente. No entanto, os Grandes Modelos de Linguagem (LLMs) atuais possuem uma estrutura monolítica, tornando a adição de conhecimento independente ou a exclusão de memórias específicas extremamente difícil.

Neste artigo, aproveitamos o mecanismo de localização funcional da Arquitetura de Roteamento Sinaptico (SRA) para propor o **Hot-Swap**. Hot-Swap é uma operação de troca de módulos onde sinapses especializadas podem ser **transplantadas cirurgicamente (Plug-In) no modelo de produção sem nenhum retreinamento**, e quando não são mais necessárias, **memórias específicas podem ser desconectadas com segurança (Unplug)**. Usando um mecanismo de máscara rígida inspirado na filtragem prévia (Pre-filtering) de bancos de dados de vetores, alcançamos o **Esquecimento Zero (Zero Forgetting)** e resolvemos o "problema do buraco negro" dos vetores nulos na similaridade por cosseno.

## 1. Introdução (Introduction)
A SRA traz a inovação operacional do Hot-Swap: Plug-In (Transplante) e Unplug (Exclusão), com uma garantia matemática de Zero Esquecimento para resolver os problemas dos modelos monolíticos.

## 2. A Arquitetura SRA
A SRA imita o isolamento espacial do cérebro. Uma condição absoluta para o Hot-Swap funcionar é a abordagem de **Tronco Compartilhado (Shared Trunk)**, onde todas as sinapses especializadas compartilham o mesmo modelo base pré-treinado para evitar a Divergência de Representação.

## 3. Hot-Swap: Plug-In (Transplante de Módulos)
Nós transplantamos o módulo especializado fisicamente apenas copiando os tensores PyTorch para os espaços vazios do modelo base congelado.

## 4. Esquecimento Zero: O Mecanismo de Máscara Rígida (Hard-Mask)
Para evitar que o roteador confunda sinapses antigas e novas, usamos o **Pré-filtragem (Pre-filtering)**, bloqueando fisicamente caminhos desnecessários com `-inf` durante a inferência.

## 5. Hot-Swap: Unplug (Remoção / Inativação de Conhecimento)
Módulos desnecessários podem ser removidos usando Desconexão Física (`pop_synapses`) cortando o final do tensor, ou Expurgo por Inativação (`clear_synapses`), zerando o conteúdo sem alterar o tamanho do tensor para manter a consistência do índice.

## 6. O Problema do Buraco Negro do Vetor Nulo
Zerar uma sinapse cria um vetor nulo com similaridade $0.0$. Se as pontuações normais forem negativas, o roteador selecionará a sinapse vazia (o buraco negro). Eu resolvi isso adicionando uma máscara $-\infty$ para detectar e bloquear essas rotas.

## 7, 8, 9. Conclusão
O ciclo de vida completo (Aprender → Transplantar → Operar → Remover → Reutilizar) da SRA evolui modelos de IA de caixas pretas monolíticas para sistemas modulares vitais para controle seguro e atualizações infinitas.

## Referências (References)
- Suzuki, J. (2026). [Neuro-inspired Synaptic Routing: Overcoming Catastrophic Forgetting via Dynamic Modular Pathways. *Technical Report*.](https://github.com/JunSuzukiJapan/SynapticRouter/blob/main/docs/paper_draft_en.md)
