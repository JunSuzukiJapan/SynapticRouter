# Hotswappable LLM: 시냅스 라우팅 아키텍처를 통한 제로샷 모듈 합성 및 외과적 지식 삭제

**Jun Suzuki**, 독립 연구자

## Abstract
거대 언어 모델(LLM)은 모든 지식을 단일 파라미터 공간에 밀집하여 저장하는 모놀리식 구조를 가지고 있어, 특정 지식의 추가·삭제가 어렵고 운용상의 유연성에 중대한 제약을 부과하고 있다. 본 논문에서는 Synaptic Routing Architecture(SRA)의 모듈성을 활용하여, 모듈의 동적 탈착——즉 **Hot-Swap**——을 실현하는 기법을 제안·검증한다. Hot-Swap이란, 사전학습된 베이스 모델에 대해 독립 학습한 특화 시냅스를 **일체의 재학습 없이 물리적으로 삽입(Plug-In)**하고, 불필요해지면 **외과적으로 제거(Unplug)**할 수 있는, 양방향 모듈 교환 작업이다. 실험 결과, 벡터 데이터베이스의 사전 필터링 기술에서 착안한 하드 마스크 메커니즘을 통해, 삽입 후에도 베이스 모델의 출력이 소수점 이하까지 완전히 일치하는 **Zero Forgetting**을 달성하였다. 또한, 제거 작업에서 코사인 유사도에서의 제로 벡터 "블랙홀 문제"를 발견·해결하여, "학습→삽입→제거→재이용"이라는 모듈러 AI의 완전한 라이프사이클을 실현했다.

## 1. Introduction

### 1.1 모놀리스 모델의 운용상 한계
"Attention Is All You Need" 이후, Transformer 아키텍처는 자연어 처리를 비롯한 많은 영역에서 지배적 지위를 확립하였다. 그러나 수백억~수천억 파라미터를 가진 모놀리식 LLM은 다음과 같은 심각한 운용상의 문제를 안고 있다.

1. **파국적 망각(Catastrophic Forgetting)**: 범용 모델에 특정 도메인(예: 사내 규정, 전문 코드)을 추가 학습시키면 기존의 범용 능력이 파괴·열화된다.
2. **학습 비용의 비대화**: 새로운 지식을 추가할 때마다 모델 전체(또는 LoRA 등의 어댑터)의 재학습·병합이 필요하여, 여러 팀의 완전 병행 개발이 곤란하다.
3. **지식 삭제의 불가능성**: 특정 지식만을 잊게 하는 "Machine Unlearning"은, 파라미터가 복잡하게 얽힌 모놀리스 모델에서는 극히 어려우며, 재학습을 시도하면 무관한 능력까지 파괴된다.

### 1.2 본 논문의 기여
필자가 제안하는 Synaptic Routing Architecture(SRA) [Suzuki, 2026]는 극소 독립 모듈(시냅스)과 경량 라우터로 구성되는 스파스 아키텍처이다. 선행 연구에서는 라우터의 자율적 태스크 분리 능력을 실증했으나, 본 논문에서는 SRA의 모듈성이 가져오는 **운용상의 혁신**——모듈의 삽입과 제거를 자유자재로 행하는 **Hot-Swap**——에 초점을 맞추어, 이하 3가지 기여를 보고한다.

1.**Hot-Swap: Plug-In(삽입)**: 독립 학습한 특화 모델의 가중치 텐서를 베이스 모델의 빈 슬롯에 물리적으로 복사하는 것만으로 배포가 완료되는 기법의 구현과 검증.
2.**Hot-Swap: Unplug(제거)**: 물리적 분리(`pop_synapses`)와 제로 클리어에 의한 퍼지(`clear_synapses`)라는 2가지 삭제 API의 설계와, 코사인 유사도에서의 "제로 벡터 블랙홀 문제"의 발견·해결.
3.**Zero Forgetting의 실험적 증명**: 벡터 DB의 사전 필터링(Pre-filtering)에서 착안한 하드 마스크 메커니즘에 의해, 삽입·제거 전후에서 베이스 모델의 출력 Loss가 소수점 이하까지 완전히 일치함을 실증.

## 2. Background: SRA Architecture

SRA(Synaptic Routing Architecture)는 생물의 뇌에서 영감을 받은 동적이고 스파스한 아키텍처이다. 여기서는 Hot-Swap의 이해에 필요한 구성 요소를 개설한다(상세는 [Suzuki, 2026] 참조).

### 2.1 Router
SRA의 심장부인 라우터는 Attention 메커니즘이 없는 **단일 선형 레이어(Linear Layer)**이다. 입력의 은닉 상태 $h$와 각 시냅스의 특징 벡터(임베딩) $e_i$의 코사인 유사도를 계산하여, Top-k개의 시냅스를 선택한다.

$$\text{logits}_i = \frac{h \cdot e_i}{\|h\| \cdot \|e_i\|} \cdot \alpha$$

여기서 $\alpha$는 스케일링 계수이다.

### 2.2 Tiny Synapses
각 시냅스는 소형 Multi-Head Attention과 MLP로 구성된 독립적인 극소 모듈이다. 라우터에 의해 선택된 시냅스만이 계산을 수행하므로 높은 계산 효율을 실현한다.

### 2.3 공유 트렁크(Shared Trunk)
Hot-Swap 성립의 절대 조건으로 극히 중요한 것이 **공유 트렁크 방식**이다. 모든 특화 시냅스는 동일한 사전학습된 베이스 모델(Embedding 레이어, Attention 레이어)에서 파생되어, 시냅스 부분만을 독립 학습한다. 이를 통해 모델 간 내부 벡터 표현의 괴리(Representation Divergence)를 방지하고, 물리적 복사에 의한 시냅스 이식을 가능하게 한다.

## 3. Hot-Swap: Plug-In(모듈 삽입)

Hot-Swap의 첫 번째 작업은 독립 학습한 특화 모듈을 베이스 모델에 **삽입(Plug-In)**하는 것이다. 본 장에서는 이 삽입 작업이 PyTorch의 텐서 작업만으로 완결되는 극히 심플한 프로세스임을 보여준다.

### 3.1 수법

```python
# hotswap_model: 프로덕션 환경의 베이스 모델(빈 슬롯 추가 완료)
# plugin_math: 수학 팀이 독립 학습한 특화 LLM

with torch.no_grad():
    for l in range(layers):
        target_block = hotswap_model.blocks[l]
        src_block = plugin_math.blocks[l]
        
        # 라우터의 임베딩 벡터를 복사
        target_block.router.synapse_emb.data[4:8] = src_block.router.synapse_emb.data
        
        # Expert (TinySynapse)의 가중치 (w1, w2)를 복사
        target_block.w1.data[4:8] = src_block.w1.data
        target_block.w2.data[4:8] = src_block.w2.data
```

베이스 모델의 텐서의 특정 인덱스(빈 슬롯)에 대해 학습 완료된 특화 모델의 텐서를 직접 대입하는 것뿐이다. SRA는 공유 트렁크 방식에 의해 베이스 모델의 공통 지식(Attention 레이어 등)을 완전히 동결한 상태에서 시냅스만을 학습하므로, 이 물리적 복사가 성립한다.

### 3.2 독립 병행 개발의 실현
이 방식에 의해 "코드 팀"과 "수학 팀"이 동일한 베이스 모델을 기반으로, 서로 일절 간섭하지 않고 각자의 특화 시냅스만을 독립 학습할 수 있다. 학습 완료 후에는 가중치 텐서를 프로덕션 환경의 베이스 모델의 빈 슬롯에 메모리 복사하는 것만으로 배포가 완료된다.

## 4. Zero Forgetting: 벡터 DB Pre-filtering에서 착안한 하드 마스크 메커니즘

### 4.1 과제: 라우터의 혼동
단순히 텐서를 물리적으로 복사하는 것만으로는, 라우터가 신구 시냅스를 혼동하여 베이스 모델의 출력이 변화할 위험이 있다.

### 4.2 벡터 DB에서의 메타데이터 필터링
Pinecone이나 Weaviate 등의 현대적 벡터 데이터베이스에서는 코사인 유사도에 의한 의미 검색 외에, 메타데이터에 의한 필터링이 중요한 역할을 한다.

-**Post-filtering(사후 필터링)**: Top-K 검색 후에 조건 외의 결과를 제외하는 방식. K건이 부족해지는 "K-NN 고갈 문제"가 발생하기 쉽다.
-**Pre-filtering(사전 필터링)**: 검색 **전**에 메타데이터 마스크로 검색 공간을 제한하고, 조건에 합치하는 후보만으로 Top-K 검색을 수행하는 방식. 노이즈가 완전히 제거된다.

### 4.3 SRA의 사전 하드 마스크
SRA의 라우터의 본질은, 입력 벡터와 각 시냅스의 임베딩 벡터의 내적을 계산하는 **인메모리 벡터 검색 엔진(Maximum Inner Product Search: MIPS)**이다.

필자는 벡터 DB의 Pre-filtering을 라우터의 순전파에 직접 내장하였다. 추론 시에 "현재 태스크에 허가된 시냅스의 집합"을 나타내는 메타데이터 마스크를 모델에 전달한다.

```python
# Router 레이어의 순전파
logits = torch.einsum("btd,nd->btn", h_norm, emb_norm) * self.scale

# Pre-filtering: 허가되지 않은 시냅스의 로짓을 -∞로 설정
logits = logits.masked_fill(~allowed_mask, float('-inf'))

# Top-K 라우팅
vals, idx = torch.topk(logits, k, dim=-1)
```

이 `masked_fill`에 의한 Pre-filtering으로, 라우터는 허가된 시냅스 중에서만 전문가를 선택한다. 다른 모델의 가중치가 아무리 공존해 있어도, 계산 그래프상에서는 완전히 무시되므로, **베이스 모델의 Loss가 합체 전과 소수점 이하까지 완전히 일치한다(수학적으로 제로 간섭)**.

### 4.4 실험 결과
베이스 모델(Code/Math/Text의 3도메인 언어 모델)에 대해 독립 학습한 특화 시냅스를 Hot-Swap한 전후에서 베이스 모델의 Validation Loss를 비교하였다. 결과, 합체 전과 합체 후에서 Loss가 소수점 이하까지 완전히 일치하여, Zero Forgetting이 실증되었다.

## 5. Hot-Swap: Unplug(모듈 제거)

Hot-Swap의 두 번째 작업은 불필요해진 모듈을 베이스 모델에서 **제거(Unplug)**하는 것이다. 지식을 "꽂을 수" 있다면, 불필요해진 지식을 "뺄 수" 있는 것도 요구된다. 모놀리스 모델에서의 Machine Unlearning은 파라미터가 복잡하게 얽혀 있어 극히 곤란하지만, SRA의 모듈 구조는 이 문제를 물리적 작업으로 해결한다.

### 5.2 접근법①: 물리적 분리(pop_synapses)
나중에 Hot-Swap으로 추가한 시냅스가 불필요해진 경우, 텐서의 말미에서 슬라이스하여 물리적으로 잘라낸다.

```python
def pop_synapses(self, num_drop: int):
    self.synapse_emb = nn.Parameter(self.synapse_emb.data[:-num_drop])
```

**장점**: VRAM 사용량이 물리적으로 축소되어, 모델을 추가 전 상태로 완전히 복원할 수 있다. OS의 드라이버를 언인스톨하듯이, AI의 뇌의 부품을 물리적으로 분리할 수 있다.

### 5.3 접근법②: 제로 클리어에 의한 퍼지(clear_synapses)
말미가 아닌 중간 인덱스의 시냅스를 삭제하고 싶은 경우, 물리적 삭제는 인덱스의 어긋남을 일으켜 메타데이터 마스크에 의한 제어 시스템이 붕괴한다. 따라서, 시냅스의 내용을 제로 클리어하여 "빈 슬롯화"한다.

```python
def clear_synapses(self, indices_to_clear: list[int]):
    for idx in indices_to_clear:
        self.synapse_emb.data[idx].zero_()
        self.w1.data[idx].zero_()
        self.w2.data[idx].zero_()
```

텐서의 크기를 바꾸지 않고 슬롯의 내용만을 무효화하므로, 인덱스의 정합성은 완벽하게 유지된다. 빈 슬롯에는 나중에 새로운 시냅스를 덮어쓰기(Hot-Swap)하여 재이용할 수 있다.

## 6. The Cosine Similarity Trap: 제로 벡터의 블랙홀 문제

### 6.1 문제의 발견
제로 클리어에 의한 퍼지를 구현했을 때, **출력이 완전히 붕괴하는** 심각한 버그에 직면하였다.

### 6.2 원인 분석
SRA의 라우터는 코사인 유사도를 사용하여 라우팅을 수행한다. 제로 클리어된 시냅스의 임베딩 벡터는 $\mathbf{0}$이 되며, 정규화해도 $\mathbf{0}$인 채로이다. 임의의 입력 벡터 $h$와 제로 벡터의 코사인 유사도는 $0.0$이 된다.

여기서 문제가 발생한다. 코사인 유사도의 치역은 $[-1.0, 1.0]$이다. 정상적인 시냅스의 코사인 유사도가 음의 값(예: $-0.5$)인 경우, **빈 시냅스($0.0$)의 점수가 수학적으로 더 높아져, 라우터가 빈 시냅스를 우선적으로 선택해 버린다**.

$$\text{similarity}(h, \mathbf{0}) = 0.0 > -0.5 = \text{similarity}(h, e_{\text{valid}})$$

"존재를 지운 빈 슬롯에 데이터가 빨려 들어가 소멸하는", 블랙홀과 같은 거동이다.

### 6.3 해결책: -∞ 마스크에 의한 완전 차단
라우터의 순전파에 제로 클리어된 시냅스를 검지·제외하는 마스크 처리를 추가하였다.

```python
logits = torch.einsum("btd,nd->btn", h_norm, emb_norm) * self.scale

# 제로 클리어된 시냅스를 검지
is_cleared = (full_emb == 0).all(dim=-1)
if is_cleared.any():
    logits = logits.masked_fill(is_cleared.view(1, 1, -1), float('-inf'))
```

$-\infty$ 마스크에 의해 다른 시냅스의 점수가 아무리 낮아도, 빈 시냅스가 선택되는 것은 수학적으로 불가능해진다.

## 7. The Complete Lifecycle of Modular AI

이상의 메커니즘에 의해, SRA는 다음의 모듈러 AI의 완전한 라이프사이클을 실현한다.

```
학습(Train) → 합체(Hot-Swap) → 운용(Serve)
       ↓                                    ↓
  독립 병행 개발                         삭제(Purge)
                                            ↓
                                    슬롯 재이용(Reuse)
                                            ↓
                                    신규 합체(Hot-Swap)
```

1. **학습**: 복수 팀이 베이스 모델을 공유하고, 각자의 특화 시냅스를 독립 병행 개발한다.
2. **합체**: 학습 완료 텐서를 프로덕션 환경의 베이스 모델에 물리적으로 복사하여 배포한다.
3. **운용**: 하드 마스크에 의한 Pre-filtering으로 Zero Forgetting을 보증하면서 추론한다.
4. **삭제**: 불필요해진 시냅스를 물리적으로 분리하거나 제로 클리어하여 퍼지한다.
5. **재이용**: 빈 슬롯에 새로운 특화 시냅스를 Hot-Swap하여 재이용한다.

## 8. Discussion

### 8.1 표현의 분기(Representation Divergence)
Hot-Swap가 성립하는 **절대 조건**은, 모든 특화 시냅스가 동일한 사전학습된 베이스 모델(공유 트렁크)에서 파생되어 있는 것이다. 완전히 독립적으로 학습시킨 모델 간에서 시냅스를 이식하면, 내부 벡터 표현의 괴리에 의해 라우팅이 붕괴한다.

### 8.2 상위 라우터(Super Router)에 의한 대안
공유 트렁크의 제약을 완화하기 위해, 독립적인 모델 전체를 캡슐화하고, Gumbel-Softmax에 의한 상위 라우터(Super Router)로 묶는 접근법도 검증되었다. 이 방식에서는 완벽한 $1.0$ vs $0.0$의 Hard Routing이 달성되어, 아키텍처가 다른 모델끼리도 계산 리소스의 완전한 동적 전환이 가능해진다.

### 8.3 보안 리스크
Hot-Swap 기능은 가동 중인 시스템 외부로부터 동적으로 가중치 파일을 로드하는 특성상, 새로운 보안 위협을 낳는다. 주요 리스크로서 (1) Pickle Exploit에 의한 임의 코드 실행, (2) 악의적 가중치 주입(Backdoor Injection), (3) 라우터 키 위조에 의한 라우팅 하이재킹, (4) 스왑 스래싱에 의한 DoS 공격이 열거된다. `safetensors` 포맷의 의무화, 암호 서명에 의한 시냅스 검증, 레이트 리미팅의 도입 등에 의한 방어가 권장된다.

### 8.4 현재의 제약과 향후 전망
본 연구는 `d_model=128`, `n_layers=4` 정도의 소규모 모델에서의 실험 단계이며, 10B 클래스의 거대 LLM에서의 실증은 향후의 중요한 과제이다. 또한, 라우터와의 동기 문제——전혀 새로운 능력을 가진 시냅스를 추가할 때 라우터 키의 적응 학습이 필요해질 가능성——에 대해서도 추가적인 조사가 필요하다.

## 9. Conclusion

본 논문에서는 SRA(Synaptic Routing Architecture)의 모듈성을 활용하여, LLM을 "Hotswappable(동적 탈착 가능)"하게 만드는 기법을 제안·검증하였다. Hot-Swap의 삽입(Plug-In) 작업에서는 텐서의 물리적 복사만으로 배포를 완료시키고, 제거(Unplug) 작업에서는 물리적 분리와 제로 클리어에 의한 퍼지의 2가지 수법을 확립하였다. 벡터 데이터베이스의 Pre-filtering 기술에서 착안한 하드 마스크 메커니즘에 의해 Zero Forgetting을 수학적으로 보증하고, 제거 시에 발견된 코사인 유사도에서의 제로 벡터 "블랙홀 문제"를 해결함으로써, 슬롯의 안전한 재이용을 실현했다.

모델이 거대화·블랙박스화를 계속하는 현대에 있어, 지능의 부품을 물리적으로 탈착하여 컨트롤할 수 있는 "Hotswappable LLM"의 접근법은, 모델의 유지보수성·안전성·운용 효율의 관점에서 극히 유망한 방향성이다.

## References

- Suzuki, J. (2026).[All You Need Is Router: Dynamic Sparse Modularity in Neural Networks. *Technical Report*.](https://github.com/JunSuzukiJapan/SynapticRouter/blob/main/docs/paper_draft.md)
- Vaswani, A. et al. (2017). Attention Is All You Need. *NeurIPS*.
- Shazeer, N. et al. (2017). Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer. *ICLR*.
- Jiang, A.Q. et al. (2024). Mixtral of Experts. *arXiv:2401.04088*.
- Jang, E. et al. (2017). Categorical Reparameterization with Gumbel-Softmax. *ICLR*.

## Appendix: Interactive Demos

본 논문에서 해설한 Hot-Swap 및 Synapse Deletion의 전 프로세스는 이하의 Google Colab 노트북에서 실제로 실행하며 체험할 수 있습니다.

- **Hot-Swap 시냅스 합체 데모(베이스 학습~독립 학습~합체~Zero Forgetting의 증명)**<br>
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/10_hotswap_plugins_demo.ipynb)
- **시냅스 삭제 데모(물리적 분리~제로 클리어~블랙홀 문제의 해결)**<br>
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/11_synapse_deletion_demo.ipynb)

## Appendix: Related Technical Documents

- **[SRA의 미래: 시냅스의 동적 핫스왑과 확장성](./sra_future_hotswap_ja.md)** — 카세트 방식·시냅스 운용, 퍼스널라이즈, 분산형 학습에 관한 고찰.
- **[SRA 핫스왑 기능에서의 보안 리스크와 공격 기법](./sra_security_risks_hotswap_ja.md)** — Pickle Exploit, Backdoor Injection, DoS 공격 등의 위협 벡터와 방어책.
- **[표현의 분기와 계층형 라우팅](./sra_representation_divergence_ja.md)** — Shared Trunk 방식과 Super Router에 의한 해결책.
- **[SRA 계층형 라우터에서의 Hard Routing 기법 비교 검증](./sra_hierarchical_hard_routing_ja.md)** — Soft / STE / Gumbel-Softmax의 비교 실험.
