# Synaptic Routing Architecture (SRA)

Synaptic Routing Architecture (SRA)는 생물학적 뇌(시냅스)에서 영감을 받은 새롭고 동적이며 희소하고 모듈화된 신경망 아키텍처입니다.
거대하고 정적인 Transformer 대신, SRA는 입력을 적절한 "시냅스"(작은 모듈)로 동적으로 라우팅하여 더 효율적인 학습과 구조적 지능을 달성합니다.

## 🎯 동기 (Motivation)

최근 몇 년 동안 AI 모델이 점점 더 거대해짐에 따라 단일 아키텍처 네트워크는 "계산 리소스의 기하급수적 증가"와 "다중 작업 학습 중 치명적 망각(Catastrophic Forgetting)"과 같은 중요한 과제에 직면해 있습니다.
SRA는 **"입력에 따라 필요한 최소 모듈(시냅스)만 동적으로 호출하고 결합하는" 희소(Sparse) 접근 방식**을 사용하여 이러한 문제를 해결하려고 시도합니다. 이를 통해 상호 간섭 없이 동일한 네트워크 내에서 서로 다른 특성을 가진 여러 작업을 학습할 수 있으며, 확장성과 높은 학습 효율성을 모두 달성하는 것을 목표로 합니다.

## 💡 기본 아이디어

일반적인 AI 모델(Transformer 등)은 거대한 하나의 '두뇌'로 모든 처리를 수행하려고 합니다. 하지만 이 방식에서는 모델을 더 똑똑하고 크게 만들 때마다 계산 부담이 너무 무거워집니다. 그래서 SRA는 **'작은 전문가 두뇌(SRA에서는 이를 '시냅스'라고 부릅니다)'를 많이 준비하고, 그때그때의 문제에 맞춰 필요한 전문가만 불러와서 사용하는** 방식을 채택했습니다.

여기서 핵심이 되는 것은 '어떤 전문가를 불러올지'를 결정하는 시스템입니다. SRA에는 '라우터(안내자)'가 존재하여, 입력된 데이터를 보고 가장 잘할 것 같은 전문가를 순식간에 골라냅니다. 이 라우터는 각 전문가가 똑똑해지는(학습하는) 것과 동시에 '누구를 선택하는 것이 정답인가'를 스스로 학습하여, 자동으로 최적의 배분을 할 수 있도록 성장해 나갑니다.

## 🧠 아키텍처 개요 (Architecture)

SRA는 주로 다음 구성 요소로 이루어져 있습니다.

1. **Synapse (시냅스 모듈)**
   - 독립적인 작은 계산 단위 (예: 미니 Transformer 또는 MLP).
   - 학습을 통해 특정 기능이나 패턴 처리에 특화됩니다.
2. **Router (라우터)**
   - 입력 토큰에 따라 사용 가능한 모든 시냅스 중 최적의 `Top-k` 시냅스만 동적으로 선택하여 희소 계산을 구현합니다.
3. **Synapse Space (시냅스 공간)**
   - 각 시냅스는 임베딩 공간에 배치되며, 시냅스 간의 거리가 "기능적 유사성"을 나타내도록 자체 조직화됩니다.
4. **Local Learning Rule (지역 학습 규칙)**
   - 표준 역전파 외에도 Hebbian, STDP 및 보상에 기반한 3요소 규칙(예: `trace × routing × reward`)을 사용하여 지역 학습 규칙을 결합하여 로드 밸런싱 및 특성화를 촉진합니다.

## 📁 디렉토리 구조 (Directory Structure)

이 저장소의 주요 구조는 다음과 같습니다.

- `src/` : SRA 모델의 핵심 구현 및 학습/평가 스크립트.
  - `sra_gpu_models.py` / `sra_language_models.py` : MoE(Mixture of Experts) 기술을 적용한 SRA 모델 구현.
  - `train_mtl_algo.py` / `train_mtl_lang.py` : 다중 작업 학습(알고리즘 추론 및 언어 모델링) 실행 스크립트.
- `docs/` : 아키텍처 결정 기록(ADR) 및 다양한 실험과 라우팅 분석에 대한 보고서.
- `data/` : 학습 및 검증에 사용되는 토이 데이터 세트 (Code, Math, Text 등).
- `tests/` : 각 구성 요소의 테스트 코드.

## 🚀 사용법 (Usage)

### 요구 사항
```bash
pip install torch
```

### 기본 실행
`copy`, `reverse`, `paren`, `addmod`와 같은 구조적 알고리즘 작업을 사용하여 모델의 학습 및 추론을 테스트할 수 있습니다.

단일 작업을 실행하려면:
```bash
python src/sra_experiment.py --task reverse --steps 2000
```

모든 작업을 순차적으로 실행하려면 (작업 스위트):
```bash
python src/sra_experiment.py --task-suite
```

### 다른 아키텍처와의 비교
SRA를 표준 `Transformer` 및 `MLP` 아키텍처와 비교하는 스크립트가 제공됩니다.

```bash
# copy 작업을 사용하여 아키텍처 비교
python src/compare_architectures.py --task copy --steps 500
```

## 📊 비교 (Comparison)

> **📝 업데이트 예정:**
> 현재 표준 Transformer 모델과의 정량적 비교 검증을 진행 중입니다.
> 초기 검증(적은 단계의 copy 작업)에서 SRA가 Transformer보다 유효성 손실(Validation Loss)을 더 빨리 줄이고 더 높은 정확도에 더 빨리 도달하는(높은 학습/샘플 효율성) 긍정적인 징후가 나타났습니다.
>
> 단계별 처리량 및 VRAM 효율성 향상을 포함한 자세한 벤치마크 결과가 향후 여기에 추가될 예정입니다.

## 🧪 실험 및 분석 (Experiments & Analysis)

- [알고리즘 추론에서의 다중 작업 학습 및 라우팅 분석](./routing_analysis_algorithmic.md)
  - SRA가 간섭 없이 여러 알고리즘 작업을 동시에 학습하고 작업 특성에 따라 전문가(시냅스)를 자율적으로 분리 및 모듈화할 수 있음을 검증한 보고서.
- [교차 도메인 언어 모델링에서의 라우팅 분석 (Code / Math / Text)](./routing_analysis_language.md)
  - SRA가 서로 다른 문법 및 어휘를 가진 도메인(코드, 수학 공식, 자연어)을 동시에 학습하고 시냅스가 기능적으로 분화(특성화)하여 각 도메인에 대한 추론을 수행하는 메커니즘을 검증한 보고서.
- [다국어 기계 번역에서의 라우팅 분석(영/불/일) 및 제로샷 일반화](../dev/multilingual_translation_routing_analysis.md)
  - 구문 구조(SVO 및 SOV)에 따라 모델이 자율적으로 번역 모듈을 전환하는 현상과 학습하지 않은 언어 쌍을 번역할 때 무의식적으로 영어를 '피벗 언어'로 활용하는 놀라운 일반화 능력에 대해 설명합니다.
- [Decision Transformer(강화 학습)에서의 인식과 정책의 완전한 분리](../dev/decision_transformer_routing_analysis.md)
  - SRA에게 게임을 플레이하게 한 결과, 환경을 보기 위한 '인식(시각)' 모듈은 모든 작업에서 공유하면서 어떻게 움직일지 결정하는 '정책(두뇌)' 모듈은 작업(보물 찾기 또는 도망)에 따라 완전히 다르게 사용하는 등 생명체와 같은 모듈 구조를 자율적으로 획득했음을 보여주는 흥미로운 보고서입니다.
- [SRA Encoder-Decoder를 통한 실용 수준의 다국어 번역 검증](../dev/sra_seq2seq_translation_analysis.md)
  - SRA를 Encoder-Decoder 형태로 확장하고 실제 코퍼스(opus100)를 사용한 30,000 스텝의 학습을 통해 "Merci beaucoup.", "Good morning."과 같은 실용적인 표현을 BLEU=1.0으로 번역할 수 있음을 입증한 보고서입니다. Cross-Attention의 도입으로 Decoder-only(BLEU=0)에서 전체 평균 BLEU=0.27로 비약적으로 향상되었으며, FR→EN 방향에서는 실용성에 근접한 BLEU=0.56의 정확도를 달성했습니다.


## 🤝 기여 및 라이선스 (Contributing & License)

이 프로젝트는 현재 초기 단계의 실험적 아키텍처입니다. 버그 보고서, 기능에 대한 논의, 성능 향상을 위한 Pull Request 등 Issue와 PR을 통한 참여를 환영합니다!

- **라이선스**: 이 저장소는 [MIT 라이선스](../../LICENSE)에 따라 릴리스됩니다. 자세한 내용은 [`LICENSE`](../../LICENSE) 파일을 참조하십시오.
