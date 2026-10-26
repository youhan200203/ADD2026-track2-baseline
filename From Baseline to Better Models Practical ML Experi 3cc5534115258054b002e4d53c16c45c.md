# From Baseline to Better Models: Practical ML Experimentation

## 개요

안녕하세요, 23기 박유한입니다. 

머신러닝을 공부하다 보면 모델 구조나 최신 방법론을 배우고 구현하는 경험은 많이 하게 되지만, 실제로 모델 성능을 개선하기 위해 어떤 실험을 해야 하는지를 체계적으로 연습할 기회는 상대적으로 적습니다. 성능이 오르지 않았을 때 원인이 데이터인지, 학습 설정인지, 모델 구조인지 판단해야 하고, 새로운 아이디어를 적용할 때도 어떤 비교 실험을 해야 그 효과를 제대로 확인할 수 있는지 고민해야 합니다.

이 프로젝트에서는 baseline에서 출발하여, 모델을 직접 개선하고 여러 실험을 반복하면서 하나의 작은 empirical ML experimentation의 전체 과정을 경험하는 것을 목표로 합니다.

## 참고 교재

Godbole et al. (2023), *Deep Learning Tuning Playbook*, Google Research.

## 커리큘럼

각 주 미팅 때는 제가 다음 미팅 전까지 준비해야 할 내용과 유의해야 하는 점들을 짧게 정리해 공유하겠습니다. 인원이 많을수록 관심 분야에 따라 2~4인이서 팀을 나누기 좋을 것 같습니다!

#### [1주차]

팀 빌딩 및 Competition / Challenge 또는 연구 질문 설정

진행 중인 대회에 참여할 수도 있고, 이미 종료된 대회나 학회 Challenge도 dataset, metric, baseline이 공개되어 있다면 프로젝트 대상으로 활용할 수 있습니다. Challenge가 아니더라도, 관심 있는 논문이나 기존 모델을 baseline으로 선정하고 이를 재현한 뒤 성능을 개선하거나 특정 요소를 확장하는 프로젝트도 가능합니다.

- 10주 안에 수행 가능한 규모인가?
- Colab, RunPod 수준의 환경에서 반복적인 실험이 가능한가?

1. https://www.kaggle.com/competitions?utm_source=chatgpt.com

가장 접근하기 쉬운 범용 ML competition 플랫폼입니다. 특히 Getting Started → Playground → 일반 competition 순으로 난이도가 높아지며, 머신러닝 입문자도 부담 없이 참여할 수 있는 대회가 제공됩니다.

1. 데이콘 https://dacon.io/competitions 

국내 AI competition 플랫폼입니다.

1. 학회 / 연구 커뮤니티 Challenge 

SemEval(NLP), DCASE(Audio) 등 매년 개최되는 shared task를 활용할 수 있습니다.

#### [2주차]

Challenge / Task 이해, Metric / Dataset 분석, Literature Review

- 기존에는 어떤 모델이나 방법이 사용되었는가?

#### [3주차]

Sanity check, Baseline Reproduction

#### [4주차]

Baseline failure diagnosis, Hypothesis 설정, Experiment plan

- 가장 적은 실험으로 Hypothesis를 어떻게 검증할 것인가?

#### [5주차~8주차]

Model improvement

- 각 실험 결과(Expected/Null/Unexpected)에 대해 multiple possible explanations 파악하기

#### [9주차]

Ablation study, Error analysis

- Ablation study는 모델이나 파이프라인에 추가한 요소를 하나씩 제거하거나 변경하면서, 각 요소가 실제 성능에 얼마나 기여하고 있는지 확인하는 실험입니다.

#### [10주차]

Final model, Reproducibility (Github Repository 배포)

- 최종 발표 자료 제작 및 마무리