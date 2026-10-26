# ADD 2026 Track 2 역할별 업무 체크리스트

## 프로젝트 공통 목표

제한된 GPU 환경에서 재현 가능한 실험을 수행하고, 공식 Qwen2.5-Omni baseline과 다음 접근을 비교한다.

1. 경량 audio deepfake detector
2. Qwen3 기반 LoRA explanation model
3. 문헌과 규정에 근거한 augmentation 및 compute-efficient training

최종 비교 대상은 다음과 같다.

```text
Official baseline
Audio -> Qwen2.5-Omni -> Evidence + Decision

Detector prototype
Audio -> Detector -> Real/Fake

Hybrid system
Audio -> Detector -> Real/Fake
      -> Qwen3 LoRA -> Evidence
      -> Submission XML/JSONL
```

---

## 0. 전원 공통 준비

### 데이터와 평가 프로토콜

- [ ] `train`, `dev`, `labels` 경로와 샘플 수를 확인한다.
- [ ] 공식 Dev 2,000개를 사용하는 `official-dev` 평가를 정의한다.
- [ ] Train/Dev 중복 샘플을 표시한 `clean-dev` 평가를 별도로 정의한다.
- [ ] 중복 샘플을 임의로 삭제하지 않고 manifest 수준에서 제외 여부를 관리한다.
- [ ] Real/Fake 클래스별 성능을 별도로 기록한다.
- [ ] 모든 실험에서 동일한 evaluation script와 BERTScore 설정을 사용한다.
- [ ] seed, 데이터 버전, 코드 revision, checkpoint revision을 기록한다.
- [ ] 공식 Dev를 반복적으로 과적합하지 않도록 Train 내부 validation 사용 여부를 결정한다.

### 규정

- [ ] 외부 speech dataset을 사용하지 않는다.
- [ ] 다른 ADD 2026 Track 데이터를 사용하지 않는다.
- [ ] 외부 ASR, VAD, speaker recognition, deepfake detector checkpoint를 사용하지 않는다.
- [ ] 공개 pretrained language model의 출처와 revision을 기록한다.
- [ ] 외부 non-speech noise 또는 RIR을 사용할 경우 출처와 subset을 기록한다.
- [ ] evaluation sample을 이용해 model weight, normalization statistic, threshold를 업데이트하지 않는다.
- [ ] 규정 해석이 애매한 모델이나 데이터는 사용 전에 주최 측에 문의한다.

### 공통 실험 기록

- [ ] 모든 실험에 `E###` 형식의 experiment ID를 부여한다.
- [ ] 한 실험에서는 핵심 변수 하나만 변경한다.
- [ ] 실행 command와 config를 저장한다.
- [ ] peak VRAM, wall-clock time, GPU 종류, GPU 수를 기록한다.
- [ ] 실패한 실험도 원인과 로그 위치를 기록한다.
- [ ] 결과를 `Expected`, `Null`, `Unexpected` 중 하나로 분류한다.
- [ ] 다음 실험 전에 가능한 설명을 최소 2개 작성한다.

### 공통 결과 지표

- [ ] Real recall
- [ ] Fake recall
- [ ] Balanced accuracy
- [ ] Overall accuracy
- [ ] Real joint score
- [ ] Fake joint score
- [ ] Final score
- [ ] XML validity rate
- [ ] 평균 Evidence 길이
- [ ] 추론 속도 및 peak VRAM

---

# 역할 1. Detector 프로토타입 개발

## 목표

설명 생성과 독립적으로 Real/Fake 판정 성능을 확보하고, Qwen의 생성 결과와 결합 가능한 샘플별 prediction 및 confidence를 제공한다.

## A. 데이터 Sanity Check

- [ ] Train 8,600개와 Dev 2,000개의 라벨 분포를 재확인한다.
- [ ] WAV sample rate, channel, bit depth를 확인한다.
- [ ] Train/Dev 중복 WAV 목록을 detector 평가 manifest에 표시한다.
- [ ] duration 분포를 Real/Fake별로 비교한다.
- [ ] 10초 이상 샘플이 모두 Fake인 현상을 보고한다.
- [ ] RMS, silence ratio, clipping ratio 분포를 계산한다.
- [ ] 단순 feature가 label을 예측하는 shortcut인지 검사한다.

### 완료 조건

- [ ] 데이터 무결성 요약표가 있다.
- [ ] Official Dev와 Clean Dev manifest가 고정되어 있다.
- [ ] 모델 없이 얻을 수 있는 shortcut 성능을 알고 있다.

### 예상 시간

- 인력: 4-8시간
- GPU: 필요 없음

## B. Sanity Baseline

- [ ] 항상 Real을 출력하는 baseline을 평가한다.
- [ ] 항상 Fake를 출력하는 baseline을 평가한다.
- [ ] duration threshold baseline을 평가한다.
- [ ] RMS, silence ratio, duration을 사용한 logistic regression을 평가한다.
- [ ] 결과를 전체 Dev와 길이 구간별로 기록한다.

### 완료 조건

- [ ] Neural detector가 반드시 넘어야 할 최소 성능이 정의되어 있다.
- [ ] duration shortcut이 Dev 성능에 미치는 영향을 수치화했다.

### 예상 시간

- 인력: 3-5시간
- GPU: 필요 없음

## C. Audio Preprocessing Pipeline

- [ ] 16 kHz mono 입력을 보장한다.
- [ ] variable-length batching과 padding 방식을 정의한다.
- [ ] waveform normalization 여부를 결정한다.
- [ ] Log-Mel spectrogram 설정을 고정한다.
- [ ] LFCC 사용 여부와 설정을 고정한다.
- [ ] Train에서 계산한 normalization statistic만 Dev에 적용한다.
- [ ] padding 길이가 label shortcut이 되지 않는지 확인한다.

### 기록할 설정

- [ ] Window length
- [ ] Hop length
- [ ] FFT size
- [ ] Mel/LFCC bin 수
- [ ] 최대 audio length
- [ ] Crop/padding 방식

### 예상 시간

- 인력: 6-10시간
- GPU: 필요 없음

## D. Detector 모델 실험

### D1. 작은 Log-Mel CNN

- [ ] 최소 CNN architecture를 구현한다.
- [ ] parameter 수를 기록한다.
- [ ] overfit-one-batch test를 통과한다.
- [ ] Train loss가 정상적으로 감소하는지 확인한다.
- [ ] Official Dev와 Clean Dev를 평가한다.

### D2. LFCC/LCNN

- [ ] 문헌의 LCNN 구조를 random initialization으로 구현하거나 검증된 코드를 규정에 맞게 사용한다.
- [ ] Log-Mel CNN과 동일한 split 및 metric으로 비교한다.
- [ ] parameter 수, VRAM, 학습 시간을 비교한다.

### D3. AASIST 또는 Raw-waveform 모델

- [ ] 외부 pretrained weight 없이 학습되는지 확인한다.
- [ ] 100-step smoke test로 VRAM과 step time을 측정한다.
- [ ] 경량 CNN 대비 성능 향상이 계산 비용을 정당화하는지 평가한다.

### 권장 우선순위

1. Log-Mel CNN
2. LFCC/LCNN
3. AASIST 또는 raw-waveform 모델

### 완료 조건

- [ ] 최소 2개 neural detector 결과가 있다.
- [ ] 가장 좋은 detector checkpoint가 저장되어 있다.
- [ ] 모든 Dev 샘플에 대한 Real/Fake probability가 있다.

### 예상 시간

- 인력: 20-35시간
- GPU: 15-45 GPU-hour
- 권장 GPU: 1 x 24 GB 이상

## E. Detector Augmentation

- [ ] Gain augmentation을 단독 평가한다.
- [ ] Resampling augmentation을 단독 평가한다.
- [ ] MP3/AAC compression을 단독 평가한다.
- [ ] Pitch shift를 단독 평가한다.
- [ ] Non-speech noise augmentation을 단독 평가한다.
- [ ] RIR/reverberation augmentation을 단독 평가한다.
- [ ] 각 증강의 적용 확률과 파라미터 범위를 기록한다.
- [ ] 가장 좋은 단일 증강 조합만 결합 실험한다.

### 완료 조건

- [ ] 증강별 ablation 표가 있다.
- [ ] 규정과 출처가 확인된 증강만 최종 파이프라인에 포함된다.

### 예상 시간

- 인력: 10-20시간
- GPU: 15-40 GPU-hour

## F. Detector Error Analysis

- [ ] High-confidence False Real 10개를 검토한다.
- [ ] High-confidence False Fake 10개를 검토한다.
- [ ] 길이 구간별 성능을 비교한다.
- [ ] silence ratio 구간별 성능을 비교한다.
- [ ] RMS 구간별 성능을 비교한다.
- [ ] Detector와 Qwen의 예측 불일치 샘플을 추출한다.
- [ ] 동일 샘플에서 두 모델이 모두 틀리는지 확인한다.
- [ ] 오류 유형별 다음 실험 가설을 작성한다.

### 최종 산출물

- [ ] Detector checkpoint
- [ ] 샘플별 prediction/confidence JSONL 또는 CSV
- [ ] 모델·증강별 결과표
- [ ] Error analysis 보고서
- [ ] Hybrid system에 전달할 inference interface

---

# 역할 2. Qwen3 LoRA 및 생성 모델 실험

## 목표

공식 Qwen2.5-Omni baseline을 기준점으로 삼고, Qwen3 계열 후보의 audio input 지원과 LoRA 학습 가능성을 확인한 뒤 LoRA rank 및 생성 전략에 따른 성능·비용을 비교한다.

## A. Qwen3 후보와 호환성 확인

- [ ] 사용할 정확한 Qwen3 checkpoint를 확정한다.
- [ ] 해당 checkpoint가 raw audio input을 지원하는지 확인한다.
- [ ] 해당 checkpoint가 공개적으로 접근 가능한지 확인한다.
- [ ] checkpoint URL, license, revision, 다운로드 날짜를 기록한다.
- [ ] ms-swift가 해당 모델과 audio template을 지원하는지 확인한다.
- [ ] 지원되는 torch, transformers, ms-swift 조합을 기록한다.
- [ ] 단일 WAV text-generation smoke test를 수행한다.
- [ ] Qwen2.5-Omni와 동일한 XML 출력을 생성할 수 있는지 확인한다.

### 중단 조건

- [ ] Audio input이 지원되지 않으면 text-only Qwen3를 억지로 사용하지 않는다.
- [ ] 외부 ASR로 transcript를 만들어 입력하는 우회 방식은 규정 검토 전 사용하지 않는다.
- [ ] ms-swift 미지원 시 직접 통합 비용을 추정하고 프로젝트 범위를 재결정한다.

### 예상 시간

- 인력: 6-12시간
- GPU: 2-5 GPU-hour

## B. 공식 Qwen2.5 Baseline 기준점 확보

- [ ] 배포 checkpoint를 로드한다.
- [ ] Dev 일부에서 inference smoke test를 수행한다.
- [ ] Dev 전체 raw prediction을 생성한다.
- [ ] XML validity rate를 계산한다.
- [ ] Detection 성능을 계산한다.
- [ ] Evidence BERTScore와 final score를 계산한다.
- [ ] 최대 VRAM과 samples/second를 기록한다.
- [ ] Real/Fake별 설명 길이를 기록한다.

### 완료 조건

- [ ] 이후 모든 Qwen3 실험과 비교할 공식 baseline 표가 있다.

### 예상 시간

- 인력: 6-10시간
- GPU: 3-10 GPU-hour

## C. LoRA 학습 파이프라인

- [ ] Qwen3용 학습 JSONL을 생성한다.
- [ ] audio path, system prompt, user prompt, assistant target을 검증한다.
- [ ] XML target이 모든 샘플에서 유효한지 확인한다.
- [ ] assistant response token만 loss에 포함되는지 확인한다.
- [ ] overfit-one-batch test를 수행한다.
- [ ] 100-step smoke test를 수행한다.
- [ ] peak VRAM과 step time을 기록한다.
- [ ] 전체 1 epoch 및 3 epoch 예상 시간을 계산한다.
- [ ] checkpoint 저장과 재로딩을 검증한다.

### 기본 통제 변수

- [ ] 동일 seed
- [ ] 동일 Train/Dev manifest
- [ ] 동일 learning rate 또는 rank별 명시적 scaling rule
- [ ] 동일 epoch/step 수
- [ ] 동일 prompt와 decoding
- [ ] 동일 evaluator

### 예상 시간

- 인력: 10-18시간
- GPU: 5-15 GPU-hour

## D. LoRA Rank 비교

### 필수 비교

- [ ] Rank 8
- [ ] Rank 16
- [ ] Rank 32

### 여유가 있을 때

- [ ] Rank 4
- [ ] Rank 64

### 각 실험에서 기록할 항목

- [ ] Trainable parameter 수
- [ ] Peak VRAM
- [ ] Step time
- [ ] Train loss
- [ ] Dev loss
- [ ] Balanced accuracy
- [ ] BERTScore
- [ ] Final score
- [ ] XML validity rate
- [ ] 평균 생성 길이
- [ ] Checkpoint 크기

### 실험 해석

- [ ] Rank 증가에 따라 Detection과 BERTScore가 모두 증가하는지 확인한다.
- [ ] Rank 증가가 과적합을 유발하는지 확인한다.
- [ ] Rank 16 대비 Rank 32의 향상이 비용을 정당화하는지 판단한다.
- [ ] 최소 2 seed 반복이 필요한 실험을 선정한다.

### 예상 시간

- 인력: 10-20시간
- GPU: 25-80 GPU-hour
- 권장 GPU: 1 x 48 GB 이상에서 먼저 검증

## E. LoRA 적용 범위 비교

- [ ] LLM module만 LoRA를 적용한다.
- [ ] Audio encoder를 동결한다.
- [ ] Aligner를 동결한다.
- [ ] Aligner만 학습하는 설정을 검토한다.
- [ ] 가능한 경우 audio/projector/LLM LoRA 조합을 비교한다.
- [ ] 전체 모델 trainable parameter 비율을 기록한다.

### 완료 조건

- [ ] 제한된 GPU에서 가장 점수 대비 효율이 좋은 trainable module 구성이 정해져 있다.

## F. 생성 전략 실험

- [ ] 공식 장문 XML 생성을 평가한다.
- [ ] 50단어, 100단어, 원본 길이 Evidence를 비교한다.
- [ ] 클래스별 고정 Evidence template baseline을 평가한다.
- [ ] Detector Decision을 prompt에 제공하는 방식을 평가한다.
- [ ] Detector 결과로 `<Decision>`을 강제하는 방식을 평가한다.
- [ ] Spectral-only 설명을 평가한다.
- [ ] Spectral + Prosody 설명을 평가한다.
- [ ] XML schema를 prompt에 명시했을 때 파싱 성공률을 비교한다.
- [ ] temperature 0에서 재현성을 확인한다.

## G. 생성 오류 분석

- [ ] Decision 오류와 Evidence 오류를 구분한다.
- [ ] 판정 정답이지만 BERTScore가 낮은 10개를 검토한다.
- [ ] XML 누락·중복·빈 Evidence 사례를 수집한다.
- [ ] Decision과 Evidence가 논리적으로 모순되는 사례를 찾는다.
- [ ] 항목별 Spectral/Prosody/Pronunciation/Noise BERTScore를 계산한다.
- [ ] 설명 길이와 BERTScore 관계를 분석한다.
- [ ] Qwen2.5와 Qwen3가 서로 다르게 판단한 샘플을 추출한다.

### 최종 산출물

- [ ] Qwen2.5 공식 baseline 결과
- [ ] Qwen3 compatibility 보고서
- [ ] Rank 8/16/32 비교표
- [ ] 최적 LoRA checkpoint
- [ ] Prompt 및 generation ablation 표
- [ ] XML failure 및 explanation error 보고서

---

# 역할 3. 문헌·증강·GPU 타당성 조사

## 목표

기존 수상 시스템과 연구 방법을 동일한 기준으로 비교하고, ADD 2026 규정과 팀의 GPU 예산 안에서 실제 실행할 수 있는 실험 우선순위를 제안한다.

## A. 규정 체크리스트

- [ ] ADD 2026 Track 2 데이터 사용 규정을 요약한다.
- [ ] Pretrained language model 예외를 정리한다.
- [ ] 금지되는 pretrained speech component를 정리한다.
- [ ] 허용되는 non-speech augmentation을 정리한다.
- [ ] Evaluation-time denoise/dereverberation 조건을 정리한다.
- [ ] Test-time/online adaptation 금지 범위를 정리한다.
- [ ] 애매한 항목과 주최 측 문의가 필요한 항목을 분리한다.

### 산출물

- [ ] `Allowed / Conditional / Prohibited` 3단계 규정표

### 예상 시간

- 인력: 4-6시간

## B. 기존 대회 수상 시스템 조사

### 조사 대상

- [ ] ADD 2022 상위 시스템
- [ ] ADD 2023 상위 시스템
- [ ] ASVspoof 관련 상위 시스템
- [ ] Audio deepfake detection benchmark 논문
- [ ] Explainable audio deepfake detection 연구
- [ ] Audio-language model 기반 탐지 연구

### 논문별 기록 양식

- [ ] 논문/팀 이름
- [ ] 대회와 순위
- [ ] Task와 metric
- [ ] 입력 representation
- [ ] Backbone
- [ ] Pretrained model 사용 여부
- [ ] 학습 데이터
- [ ] Augmentation
- [ ] Loss
- [ ] Ensemble 여부
- [ ] GPU 종류와 수
- [ ] 학습 시간
- [ ] 공개 코드와 checkpoint
- [ ] ADD 2026 규정 적합성
- [ ] 우리 프로젝트에서 재현 가능한 아이디어

### 완료 조건

- [ ] 핵심 논문 10-15개가 동일한 표로 정리되어 있다.
- [ ] 단순 요약이 아니라 실행 가능한 아이디어 Top 5가 선정되어 있다.

### 예상 시간

- 인력: 15-25시간

## C. Augmentation 조사

### 후보

- [ ] Gain
- [ ] Time shift
- [ ] Resampling
- [ ] MP3/AAC compression
- [ ] Band-pass/low-pass filtering
- [ ] Pitch shift
- [ ] Speed perturbation
- [ ] Non-speech noise
- [ ] RIR/reverberation
- [ ] RawBoost 계열 signal perturbation
- [ ] Denoise/dereverberation

### 증강별 기록

- [ ] 규정 허용 여부
- [ ] 외부 데이터 필요 여부
- [ ] 데이터 출처
- [ ] 추천 파라미터 범위
- [ ] 적용 확률
- [ ] 계산 비용
- [ ] Detector 적용 가능성
- [ ] Qwen 적용 가능성
- [ ] 예상되는 failure mode
- [ ] 필요한 ablation

### 완료 조건

- [ ] 저비용·저위험 augmentation Top 3가 선정되어 있다.
- [ ] 외부 데이터가 필요한 augmentation은 출처와 subset이 검증되어 있다.

### 예상 시간

- 인력: 6-10시간

## D. GPU 실행 가능성 조사

### 후보 환경

- [ ] RTX 4090 24 GB
- [ ] A40/A6000 48 GB
- [ ] L40S 48 GB
- [ ] A100 80 GB
- [ ] H100 80 GB
- [ ] 다중 GPU 사용 가능 여부

### 모델별 100-step Profiling

- [ ] Log-Mel CNN
- [ ] LCNN/AASIST
- [ ] Qwen2.5-Omni inference
- [ ] Qwen3 inference
- [ ] Qwen3 LoRA rank 8
- [ ] Qwen3 LoRA rank 16
- [ ] Qwen3 LoRA rank 32

### 기록할 항목

- [ ] 모델 로딩 성공 여부
- [ ] Peak VRAM
- [ ] Batch size
- [ ] Gradient accumulation
- [ ] Step time
- [ ] Samples/second
- [ ] 예상 1 epoch 시간
- [ ] 예상 전체 학습 시간
- [ ] 예상 GPU-hour
- [ ] 예상 비용
- [ ] OOM 및 안정성 문제

### 완료 조건

- [ ] 모델과 GPU 조합별 `가능 / 조건부 가능 / 불가능` 표가 있다.
- [ ] 전체 프로젝트 GPU-hour 예산이 정해져 있다.
- [ ] 비용이 큰 실험의 중단 기준이 있다.

### 예상 시간

- 인력: 8-15시간
- GPU: 5-15 GPU-hour

## E. 실험 우선순위 제안

- [ ] 후보 아이디어를 예상 효과로 평가한다.
- [ ] 후보 아이디어를 구현 난이도로 평가한다.
- [ ] 후보 아이디어를 GPU 비용으로 평가한다.
- [ ] 후보 아이디어를 규정 위험으로 평가한다.
- [ ] 후보 아이디어를 재현 가능성으로 평가한다.
- [ ] 가장 작은 실험으로 검증 가능한 순서로 정렬한다.

### 추천 평가 기준

| 항목 | 점수 |
|---|---:|
| 예상 성능 향상 | 1-5 |
| 구현 난이도 | 1-5 |
| GPU 비용 | 1-5 |
| 규정 위험 | 1-5 |
| 분석 가치 | 1-5 |

### 최종 산출물

- [ ] 수상 시스템 및 연구 방법 비교표
- [ ] 규정 적합성 표
- [ ] Augmentation 우선순위 표
- [ ] GPU/비용 feasibility matrix
- [ ] 다음 실험 Top 5 제안

---

# 파트 간 전달물

| 제공 역할 | 전달받는 역할 | 전달물 |
|---|---|---|
| 문헌·GPU | Detector | 모델 및 증강 우선순위, 규정 위험 |
| 문헌·GPU | Qwen3 LoRA | 지원 GPU, 권장 rank, 예상 시간 |
| Detector | Qwen3 LoRA | 샘플별 prediction과 confidence |
| Qwen3 LoRA | Detector | Qwen prediction과 불일치 목록 |
| Detector/Qwen3 | 문헌·GPU | 실제 VRAM, 시간, 실패 로그 |
| 문헌·GPU | 전체 | 결과 해석과 다음 실험 우선순위 |

---

# 주간 체크포인트

## 1주차

- [ ] Detector: duration 및 handcrafted-feature baseline
- [ ] Qwen3: 후보 checkpoint와 audio/ms-swift 호환성 확인
- [ ] 문헌: 규정표와 조사 대상 논문 목록
- [ ] 공통: 평가 manifest 및 experiment ID 규칙 확정

## 2주차

- [ ] Detector: Log-Mel CNN 첫 결과
- [ ] Qwen3: 공식 Qwen2.5 baseline 결과
- [ ] 문헌: 핵심 논문 10개 비교표
- [ ] GPU: 100-step profiling 계획과 1차 결과

## 3주차

- [ ] Detector: LCNN/AASIST smoke test
- [ ] Qwen3: LoRA rank 8/16 smoke test
- [ ] 문헌: augmentation Top 3 확정
- [ ] 공통: 첫 error analysis 리뷰

## 4주차

- [ ] Detector: 최적 architecture 후보 선정
- [ ] Qwen3: rank 8/16/32 비교 완료
- [ ] GPU: 전체 실험 예산 재산정
- [ ] 공통: 5-8주차 본 실험 확정

## 5-8주차

- [ ] Detector augmentation 및 architecture 개선
- [ ] Qwen3 LoRA, prompt, explanation length 실험
- [ ] Detector + generator hybrid 결합
- [ ] 실패 결과 원인 분석과 다음 실험 결정

## 9주차

- [ ] 최종 ablation study
- [ ] Official Dev와 Clean Dev 비교
- [ ] Error analysis 완료
- [ ] 규정 준수 최종 점검

## 10주차

- [ ] 최종 checkpoint와 inference pipeline 고정
- [ ] 동일 환경에서 재현 실행
- [ ] GitHub README와 실행 방법 정리
- [ ] 최종 발표 자료 작성

---

# 역할별 총 예상 소요

| 역할 | 인력 시간 | GPU 시간 | 주요 병목 |
|---|---:|---:|---|
| Detector 프로토타입 | 45-80시간 | 30-70 GPU-hour | 모델·증강 반복 실험 |
| Qwen3 LoRA | 50-95시간 | 50-150 GPU-hour | Audio 호환성과 VRAM |
| 문헌·증강·GPU | 35-60시간 | 5-15 GPU-hour | 자료 비교와 profiling |

위 시간은 초기 추정치다. 모든 큰 학습은 100-step profiling 이후 실제 step time을 기준으로 다시 계산한다.

---

# 회의용 상태 요약

각 담당자는 주간 회의 전에 다음 다섯 줄을 업데이트한다.

```text
이번 주 완료:
핵심 결과:
실패 또는 blocker:
가능한 원인:
다음 주 최소 실험:
```
