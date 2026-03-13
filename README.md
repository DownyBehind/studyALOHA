# ALOHA + LeRobot Ubuntu 실험 정리 가이드  
**Ubuntu 기반 ALOHA simulation + LeRobot imitation learning 전체 실험 절차, 재현 방법, 개념 정리**

본 문서는 Ubuntu 환경에서 **MuJoCo 기반 ALOHA simulation**과 **LeRobot 기반 imitation learning(ACT policy)** 실험을 재현할 수 있도록 정리한 가이드입니다.  
목표는 다음과 같습니다.

1. **재부팅 또는 다음 세션에서도 동일한 실험을 재현**할 수 있도록 합니다.
2. 단순 명령어 나열이 아니라 **각 단계의 이론적 의미와 역할**을 이해할 수 있도록 합니다.
3. 개인 실험 로그를 넘어 **다른 사용자가 따라 할 수 있는 형식의 정식 실험 가이드**를 제공합니다.

본 문서는 두 층으로 구성됩니다.

- **일반화된 설명**: 동일한 환경에서 따라 할 수 있도록 공통 절차를 서술합니다.
- **실제 실험 환경 및 명령어**: 사용된 PC, 경로, 가상환경, 실행 명령을 예시로 제시합니다.

---

# 0. 참고 자료

본 가이드는 아래 공식 자료 및 실제 실험 과정을 바탕으로 정리하였습니다.

- Aloha Sim GitHub: https://github.com/google-deepmind/aloha_sim
- LeRobot 설치 문서: https://huggingface.co/docs/lerobot/en/installation
- MuJoCo Python 문서: https://mujoco.readthedocs.io/en/stable/python.html
- PyTorch CUDA memory 관리 문서: https://docs.pytorch.org/docs/stable/notes/cuda.html

공식 문서는 버전에 따라 변경될 수 있으므로, 버전 차이가 있을 경우 위 링크를 먼저 확인하는 것이 권장됩니다.

## 0.1 저장소 구조 및 Git 관리

본 프로젝트는 **studyALOHA** 단일 Git 저장소로 관리하며, 아래 두 디렉터리는 **외부 저장소를 클론한 서브모듈**입니다.

| 디렉터리   | 원본 저장소 |
|-----------|-------------|
| `aloha_sim/` | [google-deepmind/aloha_sim](https://github.com/google-deepmind/aloha_sim) |
| `lerobot/`   | [huggingface/lerobot](https://github.com/huggingface/lerobot) |

이 구조는 **Git 서브모듈**로 관리하는 것이 권장됩니다. 상위 저장소(studyALOHA)는 "현재 실험에서 사용하는 aloha_sim·lerobot의 커밋"만 기록하며, 실제 코드는 각 원본 저장소에서 유지됩니다.

### 이 저장소를 처음 클론할 때

```bash
# 서브모듈까지 한 번에 받기
git clone --recurse-submodules <이 저장소 URL> studyALOHA
cd studyALOHA
```

이미 일반 `git clone`만 수행한 경우:

```bash
git submodule update --init --recursive
```

### 서브모듈로 전환하는 방법 (aloha_sim, lerobot을 이미 일반 클론한 경우)

`aloha_sim`, `lerobot`을 일반 클론한 상태에서 서브모듈로 전환하려면 아래 순서를 따릅니다. **aloha_sim 또는 lerobot 내부를 수정한 경우, 먼저 백업하거나 해당 저장소에 커밋해 두는 것이 좋습니다.**

```bash
# 1) 상위 저장소 루트에서
cd /path/to/studyALOHA

# 2) 기존 디렉터리 제거 (내부 .git 때문에 내용만 지우고 서브모듈로 다시 받을 예정)
rm -rf aloha_sim lerobot

# 3) 서브모듈로 추가 (원격 URL + 사용할 브랜치/커밋이 기록됨)
git submodule add https://github.com/google-deepmind/aloha_sim.git aloha_sim
git submodule add https://github.com/huggingface/lerobot.git lerobot

# 4) 커밋
git add .gitmodules aloha_sim lerobot
git commit -m "Add aloha_sim and lerobot as submodules"
```

본 저장소는 **삭제 없이** 기존 `aloha_sim`, `lerobot` 디렉터리를 서브모듈로 등록해 두었습니다. 학습 중인 로컬 파일은 유지한 채, 아래 "버전 올리기" 절에 따라 원본만 `pull`하여 반영할 수 있습니다.

### 서브모듈이 가리키는 버전 올리기 (원본에서 업데이트 받기)

학습을 진행 중이어도 **원본(aloha_sim, lerobot)의 최신 변경만 반영**할 수 있습니다. 각 서브모듈 디렉터리에서 `pull`한 뒤, 상위 저장소에 "사용할 커밋"만 커밋하면 됩니다. **디렉터리를 삭제할 필요가 없으며, 로컬에서 생성한 파일(학습 결과 등)은 그대로 두고** 원본 코드만 갱신할 수 있습니다.

```bash
# aloha_sim 최신 반영
cd aloha_sim
git fetch origin
git pull origin main   # 기본 브랜치가 main이 아닐 수 있음 (예: master)
cd ..
git add aloha_sim
git commit -m "Update aloha_sim to latest"

# lerobot도 같은 방식
cd lerobot
git fetch origin
git pull origin main
cd ..
git add lerobot
git commit -m "Update lerobot to latest"
```

각 서브모듈 디렉터리에서 `git status`로 로컬 수정·추가 파일을 확인할 수 있습니다. 학습 체크포인트 등은 유지한 채 원본 코드만 갱신하려면, 위와 같이 `pull` 후 studyALOHA 루트에서 `git add`·`commit`을 수행하면 됩니다.

---

# 1. 본 문서가 다루는 전체 그림

실험 절차를 크게 요약하면 다음과 같습니다.

1. **Ubuntu에서 NVIDIA GPU 드라이버 정상화**
2. **PyTorch CUDA 사용 가능 여부 확인**
3. **MuJoCo 설치 및 최소 physics step 테스트**
4. **Aloha Sim viewer 실행 확인**
5. **LeRobot 설치 및 ALOHA environment 연동 확인**
6. **ALOHA insertion demonstration dataset으로 ACT policy 학습**
7. **checkpoint 저장 및 resume 학습**
8. **3k / 30k / 60k step policy evaluation**
9. **reward / success rate / video를 통한 policy 수준 해석**

본 문서는 단순 설치 가이드가 아니라, 위 전체 파이프라인을 다룹니다.

```text
GPU/OS 준비
→ Python 환경 준비
→ MuJoCo / Aloha Sim 동작 확인
→ LeRobot 학습 환경 준비
→ Dataset 기반 imitation learning
→ Checkpoint 저장
→ Evaluation
→ 결과 해석
```

---

# 2. ALOHA와 LeRobot의 역할 구분

본 절은 전체 이해의 핵심입니다.

## 2.1 ALOHA란

ALOHA는 본 가이드에서 **로봇 조작 문제를 정의하는 환경(domain)** 을 의미합니다.

ALOHA가 제공하는 항목은 다음과 같습니다.

- 로봇 구조: 양팔 로봇, 관절, 그리퍼
- 관측: 카메라 이미지, 상태값
- 행동 공간: joint action 또는 제어 입력
- 태스크: insertion, cube transfer 등
- 보상 및 성공 판정 규칙
- MuJoCo 기반 물리 시뮬레이션

즉 ALOHA는 **“어떤 문제를 풀 것인가”** 를 정의합니다.

예를 들어 `AlohaInsertion-v0`는 다음을 의미합니다.

- 로봇이 peg와 hole을 사용한 insertion task를 수행하며
- 이미지와 상태를 관측으로 받고
- action을 출력하며
- 환경이 reward와 success를 판정합니다.

## 2.2 LeRobot이란

LeRobot은 **정책(policy)을 학습·저장·평가하는 프레임워크**입니다.

LeRobot이 담당하는 역할은 다음과 같습니다.

- dataset 로딩
- policy architecture 구성 (예: ACT)
- optimizer / scheduler 설정
- train loop 실행
- checkpoint 저장
- eval loop 실행

즉 LeRobot은 **“그 문제를 어떤 모델로 학습시킬 것인가”** 를 담당합니다.

## 2.3 둘의 관계

요약하면 다음과 같습니다.

- **ALOHA = 문제 정의**
- **LeRobot = 학습 엔진**

비유하면 다음과 같습니다.

- ALOHA = 시험장, 시험 문제, 채점 기준
- LeRobot = 수험생을 훈련시키는 학습 시스템

본 가이드에서 진행한 실험은 다음과 같이 정의할 수 있습니다.

- **ALOHA insertion task** 를 대상으로
- **ALOHA sim demonstration dataset** 을 사용하여
- **LeRobot의 ACT policy** 를 학습하고
- **ALOHA env에서 평가**한 실험입니다.

---

# 3. 왜 simulation + imitation learning을 사용하는가

실제 로봇을 직접 학습시키는 것은 비용과 위험이 큽니다.

## 3.1 실제 로봇 학습의 어려움

- 하드웨어 비용이 큽니다.
- 시행착오 속도가 느립니다.
- 잘못된 policy가 기구를 손상시킬 수 있습니다.
- dataset 수집 비용이 큽니다.
- reset 및 반복 실험이 번거롭습니다.

## 3.2 시뮬레이션의 장점

- 반복 실험이 빠릅니다.
- 실패 비용이 낮습니다.
- evaluation을 여러 번 자동 수행하기 쉽습니다.
- debugging과 visualization이 용이합니다.

## 3.3 imitation learning의 장점

본 실험은 reinforcement learning이 아니라 **imitation learning**입니다.

즉 policy가 보상을 직접 탐색하는 것이 아니라,  
이미 수집된 **demonstration trajectory**를 참고하여 그 행동을 모방하도록 학습합니다.

이 방식은 다음에 유리합니다.

- 초기 학습 안정성
- sparse reward 문제 회피
- 조작 태스크에서의 빠른 성능 확보

ALOHA insertion과 같이 정밀 조작이 필요한 task에서는 imitation learning이 일반적인 시작점입니다.

---

# 4. 실험에 사용한 환경 (예시)

아래는 본 실험에 사용한 환경입니다.  
다른 사용자는 **자신의 환경에 맞게 경로 및 사양을 변경**하여 적용하면 됩니다.

## 4.1 하드웨어

- GPU: **NVIDIA RTX 3070 Laptop GPU (8GB)**
- CPU: Ryzen 9 5900HX
- RAM: 32GB

## 4.2 운영체제

- Ubuntu

## 4.3 작업 경로

```bash
~/study/workspace/physicalAI/studyALOHA
```

## 4.4 Python 환경 구성

본 실험에서는 두 개의 가상환경을 사용하였습니다.

### A. `aloha-venv`
용도:
- MuJoCo
- Aloha Sim viewer
- 시뮬레이터 검증

### B. `lerobot-py312`
용도:
- LeRobot 학습
- ALOHA env + dataset + ACT training + eval

환경을 분리한 이유는 다음과 같습니다.

- Aloha Sim은 Python 3.10 venv에서 안정적으로 동작을 확인하였고
- LeRobot main branch는 Python 3.12를 요구하며
- 두 환경을 분리하는 것이 의존성 충돌 회피에 유리하기 때문입니다.

---

# 5. 실험의 전체 흐름

실험은 크게 4단계로 구분됩니다.

## 단계 1. 시스템/GPU 세팅
- Secure Boot 비활성화
- NVIDIA 드라이버 정상화
- `nvidia-smi` 확인

## 단계 2. 시뮬레이터 세팅
- Python venv 생성
- MuJoCo 설치
- Aloha Sim clone 및 viewer 확인

## 단계 3. 학습 프레임워크 세팅
- Conda 설치
- Python 3.12 env 생성
- LeRobot 설치
- ALOHA env 등록 확인

## 단계 4. 정책 학습 및 평가
- ACT policy 학습
- checkpoint 저장
- 3k / 30k / 60k eval
- reward / success 추세 해석

---

# 6. GPU 드라이버 문제 해결 과정

초기에는 `nvidia-smi`가 실패하였습니다.

예시 오류:

```bash
nvidia-smi
NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver.
```

## 6.1 원인

실제 상태는 다음과 같았습니다.

- GPU 자체는 시스템에서 보임
- Ubuntu 추천 드라이버도 존재
- 하지만 `lsmod | grep nvidia` 결과가 비어 있음
- `Secure Boot enabled`

이 조합은 Ubuntu에서 자주 보이는 패턴으로,  
**Secure Boot 때문에 NVIDIA kernel module이 로드되지 않는 경우**가 많습니다.

## 6.2 해결 절차

1. BIOS/UEFI 진입
2. Secure Boot 비활성화
3. Ubuntu 부팅
4. NVIDIA 드라이버 설치/재설치
5. 재부팅 후 확인

## 6.3 확인 명령어

```bash
mokutil --sb-state
lsmod | grep nvidia
nvidia-smi
```

정상 예시:

```bash
SecureBoot disabled
```

`nvidia-smi` 실행 시 아래 정보가 표시되어야 합니다.

- Driver Version
- CUDA Version
- GPU 이름 (RTX 3070 Laptop GPU)

---

# 7. Ubuntu 재부팅 후 가장 먼저 할 확인

재부팅 또는 다음 세션에서 작업을 재개할 때는 먼저 아래 항목을 확인하는 것이 좋습니다.

## 7.1 GPU 상태 확인

```bash
nvidia-smi
```

## 7.2 GPU 모듈 확인

```bash
lsmod | grep nvidia
```

## 7.3 CUDA 가능한 PyTorch 확인
(LeRobot 환경에서)

```bash
conda activate lerobot-py312
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"
```

---

# 8. Aloha Sim 재실행 방법

시뮬레이터가 정상 동작하는지 확인하려면 아래 순서대로 실행합니다.

## 8.1 가상환경 활성화

```bash
cd ~/study/workspace/physicalAI/studyALOHA
source aloha-venv/bin/activate
```

## 8.2 OpenGL backend 설정

```bash
export MUJOCO_GL=egl
```

설명:
- `egl`은 headless 또는 GPU 가속 렌더링에 유리합니다.
- 환경에 따라 `glfw` 또는 자동 선택이 적합할 수 있으며, Ubuntu에서는 `egl`을 먼저 시도하는 것이 일반적으로 안전합니다.

## 8.3 Aloha Sim 저장소로 이동

```bash
cd ~/study/workspace/physicalAI/studyALOHA/aloha_sim
```

## 8.4 Viewer 실행

```bash
python aloha_sim/viewer.py --policy=no_policy --task_name=HandOverBanana
```

설명:
- `no_policy`는 학습된 정책 없이 viewer와 task만 확인하는 모드입니다.
- 이 단계는 시뮬레이터 기동 여부를 확인하는 최소 검증입니다.

---

# 9. MuJoCo 최소 테스트

MuJoCo가 정상 설치되었는지 빠르게 검증하는 최소 예시는 아래와 같습니다.

```bash
python - <<'PY'
import mujoco

xml = """
<mujoco>
  <worldbody>
    <light pos="0 0 3"/>
    <geom type="plane" size="1 1 0.1"/>
    <body pos="0 0 1">
      <joint type="free"/>
      <geom type="box" size="0.1 0.1 0.1"/>
    </body>
  </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

for _ in range(100):
    mujoco.mj_step(model, data)

print("MuJoCo step OK")
print("qpos:", data.qpos)
PY
```

정상이라면 `MuJoCo step OK`가 출력됩니다.

이 테스트는 다음을 한 번에 확인합니다.

- Python binding이 import되는지
- XML 모델 생성이 되는지
- physics step이 정상적으로 수행되는지

---

# 10. LeRobot 환경 재실행 방법

LeRobot 학습·평가를 다시 시작할 때는 아래 명령을 사용합니다.

## 10.1 Conda env 활성화

```bash
conda activate lerobot-py312
```

## 10.2 저장소 이동

```bash
cd ~/study/workspace/physicalAI/studyALOHA/lerobot
```

## 10.3 GPU 상태 확인

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"
```

---

# 11. ALOHA environment 등록 확인

LeRobot에서 ALOHA env가 정상 등록되었는지 확인할 때 사용한 코드는 아래와 같습니다.

```bash
python - <<'PY'
import gymnasium as gym
import gym_aloha

ids = sorted([k for k in gym.envs.registry.keys() if "aloha" in k.lower()])
print("ALOHA env ids:")
for env_id in ids:
    print(env_id)
PY
```

확인된 env id 예시:

```bash
gym_aloha/AlohaInsertion-v0
gym_aloha/AlohaTransferCube-v0
```

ALOHA insertion task 사용 시에는 위와 동일한 이름 체계를 사용해야 합니다.

---

# 12. 사용한 데이터셋

학습에 사용한 dataset은 다음과 같습니다.

```bash
lerobot/aloha_sim_insertion_human
```

이 dataset은 ALOHA insertion task용 demonstration dataset이며,  
다음 정보를 포함합니다.

- top camera image
- robot state
- action trajectory

즉 policy는 이 demonstration을 참고하여 다음을 학습합니다.

- 이미지를 해석하여 물체·로봇 상태를 파악하고
- 주어진 상태에서 사람이 수행한 action을 모방하도록

이것이 imitation learning의 핵심입니다.

---

# 13. 사용한 policy: ACT

## 13.1 ACT란

ACT는 **Action Chunking Transformer**입니다.

핵심 아이디어는 다음과 같습니다.

- action을 한 step씩 예측하는 대신
- **여러 step의 action chunk를 한 번에 예측**
- image + proprioception(state)를 함께 입력으로 사용
- 시간적으로 연결된 조작 행동을 더 안정적으로 모델링

## 13.2 ALOHA에 적합한 이유

ALOHA insertion과 같은 task에서는:

- 한 순간의 action보다
- 일정 구간 동안의 연속적인 행동 구조가 중요합니다.

예를 들어:
- peg 접근
- 자세 정렬
- 삽입 시도
- 미세 보정

은 모두 시간 연속성이 강합니다.  
ACT는 이러한 robot manipulation imitation learning에 적합합니다.

## 13.3 본 실험의 주요 ACT 설정 예시

로그에서 확인한 대표 설정:

- `vision_backbone = resnet18`
- `chunk_size = 100`
- `n_action_steps = 100`
- `n_obs_steps = 1`
- `dim_model = 512`

즉 현재 policy는 대략:

- 현재 관측(image + state)을 보고
- 앞으로 100 step 정도의 action sequence를 예측하는 구조

로 이해할 수 있습니다.

---

# 14. 학습 명령어 정리

아래는 실제로 사용한 명령어입니다.  
경로는 본 실험 환경 기준이며, 다른 사용자는 자신의 환경에 맞게 변경하면 됩니다.

## 14.1 3k 학습

```bash
conda activate lerobot-py312
cd ~/study/workspace/physicalAI/studyALOHA/lerobot

OUTDIR=/home/kimdawoon/study/workspace/physicalAI/studyALOHA/outputs/act_aloha_insertion_3k

TOKENIZERS_PARALLELISM=false MUJOCO_GL=egl \
lerobot-train \
  --output_dir="$OUTDIR" \
  --job_name=act_aloha_insertion_3k \
  --policy.type=act \
  --policy.device=cuda \
  --policy.repo_id=kimdawoon/act-aloha-insertion-3k \
  --policy.push_to_hub=false \
  --env.type=aloha \
  --env.task=AlohaInsertion-v0 \
  --dataset.repo_id=lerobot/aloha_sim_insertion_human \
  --batch_size=2 \
  --steps=3000 \
  --log_freq=50 \
  --save_freq=500 \
  --wandb.enable=false
```

설명:
- `steps=3000`: smoke test 이후 첫 의미 있는 짧은 학습
- `batch_size=2`: RTX 3070 8GB에서 초기엔 가능했던 값
- `save_freq=500`: 500 step마다 checkpoint 저장

---

# 15. checkpoint 구조 해석

LeRobot이 저장하는 checkpoint는 대략 다음 구조를 가집니다.

```text
checkpoints/
  000500/
    pretrained_model/
      config.json
      model.safetensors
      train_config.json
      ...
    training_state/
      optimizer_state.safetensors
      rng_state.safetensors
      training_step.json
  001000/
  001500/
  ...
  last/
```

## 15.1 `pretrained_model`
policy 본체가 저장됩니다.

주요 파일:
- `model.safetensors`
- `train_config.json`

이 경로를 **eval용 policy path**로 사용합니다.

## 15.2 `training_state`
resume에 필요한 정보가 저장됩니다.

예:
- optimizer state
- random state
- current step

요약하면:

- `pretrained_model` = 평가/추론용
- `training_state` = 학습 재개용

입니다.

---

# 16. 배치(batch size)의 의미

## 16.1 정의
batch size는 **한 번의 update에 사용하는 샘플 수**를 의미합니다.

예:
- `batch_size=2` → step마다 2개 sample을 동시에 사용
- `batch_size=1` → step마다 1개 sample만 사용

## 16.2 batch size를 늘리면
장점:
- gradient가 덜 noisy해질 수 있음
- throughput이 올라갈 수 있음

단점:
- GPU 메모리를 많이 사용
- image 기반 policy에서는 특히 메모리 부담이 큼

## 16.3 본 실험에서의 의미
RTX 3070 Laptop 8GB 환경에서는:

- 짧은 학습: `batch_size=2` 가능
- 장시간 학습 + eval 동시 진행: OOM 발생 가능
- 안정적 장기 학습: `batch_size=1`이 실용적

즉 이 환경에서는 **배치를 늘리기보다 학습을 오래 진행하고 checkpoint를 잘 관리하는 것**이 더 중요하였습니다.

---

# 17. step 수는 무엇을 의미하는가

## 17.1 step의 의미
여기서 `steps`는 **optimizer update 횟수**로 이해하면 됩니다.

즉:
- 1 step = 1회 parameter update
- 3000 step = 3000회 학습 업데이트

## 17.2 dataset frame 수보다 더 많이 학습할 수 있는 이유
로그상 dataset frame 수는 25000인데, 학습은 60000 step까지 진행할 수 있습니다.

이는 정상입니다.

이유:
- dataset을 한 번만 사용하는 것이 아니라
- 여러 epoch에 걸쳐 반복하여 사용하기 때문입니다.

즉:
- `dataset.num_frames`는 데이터 크기
- `steps`는 학습을 얼마나 진행할지

를 나타냅니다.

## 17.3 본 실험에서의 의미
- 3k: 초기 정책이 구조를 배우기 시작
- 30k: task를 어느 정도 이해
- 60k: 상당히 깊은 상태까지 자주 도달

insertion과 같이 어려운 task는 일반적으로 **몇천 step으로 완료되지 않습니다.**

---

# 18. OOM 원인

본 실험에서 발생한 OOM은 크게 두 가지 유형이었습니다.

## 18.1 장시간 train 중 OOM
원인:
- vision backbone + ACT + image input
- 8GB GPU의 제한
- `batch_size=2` 장시간 유지

해결:
- `batch_size=1`
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

## 18.2 train 중 자동 eval에서 OOM
원인:
- train은 batch 1로 버텨도
- 중간 eval은 별도 env와 policy inference가 추가됨
- eval batch가 기본적으로 커서 메모리 사용량 증가

해결:
- 학습 중 eval 사실상 비활성화
- 학습과 eval을 분리

## 18.3 권장 메모리 완화 옵션

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

이 옵션은 PyTorch allocator가 메모리 단편화를 완화하는 데 도움이 될 수 있습니다.

---

# 19. resume 학습 명령어 정리

## 19.1 30k run을 이어서 재개할 때

```bash
OUTDIR=/home/kimdawoon/study/workspace/physicalAI/studyALOHA/outputs/act_aloha_insertion_30k
CFG=$OUTDIR/checkpoints/020000/pretrained_model/train_config.json

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
TOKENIZERS_PARALLELISM=false MUJOCO_GL=egl \
lerobot-train \
  --config_path="$CFG" \
  --output_dir="$OUTDIR" \
  --resume=true \
  --batch_size=1 \
  --eval_freq=100000000 \
  --save_freq=5000 \
  --log_freq=100 \
  --wandb.enable=false
```

## 19.2 60k로 이어갈 때

```bash
OUTDIR=/home/kimdawoon/study/workspace/physicalAI/studyALOHA/outputs/act_aloha_insertion_30k
CFG=$OUTDIR/checkpoints/030000/pretrained_model/train_config.json

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
TOKENIZERS_PARALLELISM=false MUJOCO_GL=egl \
lerobot-train \
  --config_path="$CFG" \
  --output_dir="$OUTDIR" \
  --resume=true \
  --batch_size=1 \
  --steps=60000 \
  --eval_freq=100000000 \
  --save_freq=5000 \
  --log_freq=100 \
  --wandb.enable=false
```

## 19.3 100k까지 이어갈 때

```bash
OUTDIR=/home/kimdawoon/study/workspace/physicalAI/studyALOHA/outputs/act_aloha_insertion_30k
CFG=$OUTDIR/checkpoints/060000/pretrained_model/train_config.json

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
TOKENIZERS_PARALLELISM=false MUJOCO_GL=egl \
lerobot-train \
  --config_path="$CFG" \
  --output_dir="$OUTDIR" \
  --resume=true \
  --batch_size=1 \
  --steps=100000 \
  --eval_freq=100000000 \
  --save_freq=5000 \
  --log_freq=100 \
  --wandb.enable=false
```

---

# 20. 평가(eval) 명령어 정리

## 20.1 3k checkpoint 평가

```bash
MODEL=/home/kimdawoon/study/workspace/physicalAI/studyALOHA/outputs/act_aloha_insertion_3k/checkpoints/003000/pretrained_model

MUJOCO_GL=egl \
lerobot-eval \
  --policy.path=$MODEL \
  --env.type=aloha \
  --env.task=AlohaInsertion-v0 \
  --eval.n_episodes=10 \
  --eval.batch_size=10 \
  --env.render_mode=rgb_array
```

## 20.2 30k checkpoint 평가

```bash
MODEL=/home/kimdawoon/study/workspace/physicalAI/studyALOHA/outputs/act_aloha_insertion_30k/checkpoints/030000/pretrained_model

MUJOCO_GL=egl \
lerobot-eval \
  --policy.path=$MODEL \
  --env.type=aloha \
  --env.task=AlohaInsertion-v0 \
  --eval.n_episodes=10 \
  --eval.batch_size=10 \
  --env.render_mode=rgb_array
```

## 20.3 60k checkpoint 평가

```bash
MODEL=/home/kimdawoon/study/workspace/physicalAI/studyALOHA/outputs/act_aloha_insertion_30k/checkpoints/060000/pretrained_model

MUJOCO_GL=egl \
lerobot-eval \
  --policy.path=$MODEL \
  --env.type=aloha \
  --env.task=AlohaInsertion-v0 \
  --eval.n_episodes=10 \
  --eval.batch_size=10 \
  --env.render_mode=rgb_array
```

## 20.4 human render가 필요할 때
`human` render 모드 사용 시 `pygame`이 필요할 수 있습니다.

설치:
```bash
pip install pygame
```

그 후:
```bash
MUJOCO_GL=egl \
lerobot-eval \
  --policy.path=$MODEL \
  --env.type=aloha \
  --env.task=AlohaInsertion-v0 \
  --eval.n_episodes=10 \
  --eval.batch_size=1 \
  --env.render_mode=human
```

동작을 시각적으로 확인하려면 `batch_size=1`이 더 적합합니다.

---

# 21. reward와 success rate 해석

평가 시 확인하는 주요 지표는 다음과 같습니다.

- `avg_sum_reward`
- `avg_max_reward`
- `pc_success`

## 21.1 `pc_success`
최종 성공률입니다.

예:
- `0.0` → 성공 판정을 한 번도 못 받음
- `0.2` → 10개 중 2개 성공

## 21.2 `avg_sum_reward`
에피소드 전체에서 받은 reward 총합의 평균입니다.

높을수록 policy가 reward 구조상 더 유리한 행동을 하고 있음을 의미합니다.

## 21.3 `avg_max_reward`
각 episode에서 도달한 최대 reward의 평균입니다.

이 값이 상승하는 것은 일반적으로 policy가 **더 깊은 성공 관련 상태**까지 도달하고 있음을 의미합니다.

---

# 22. 본 실험의 성능 추이

## 22.1 3k eval
- `avg_sum_reward = 3.9`
- `avg_max_reward = 0.1`
- `pc_success = 0.0`

해석:
- 거의 초기 학습 단계

## 22.2 30k eval
- `avg_sum_reward = 23.9`
- `avg_max_reward = 0.4`
- `pc_success = 0.0`

해석:
- task 구조를 이해하기 시작한 단계

## 22.3 60k eval
- `avg_sum_reward = 160.3`
- `avg_max_reward = 1.4`
- `pc_success = 0.0`

고득점 episode:
- `379`
- `442`
- `421`
- `330`

해석:
- 성공률은 아직 0이지만
- policy가 상당히 깊은 상태까지 자주 들어감
- 마지막 정밀 삽입/성공 판정 직전에서 실패할 가능성 높음

즉 현재 policy는 **완전 실패가 아니라, 성공 직전 behavior를 반복적으로 만들어내는 단계**로 보는 것이 자연스럽습니다.

---

# 23. 전체 프로세스에서의 위치

Robot learning 전체 흐름을 크게 나누면 다음과 같습니다.

1. 환경 구축
2. smoke test
3. 초기 학습
4. task structure 학습
5. 성공률 상승 구간
6. 안정화/정교화

본 실험은 다음까지 진행된 상태입니다.

- 환경 구축 완료
- smoke test 완료
- 초기 학습 완료
- task structure 학습 진행 중

즉 **인프라 구축 단계는 완료되었고, 실제 정책 성능을 끌어올리는 구간**에 해당합니다.

요약하면:

- 3k: 어떻게 움직일지 감을 잡는 단계
- 30k: task를 이해하기 시작한 단계
- 60k: 거의 맞는 행동을 자주 만드는 단계
- 100k 이후: success가 0을 벗어날 수도 있는 단계

---

# 24. 작업 재개 시 빠른 체크리스트

## 24.1 GPU 상태 확인

```bash
nvidia-smi
```

## 24.2 Aloha Sim viewer 확인

```bash
cd ~/study/workspace/physicalAI/studyALOHA
source aloha-venv/bin/activate
export MUJOCO_GL=egl
cd aloha_sim
python aloha_sim/viewer.py --policy=no_policy --task_name=HandOverBanana
```

## 24.3 LeRobot 환경 진입

```bash
conda activate lerobot-py312
cd ~/study/workspace/physicalAI/studyALOHA/lerobot
```

## 24.4 GPU/PyTorch 확인

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

## 24.5 60k checkpoint 재평가

```bash
MODEL=/home/kimdawoon/study/workspace/physicalAI/studyALOHA/outputs/act_aloha_insertion_30k/checkpoints/060000/pretrained_model

MUJOCO_GL=egl \
lerobot-eval \
  --policy.path=$MODEL \
  --env.type=aloha \
  --env.task=AlohaInsertion-v0 \
  --eval.n_episodes=10 \
  --eval.batch_size=10 \
  --env.render_mode=rgb_array
```

## 24.6 100k까지 이어서 학습

```bash
OUTDIR=/home/kimdawoon/study/workspace/physicalAI/studyALOHA/outputs/act_aloha_insertion_30k
CFG=$OUTDIR/checkpoints/060000/pretrained_model/train_config.json

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
TOKENIZERS_PARALLELISM=false MUJOCO_GL=egl \
lerobot-train \
  --config_path="$CFG" \
  --output_dir="$OUTDIR" \
  --resume=true \
  --batch_size=1 \
  --steps=100000 \
  --eval_freq=100000000 \
  --save_freq=5000 \
  --log_freq=100 \
  --wandb.enable=false
```

---

# 25. 앞으로의 추천 실험

## 추천 1. high reward episode 영상 확인
reward가 높았던 episode의 mp4를 우선 확인합니다.

60k eval 기준 reward가 높은 episode가 있으므로,  
영상에서 다음 항목을 확인하는 것이 좋습니다.

- peg를 hole 근처까지 정확히 가져가는가
- 마지막 정렬에서 흔들리는가
- 삽입 직전 자세가 무너지는가
- 한쪽 팔/그리퍼 타이밍이 어긋나는가

## 추천 2. 100k까지 이어서 학습
현재 reward 추세만 보면 계속 학습할 가치가 충분합니다.

## 추천 3. 100k 이후 다시 eval
이때 다음 항목을 확인합니다.

- `pc_success`가 0을 벗어나는가
- high reward episode 수가 늘어나는가
- avg_sum_reward가 더 상승하는가

---

# 26. 마지막 정리

본 가이드에서 다룬 내용을 요약하면 다음과 같습니다.

**Ubuntu 환경에서 NVIDIA GPU, MuJoCo, Aloha Sim, LeRobot, ALOHA insertion demonstration dataset, ACT policy를 이용한 robot imitation learning 실험 파이프라인을 구축하고, 3k → 30k → 60k 단계별 학습과 evaluation을 통해 policy의 성능 변화를 확인하였습니다.**

현재 상태를 정리하면 다음과 같습니다.

**정책은 아직 최종 success 판정을 달성하지 못했으나, reward 관점에서는 ALOHA insertion task의 상당 부분을 학습하였으며, 고득점 episode가 반복적으로 나타나는 것으로 보아 추가 학습을 계속할 가치가 충분한 상태입니다.**

---

# 부록 A. Git으로 프로젝트·서브모듈 관리

본 프로젝트는 **studyALOHA**(상위 저장소) 하나로 전체를 관리하며, `aloha_sim`·`lerobot`은 **서브모듈**로 원본 저장소를 참조합니다. 아래는 일상적인 Git 사용 방법 정리입니다.

## A.1 저장소 구도

| 대상 | 역할 | 원격 URL 예시 |
|------|------|----------------|
| **studyALOHA** (상위) | 내 실험 문서·설정·서브모듈 “버전” 관리 | `https://github.com/DownyBehind/studyALOHA.git` |
| **aloha_sim** (서브모듈) | 원본 코드만 참조, 업데이트는 원본에서 pull | `https://github.com/google-deepmind/aloha_sim.git` |
| **lerobot** (서브모듈) | 위와 동일 | `https://github.com/huggingface/lerobot.git` |

상위 저장소에는 **README, .gitignore, .gitmodules**와 **서브모듈이 가리키는 커밋**만 커밋·푸시합니다. 서브모듈 폴더 내 실제 코드는 각 원본 저장소에서 관리됩니다.

## A.2 프로젝트(studyALOHA) 업데이트

문서 수정, .gitignore 변경, 서브모듈이 가리키는 커밋 변경 후 **상위 저장소만** 푸시하는 흐름입니다.

```bash
# 1) studyALOHA 루트에서
cd /path/to/studyALOHA

# 2) 변경 사항 확인 (서브모듈은 "어떤 커밋을 쓰는지"만 보임)
git status

# 3) 올릴 파일 스테이징 (서브모듈 버전을 올렸다면 aloha_sim, lerobot 도 추가)
git add README.md .gitignore   # 필요 시 aloha_sim lerobot

# 4) 커밋 후 원격에 반영
git commit -m "문서 정리 및 서브모듈 버전 반영"
git push origin master
```

다른 PC에서 이 저장소를 **처음** 받을 때:

```bash
git clone --recurse-submodules https://github.com/DownyBehind/studyALOHA.git studyALOHA
cd studyALOHA
```

이미 클론만 해 둔 경우(서브모듈이 비어 있을 때):

```bash
git pull origin master
git submodule update --init --recursive
```

## A.3 서브모듈(aloha_sim, lerobot) 업데이트

원본(google-deepmind/aloha_sim, huggingface/lerobot)이 업데이트됐을 때, **그 변경만 받아서** studyALOHA에 “이제 이 커밋 쓴다”고 기록하는 방법이다. 로컬에서 만든 학습 결과·체크포인트는 건드리지 않는다.

### aloha_sim 최신 반영

```bash
cd /path/to/studyALOHA/aloha_sim
git fetch origin
git pull origin main   # 또는 master 등, 원본의 기본 브랜치에 맞춤
cd ..
git add aloha_sim
git commit -m "Update aloha_sim to latest"
git push origin master   # 상위 저장소에 반영
```

### lerobot 최신 반영

```bash
cd /path/to/studyALOHA/lerobot
git fetch origin
git pull origin main
cd ..
git add lerobot
git commit -m "Update lerobot to latest"
git push origin master
```

### 두 서브모듈 한 번에 최신으로 맞추기

```bash
cd /path/to/studyALOHA
git submodule update --remote aloha_sim
git submodule update --remote lerobot
git add aloha_sim lerobot
git commit -m "Update aloha_sim and lerobot to latest"
git push origin master
```

`--remote`는 각 서브모듈의 원격 추적 브랜치 기준 최신 커밋으로 맞춥니다. 브랜치가 `main`이 아니면 `.gitmodules` 또는 해당 서브모듈의 `branch` 설정을 확인합니다.

## A.4 일상적인 Git 관리 요약

| 하고 싶은 일 | 어디서 실행 | 대략적인 순서 |
|-------------|-------------|----------------|
| README·설정만 수정해서 GitHub에 반영 | studyALOHA 루트 | `git add` → `commit` → `push` |
| 원본 aloha_sim 코드만 최신으로 맞추기 | `aloha_sim`에서 pull 후 studyALOHA 루트에서 | `cd aloha_sim` → `git pull origin main` → `cd ..` → `git add aloha_sim` → `commit` → `push` |
| 원본 lerobot 코드만 최신으로 맞추기 | `lerobot`에서 pull 후 studyALOHA 루트에서 | 위와 동일하게 `lerobot` 기준으로 |
| 다른 PC에서 프로젝트 받기 | 새 PC | `git clone --recurse-submodules <URL>` 또는 클론 후 `git submodule update --init --recursive` |
| 서브모듈 안 로컬 변경(학습 결과 등) 확인 | `aloha_sim` 또는 `lerobot` 안에서 | `git status` (상위 저장소에는 서브모듈 “커밋”만 올리면 됨) |

서브모듈 폴더 안에서 수정한 파일은 **원본 저장소에 커밋하지 않는 한** 상위 저장소 `git status`에 “modified content”로만 보입니다. 학습 체크포인트·결과는 유지한 채 원본 코드만 반영하려면 A.3과 같이 pull 후 상위에서 `git add`·`commit`·`push`를 수행하면 됩니다.