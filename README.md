# ALOHA + LeRobot Ubuntu 실험 정리 가이드  
**Ubuntu 기반 ALOHA simulation + LeRobot imitation learning 전체 실험 절차, 재현 방법, 개념 정리**

이 문서는 Ubuntu 환경에서 **MuJoCo 기반 ALOHA simulation**과 **LeRobot 기반 imitation learning(ACT policy)** 실험을 재현할 수 있도록 정리한 문서다.  
목표는 세 가지다.

1. **다음 날 PC를 다시 켜도 바로 실험을 재현**할 수 있게 한다.
2. 단순 명령어 나열이 아니라, **각 단계의 이론적 의미와 역할**까지 이해할 수 있게 한다.
3. 개인 실험 로그를 넘어서, **다른 사람이 읽어도 따라갈 수 있는 형태의 포멀한 실험 가이드**로 만든다.

이 문서는 두 층으로 구성되어 있다.

- **일반화된 설명**: 누구나 비슷한 환경에서 따라 할 수 있도록 설명
- **이번 실험의 실제 환경과 명령어**: 사용한 PC, 경로, 가상환경, 실제 실행 명령을 예시로 제공

---

# 0. 참고 자료

이 가이드는 아래 공식 자료와 오늘의 실제 실험 과정을 바탕으로 정리했다.

- Aloha Sim GitHub: https://github.com/google-deepmind/aloha_sim
- LeRobot 설치 문서: https://huggingface.co/docs/lerobot/en/installation
- MuJoCo Python 문서: https://mujoco.readthedocs.io/en/stable/python.html
- PyTorch CUDA memory 관리 문서: https://docs.pytorch.org/docs/stable/notes/cuda.html

공식 문서는 시간이 지나며 변경될 수 있으므로, 큰 버전 차이가 생기면 위 링크를 먼저 확인하는 것이 좋다.

## 0.1 저장소 구조 및 Git 관리

이 프로젝트는 **studyALOHA** 하나의 Git 저장소로 관리하며, 아래 두 디렉터리는 **외부에서 클론한 저장소**다.

| 디렉터리   | 원본 저장소 |
|-----------|-------------|
| `aloha_sim/` | [google-deepmind/aloha_sim](https://github.com/google-deepmind/aloha_sim) |
| `lerobot/`   | [huggingface/lerobot](https://github.com/huggingface/lerobot) |

이런 구조는 **Git 서브모듈**로 관리하는 것이 좋다. 상위 저장소(studyALOHA)는 “지금 이 실험은 aloha_sim의 이 커밋, lerobot의 이 커밋과 함께 돌아간다”만 기록하고, 실제 코드는 각 저장소가 그대로 유지된다.

### 이 저장소를 처음 클론할 때

```bash
# 서브모듈까지 한 번에 받기
git clone --recurse-submodules <이 저장소 URL> studyALOHA
cd studyALOHA
```

이미 `git clone`을 했다면:

```bash
git submodule update --init --recursive
```

### 서브모듈로 전환하는 방법 (이미 aloha_sim, lerobot을 일반 클론해 둔 경우)

현재처럼 `aloha_sim`, `lerobot`을 그냥 클론해 둔 상태에서 서브모듈로 바꾸려면 아래 순서를 따른다. **aloha_sim / lerobot 안에서 수정한 내용이 있다면 먼저 백업하거나 커밋해 두자.**

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

이 저장소는 **삭제 없이** 기존 `aloha_sim`, `lerobot`을 서브모듈로 등록해 두었다. 학습 중인 로컬 파일은 그대로 두고, 위 "버전 올리기"처럼 원본만 `pull`해서 반영하면 된다.

### 서브모듈이 가리키는 버전 올리기 (원본에서 업데이트 받기)

학습 중에도 **원본(aloha_sim, lerobot)의 최신 변경만 받아서 반영**하고 싶다면, 각 폴더에서 `pull`한 뒤 상위 저장소에 "이제 이 커밋을 쓴다"만 커밋하면 된다. **폴더를 지울 필요 없고, 로컬에서 만든 파일(학습 결과 등)은 그대로 둔 채** 원본만 갱신할 수 있다.

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

각 서브모듈에서 `git status`로 로컬 수정/추가 파일을 확인할 수 있다. 학습 체크포인트 등은 그대로 두고, 원본 코드만 올리고 싶다면 위처럼 `pull` 후 studyALOHA 쪽에서 `git add`·`commit`만 하면 된다.

---

# 1. 이 문서가 다루는 전체 그림

오늘 한 일을 아주 크게 요약하면 다음과 같다.

1. **Ubuntu에서 NVIDIA GPU 드라이버를 정상화**
2. **PyTorch에서 CUDA 사용 가능 여부 확인**
3. **MuJoCo 설치 및 최소 physics step 테스트**
4. **Aloha Sim viewer 실행 확인**
5. **LeRobot 설치 및 ALOHA environment 연동 확인**
6. **ALOHA insertion demonstration dataset으로 ACT policy 학습**
7. **checkpoint 저장 및 resume 학습**
8. **3k / 30k / 60k step policy evaluation**
9. **reward / success rate / video를 통해 현재 policy 수준 해석**

즉 이 문서는 단순 설치 가이드가 아니라, 다음 전체 파이프라인을 다룬다.

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

이 부분이 가장 중요하다.

## 2.1 ALOHA는 무엇인가

ALOHA는 여기서 **로봇 조작 문제를 정의하는 환경(domain)** 이다.

ALOHA가 제공하는 것은 대략 다음이다.

- 로봇 구조: 양팔 로봇, 관절, 그리퍼
- 관측: 카메라 이미지, 상태값
- 행동 공간: joint action 또는 제어 입력
- 태스크: insertion, cube transfer 등
- 보상 및 성공 판정 규칙
- MuJoCo 기반 물리 시뮬레이션

즉 ALOHA는 **“어떤 문제를 풀 것인가”** 를 정의한다.

예를 들어 `AlohaInsertion-v0`는 다음을 의미한다.

- 로봇이 peg와 hole을 사용한 insertion task를 수행해야 하고
- 이미지와 상태를 관측으로 받고
- action을 출력하며
- 환경이 reward와 success를 판정한다.

## 2.2 LeRobot은 무엇인가

LeRobot은 **정책(policy)을 학습, 저장, 평가하는 프레임워크**다.

LeRobot이 담당하는 일은:

- dataset 로딩
- policy architecture 구성 (예: ACT)
- optimizer / scheduler 설정
- train loop 실행
- checkpoint 저장
- eval loop 실행

즉 LeRobot은 **“그 문제를 어떤 모델로 학습시킬 것인가”** 를 담당한다.

## 2.3 둘의 관계

둘의 관계를 가장 쉽게 말하면:

- **ALOHA = 문제 정의**
- **LeRobot = 학습 엔진**

비유하면 다음과 같다.

- ALOHA = 시험장, 시험 문제, 채점 기준
- LeRobot = 수험생을 훈련시키는 학습 시스템

오늘 진행한 실험은 정확히 말하면:

- **ALOHA insertion task를 대상으로**
- **ALOHA sim demonstration dataset을 사용해서**
- **LeRobot의 ACT policy를 학습**
- 그리고 **ALOHA env에서 다시 평가**

한 것이다.

---

# 3. 왜 simulation + imitation learning을 하는가

실제 로봇을 바로 학습시키는 것은 비용과 위험이 크다.

## 3.1 실제 로봇 학습의 어려움

- 하드웨어 비용이 크다
- 시행착오가 느리다
- 잘못된 policy가 기구를 손상시킬 수 있다
- dataset 수집 비용이 크다
- reset과 반복 실험이 번거롭다

## 3.2 시뮬레이션의 장점

- 반복 실험이 빠르다
- 실패 비용이 낮다
- evaluation을 여러 번 자동으로 돌리기 쉽다
- debugging과 visualization이 쉽다

## 3.3 imitation learning의 장점

이번 실험은 reinforcement learning이 아니라 **imitation learning**이다.

즉 policy가 보상을 직접 탐색하는 것이 아니라,  
이미 존재하는 **demonstration trajectory**를 보고 그 행동을 모방하도록 학습한다.

이 방식은 다음에 유리하다.

- 초기 학습 안정성
- sparse reward 문제 회피
- 조작 태스크에서 빠른 성능 확보

ALOHA insertion처럼 정밀 조작이 필요한 task에서는 imitation learning이 매우 흔한 시작점이다.

---

# 4. 이번 실험의 실제 환경 (예시)

아래는 이번 실험에 사용한 환경이다.  
다른 사람이 따라 할 때는 **이 부분을 자신의 환경에 맞게 바꾸면 된다.**

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

이번 실험에서는 두 개의 환경을 사용했다.

### A. `aloha-venv`
용도:
- MuJoCo
- Aloha Sim viewer
- 시뮬레이터 검증

### B. `lerobot-py312`
용도:
- LeRobot 학습
- ALOHA env + dataset + ACT training + eval

이렇게 분리한 이유는:

- Aloha Sim은 먼저 Python 3.10 venv에서 안정적으로 확인했고
- LeRobot main branch는 Python 3.12 요구사항이 있어
- 두 환경을 분리하는 것이 충돌 회피에 유리했기 때문이다.

---

# 5. 오늘 한 일의 전체 흐름

오늘 실험은 크게 4개 단계로 나눌 수 있다.

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

처음에는 `nvidia-smi`가 실패했다.

예시 오류:

```bash
nvidia-smi
NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver.
```

## 6.1 원인

실제 상태는 다음과 같았다.

- GPU 자체는 시스템에서 보임
- Ubuntu 추천 드라이버도 존재
- 하지만 `lsmod | grep nvidia` 결과가 비어 있음
- `Secure Boot enabled`

이 조합은 Ubuntu에서 자주 보이는 패턴으로,  
**Secure Boot 때문에 NVIDIA kernel module이 로드되지 않는 경우**가 많다.

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

그리고 `nvidia-smi`에 아래 정보가 보여야 한다.

- Driver Version
- CUDA Version
- GPU 이름 (RTX 3070 Laptop GPU)

---

# 7. Ubuntu 재부팅 후 가장 먼저 할 확인

다음 날 다시 시작할 때는 먼저 아래를 확인하면 좋다.

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

내일 다시 켰을 때, 먼저 시뮬레이터가 정상인지 확인하고 싶다면 아래 순서로 실행한다.

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
- `egl`은 headless 또는 GPU 가속 렌더링에 유리하다.
- 환경에 따라 `glfw` 또는 자동 선택이 더 잘 맞을 수도 있지만, Ubuntu에서는 `egl`을 먼저 시도하는 것이 일반적으로 안전하다.

## 8.3 Aloha Sim 저장소로 이동

```bash
cd ~/study/workspace/physicalAI/studyALOHA/aloha_sim
```

## 8.4 Viewer 실행

```bash
python aloha_sim/viewer.py --policy=no_policy --task_name=HandOverBanana
```

설명:
- `no_policy`는 학습된 정책 없이 viewer와 task만 확인하는 모드다.
- 이 단계는 “시뮬레이터가 켜지는가?”를 보는 최소 검증이다.

---

# 9. MuJoCo 최소 테스트

MuJoCo가 정상 설치되었는지 빠르게 검증하는 최소 예시는 아래와 같다.

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

정상이라면 `MuJoCo step OK`가 출력된다.

이 테스트의 의미는:

- Python binding이 import 되는지
- XML 모델 생성이 되는지
- physics step이 정상적으로 돌아가는지

를 한 번에 확인하는 것이다.

---

# 10. LeRobot 환경 재실행 방법

LeRobot 학습/평가를 다시 시작할 때는 아래를 사용한다.

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

LeRobot 쪽에서 ALOHA env가 제대로 등록되어 있는지 확인할 때 사용한 코드는 아래와 같다.

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

오늘 확인된 env id:

```bash
gym_aloha/AlohaInsertion-v0
gym_aloha/AlohaTransferCube-v0
```

즉 ALOHA insertion task를 사용할 때는 정확히 이 이름 체계를 써야 한다.

---

# 12. 오늘 사용한 데이터셋

학습에 사용한 dataset은 다음이다.

```bash
lerobot/aloha_sim_insertion_human
```

이 dataset은 ALOHA insertion task에 대한 demonstration dataset이며,  
대략 다음 정보를 포함한다.

- top camera image
- robot state
- action trajectory

즉 policy는 이 demonstration을 보고 다음을 학습한다.

- 이미지를 해석해 물체/로봇 상태를 파악하고
- 주어진 상태에서 사람이 했던 action을 따라 하도록

이게 imitation learning의 핵심이다.

---

# 13. 오늘 사용한 policy: ACT

## 13.1 ACT란 무엇인가

ACT는 **Action Chunking Transformer**다.

핵심 아이디어는 다음과 같다.

- action을 한 step씩 예측하는 대신
- **여러 step의 action chunk를 한 번에 예측**
- image + proprioception(state)를 함께 입력으로 사용
- 시간적으로 연결된 조작 행동을 더 안정적으로 모델링

## 13.2 왜 ALOHA에 적합한가

ALOHA insertion 같은 task는:

- 한 순간의 action보다
- 일정 구간 동안의 연속적인 행동 구조가 중요하다.

예를 들어:
- peg 접근
- 자세 정렬
- 삽입 시도
- 미세 보정

은 모두 시간 연속성이 강하다.  
ACT는 이런 종류의 robot manipulation imitation learning에 잘 맞는다.

## 13.3 이번 실험의 주요 ACT 설정 예시

로그에서 확인한 대표 설정:

- `vision_backbone = resnet18`
- `chunk_size = 100`
- `n_action_steps = 100`
- `n_obs_steps = 1`
- `dim_model = 512`

즉 현재 policy는 대략:

- 현재 관측(image + state)을 보고
- 앞으로 100 step 정도의 action sequence를 예측하는 구조

로 볼 수 있다.

---

# 14. 학습 명령어 정리

아래는 오늘 실제로 사용한 명령어다.  
경로는 이번 실험 환경 기준이며, 다른 사용자는 자신의 경로로 바꾸면 된다.

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

LeRobot이 저장하는 checkpoint는 대략 다음 구조를 가진다.

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
여기에는 policy 그 자체가 들어 있다.

주요 파일:
- `model.safetensors`
- `train_config.json`

이 경로는 **eval용 policy path**로 사용한다.

## 15.2 `training_state`
여기에는 resume에 필요한 정보가 들어 있다.

예:
- optimizer state
- random state
- current step

즉:

- `pretrained_model` = 평가/추론용
- `training_state` = 학습 재개용

이다.

---

# 16. 배치(batch size)는 무엇을 의미하는가

## 16.1 정의
batch size는 **한 번의 update에 몇 개의 샘플을 동시에 사용하는지**를 뜻한다.

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

## 16.3 이번 실험에서의 의미
RTX 3070 Laptop 8GB에서는:

- 짧은 학습: `batch_size=2` 가능
- 장시간 학습 + eval 동시 진행: OOM 발생 가능
- 안정적 장기 학습: `batch_size=1`이 실용적

즉 이 환경에서는 **배치를 늘리는 것보다, 학습을 오래 돌리고 checkpoint를 잘 관리하는 쪽**이 더 중요했다.

---

# 17. step 수는 무엇을 의미하는가

## 17.1 step의 의미
여기서 `steps`는 대체로 **optimizer update 횟수**로 이해하면 된다.

즉:
- 1 step = 1회 parameter update
- 3000 step = 3000회 학습 업데이트

## 17.2 왜 dataset frame 수보다 더 많이 돌 수 있나
로그에 보면 dataset frame 수는 25000인데, 학습은 60000 step까지 진행했다.

이건 이상한 게 아니다.

이유:
- dataset을 한 번만 보고 끝나는 것이 아니라
- 여러 epoch에 걸쳐 반복해서 보기 때문이다.

즉:
- `dataset.num_frames`는 데이터 크기
- `steps`는 얼마나 오래 학습할지

를 나타낸다.

## 17.3 이번 실험의 의미
- 3k: 초기 정책이 구조를 배우기 시작
- 30k: task를 어느 정도 이해
- 60k: 상당히 깊은 상태까지 자주 도달

즉 insertion처럼 어려운 task는 보통 **몇 천 step으로 끝나지 않는다.**

---

# 18. OOM이 왜 났는가

오늘 본 OOM은 크게 두 가지였다.

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

이 옵션은 PyTorch allocator가 메모리 단편화를 완화하는 데 도움이 될 수 있다.

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
`human` render는 `pygame`이 필요할 수 있다.

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

실제로 동작을 눈으로 확인하고 싶다면 `batch_size=1`이 더 적합하다.

---

# 21. reward와 success rate 해석

평가 시 본 주요 지표는 다음과 같다.

- `avg_sum_reward`
- `avg_max_reward`
- `pc_success`

## 21.1 `pc_success`
최종 성공률이다.

예:
- `0.0` → 성공 판정을 한 번도 못 받음
- `0.2` → 10개 중 2개 성공

## 21.2 `avg_sum_reward`
에피소드 전체에서 받은 reward 총합의 평균이다.

높을수록 policy가 reward 구조상 더 좋은 행동을 하고 있다는 뜻이다.

## 21.3 `avg_max_reward`
각 episode에서 도달한 최대 reward의 평균이다.

이 값이 오르는 것은 보통 policy가 **더 깊은 성공 관련 상태**까지 들어가고 있음을 의미한다.

---

# 22. 이번 실험의 성능 추이

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

즉 현재 policy는 **완전 실패 정책이 아니라, 성공 직전 behavior를 반복적으로 만들어내는 단계**라고 보는 것이 자연스럽다.

---

# 23. 전체 프로세스에서 지금 위치는 어디인가

Robot learning 전체 흐름을 크게 나누면 다음과 같다.

1. 환경 구축
2. smoke test
3. 초기 학습
4. task structure 학습
5. 성공률 상승 구간
6. 안정화/정교화

이번 실험은 이미:

- 환경 구축 완료
- smoke test 완료
- 초기 학습 완료
- task structure 학습 진행 중

까지 왔다.

즉 지금은 **인프라 구축 단계는 끝났고, 실제 정책 성능을 끌어올리는 구간**이다.

아주 거칠게 말하면:

- 3k: 어떻게 움직일지 감을 잡는 단계
- 30k: task를 이해하기 시작한 단계
- 60k: 거의 맞는 행동을 자주 만드는 단계
- 100k 이후: success가 0을 벗어날 수도 있는 단계

---

# 24. 내일 다시 시작할 때 빠른 체크리스트

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
특히 reward가 높았던 episode의 mp4를 먼저 본다.

60k eval 기준으로는 reward가 높은 episode가 있었으므로,  
영상에서 다음을 확인하면 좋다.

- peg를 hole 근처까지 정확히 가져가는가
- 마지막 정렬에서 흔들리는가
- 삽입 직전 자세가 무너지는가
- 한쪽 팔/그리퍼 타이밍이 어긋나는가

## 추천 2. 100k까지 이어서 학습
현재 reward 추세만 보면 충분히 계속 학습할 가치가 있다.

## 추천 3. 100k 이후 다시 eval
그때 다음을 확인한다.

- `pc_success`가 0을 벗어나는가
- high reward episode 수가 늘어나는가
- avg_sum_reward가 더 상승하는가

---

# 26. 마지막 정리

오늘 한 일을 가장 포멀하게 요약하면 다음과 같다.

**Ubuntu 환경에서 NVIDIA GPU, MuJoCo, Aloha Sim, LeRobot, ALOHA insertion demonstration dataset, ACT policy를 이용한 robot imitation learning 실험 파이프라인을 구축하고, 3k → 30k → 60k 단계별 학습과 evaluation을 통해 policy의 성능 변화를 확인했다.**

그리고 현재 상태를 가장 정확하게 정리하면 다음과 같다.

**정책은 아직 최종 success 판정을 만들지 못했지만, reward 관점에서는 ALOHA insertion task의 상당 부분을 이미 학습했으며, 고득점 episode가 반복적으로 나타나는 것으로 보아 추가 학습을 계속할 가치가 충분한 상태다.**

---

# 부록 A. Git으로 프로젝트·서브모듈 관리

이 프로젝트는 **studyALOHA**(상위 저장소) 하나로 전체를 관리하고, `aloha_sim`·`lerobot`은 **서브모듈**로 원본 저장소를 가리킨다. 아래는 일상적인 Git 사용 방법 정리다.

## A.1 저장소 구도

| 대상 | 역할 | 원격 URL 예시 |
|------|------|----------------|
| **studyALOHA** (상위) | 내 실험 문서·설정·서브모듈 “버전” 관리 | `https://github.com/DownyBehind/studyALOHA.git` |
| **aloha_sim** (서브모듈) | 원본 코드만 참조, 업데이트는 원본에서 pull | `https://github.com/google-deepmind/aloha_sim.git` |
| **lerobot** (서브모듈) | 위와 동일 | `https://github.com/huggingface/lerobot.git` |

상위 저장소에는 **README, .gitignore, .gitmodules**와 **서브모듈이 가리키는 커밋**만 커밋·푸시한다. 서브모듈 폴더 안의 실제 코드는 각 원본 저장소에서 관리된다.

## A.2 프로젝트(studyALOHA) 업데이트

문서 수정, .gitignore 변경, 서브모듈이 가리키는 커밋을 바꾼 뒤 **상위 저장소만** 올리는 흐름이다.

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

`--remote`는 각 서브모듈의 원격 추적 브랜치 기준 최신 커밋으로 맞춘다. 브랜치가 `main`이 아니면 `.gitmodules` 또는 해당 서브모듈의 `branch` 설정을 확인한다.

## A.4 일상적인 Git 관리 요약

| 하고 싶은 일 | 어디서 실행 | 대략적인 순서 |
|-------------|-------------|----------------|
| README·설정만 수정해서 GitHub에 반영 | studyALOHA 루트 | `git add` → `commit` → `push` |
| 원본 aloha_sim 코드만 최신으로 맞추기 | `aloha_sim`에서 pull 후 studyALOHA 루트에서 | `cd aloha_sim` → `git pull origin main` → `cd ..` → `git add aloha_sim` → `commit` → `push` |
| 원본 lerobot 코드만 최신으로 맞추기 | `lerobot`에서 pull 후 studyALOHA 루트에서 | 위와 동일하게 `lerobot` 기준으로 |
| 다른 PC에서 프로젝트 받기 | 새 PC | `git clone --recurse-submodules <URL>` 또는 클론 후 `git submodule update --init --recursive` |
| 서브모듈 안 로컬 변경(학습 결과 등) 확인 | `aloha_sim` 또는 `lerobot` 안에서 | `git status` (상위 저장소에는 서브모듈 “커밋”만 올리면 됨) |

서브모듈 폴더 안에서 수정한 파일은 **원본 저장소에 커밋하지 않는 한** 상위 저장소 `git status`에 “modified content”로만 보인다. 학습 체크포인트·결과는 그대로 두고, 원본 코드만 따라가고 싶다면 A.3처럼 pull 후 상위에서 `git add`·`commit`·`push`만 하면 된다.