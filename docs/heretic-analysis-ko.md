# Heretic 전수조사 분석 정리 (한국어)

> 이 문서는 Heretic 저장소를 코드 단위로 전수조사한 결과와, 활용·수익화 방향까지
> 정리한 기록입니다. (작성일: 2026-09-22)

## 관련 GitHub / 링크

| 구분 | 주소 |
| :--- | :--- |
| 원본 저장소 (upstream) | https://github.com/p-e-w/heretic |
| 이 저장소 (fork) | https://github.com/bmshin94/heretic |
| Codeberg 미러 | https://codeberg.org/p-e-w/heretic |
| 공식 홈페이지 | https://heretic-project.org |
| 공식 튜토리얼 | https://heretic-project.org/tutorial |
| PyPI 패키지 | `heretic-llm` |
| Hugging Face 조직 | https://huggingface.co/heretic-org |
| Heretic 태그 모델 (5,000개+) | https://huggingface.co/models?other=heretic |
| Trendshift | https://trendshift.io/repositories/20538 |
| Discord | https://discord.gg/gdXc48gSyT |
| Matrix | https://matrix.to/#/#heretic:matrix.org |
| 원 논문 (Arditi et al. 2024) | https://arxiv.org/abs/2406.11717 |
| Projected abliteration (Jim Lai) | https://huggingface.co/blog/grimjim/projected-abliteration |
| Norm-preserving biprojected abliteration | https://huggingface.co/blog/grimjim/norm-preserving-biprojected-abliteration |

---

## 1. Heretic이란 무엇인가

**한 줄 정의:** 이미 학습이 끝난 트랜스포머 LLM에서 "거부(refusal) 행동"을 재학습 없이
제거하는 **완전 자동** 도구.

| 항목 | 값 |
| :--- | :--- |
| 저자 | Philipp Emanuel Weidmann (p-e-w) |
| 라이선스 | **AGPL-3.0-or-later** (수익화에 결정적 영향) |
| 버전 | `2.0.0.dev0` |
| 언어/런타임 | Python 3.10+ / PyTorch 2.2+ (gpt-oss MXFP4는 2.6+) |
| 핵심 기법 | Directional ablation (abliteration) + Optuna TPE 다목적 최적화 |
| 규모 | 54개 파일, Python 약 5,900줄 |
| 지원 아키텍처 | dense, 다수 멀티모달, 여러 MoE 변종, Qwen3.5 하이브리드 |

### 이 저장소(fork)의 상태
- `bmshin94/heretic` — upstream 포크
- PR #1 (`feat/claude-guide`) 병합 → `CLAUDE.md`에 "카리나" 개발 파트너 페르소나 정의 추가

---

## 2. 폴더 구조 전수조사

```
heretic/
├── README.md                  # 19KB, 논문급 문서 (trendshift 1위 배지)
├── pyproject.toml             # CLI 엔트리: heretic = heretic.main:main
├── uv.lock                    # 916KB, 의존성 완전 고정 (재현성 집착)
├── config.default.toml        # 8.2KB 기본 설정 (모든 튜닝 노브)
├── config.noslop.toml         # AI 문체(슬롭) 제거 설정
├── config.nohumor.toml        # 유머 제거 설정
├── config.piqa.toml           # 벤치마크 점수 기반 최적화 설정
├── src/heretic/
│   ├── main.py      (1520줄)  # 전체 오케스트레이션 + 대화형 CLI
│   ├── model.py      (867줄)  # 핵심: abliteration 로직 (LoRA 기반)
│   ├── config.py     (598줄)  # Pydantic-settings (CLI + TOML + env 통합)
│   ├── utils.py      (742줄)
│   ├── system.py     (478줄)  # 하드웨어 감지 + 배치사이즈 자동 벤치마킹
│   ├── reproduce.py  (391줄)  # 재현성 검증 (SHA256 해시 대조)
│   ├── analyzer.py   (357줄)  # 연구용 PaCMAP 시각화 + GIF 애니메이션
│   ├── plugin.py     (305줄)  # 동적 플러그인 로더
│   ├── evaluator.py  (264줄)  # 스코어러 관리 + 다목적 최적화 연결
│   ├── scorer.py      (68줄)  # Scorer 추상 베이스 클래스
│   └── scorers/
│       ├── keyword_rate.py    # 거부 키워드 탐지율
│       ├── kl_divergence.py   # 원본 모델과의 KL 발산
│       └── benchmark_score.py # lm-eval 벤치마크 (PIQA 등)
├── tests/                     # 5개 모델 × 플랫폼별 SHA256 재현성 테스트
└── .github/workflows/         # Python 3.10~3.13 매트릭스 CI (ruff, ty, build)
```

---

## 3. 내부 동작 원리

### STEP 1 — "거부 방향" 찾기
1. 무해 프롬프트 400개(`mlabonne/harmless_alpaca`) + 유해 프롬프트 400개(`mlabonne/harmful_behaviors`) 투입
2. 레이어별 **첫 토큰 잔차 벡터(residual)** 수집
3. 두 집단 평균의 차이 → 모델 내부의 "거부해라" 방향 축

```python
residual_directions = F.normalize(bad_means - good_means, p=2, dim=1)
```

4. `orthogonalize_direction = true`면 무해 방향과 직교하는 성분만 남김 (projected abliteration)

### STEP 2 — 그 방향을 발현 불가로 (`model.py:461`)
가중치를 파괴하지 않고 **rank-1 LoRA 어댑터**로 표현하는 것이 핵심 설계.

```python
# delta W = -lambda * v * (v^T W)
lora_A = (v @ W).view(1, -1)        # v^T W
lora_B = (-weight * v).view(-1, 1)  # -lambda * v
```

- 원본 가중치 보존 → 어댑터를 0으로 리셋하면 즉시 원복 (200회 실험이 가능한 이유)
- 결과물: **어댑터만(수십 MB)** 또는 **머지된 전체 모델** 선택 가능

수술 대상 컴포넌트: `attn.o_proj`, `mlp.down_proj` (MoE는 전문가별 전부)

### STEP 3 — Optuna로 200회 자동 탐색 (`main.py:639~805`)

| 파라미터 | 의미 |
| :--- | :--- |
| `direction_scope` | `global`(단일 방향) vs `per layer`(레이어별 방향) |
| `direction_index` | **float** — 정수가 아니면 인접 두 방향을 선형보간 (탐색공간 확장) |
| `max_weight` | 절제 강도 최대치 |
| `max_weight_position` | 강도가 최대인 레이어 위치 (레이어 60~100% 구간) |
| `min_weight`, `min_weight_distance` | 강도 커널의 감쇠 모양/범위 |

- 컴포넌트별로 **독립 최적화**
- MLP는 하한을 `-0.25`로 두고 0으로 클램프 → **"MLP 미개입"도 선택지** (MLP 절제가 지능을 더 손상시키기 때문, issue #202)
- 샘플러: `TPESampler(multivariate=True, n_startup_trials=60, n_ei_candidates=128)`

### STEP 4 — 두 목표 동시 최소화 (다목적)

```toml
scorers = [
  { plugin = "heretic.scorers.keyword_rate.KeywordRate",  optimization = "minimize" },
  { plugin = "heretic.scorers.kl_divergence.KLDivergence", optimization = "minimize" },
]
```

- **KeywordRate** — 응답에 37개 거부 키워드(`sorry`, `i cannot`, `as an ai`, `unethical` 등) 포함 여부. 빈 응답도 거부로 카운트하여 꼼수 최적화 방지
- **KLDivergence** — 첫 토큰 확률분포가 원본에서 얼마나 벗어났는지 = 지능 손상 측정
- 결과는 **파레토 최적 해집합**으로 제시되고 사용자가 트라이얼을 선택

### STEP 5 — 결과 활용
로컬 저장 / HF 업로드 / 채팅 테스트 / 벤치마크 실행 (대화형 메뉴)

---

## 4. 성능 (README 기준)

| 모델 | 거부 횟수 | KL 발산 (지능 손상) |
| :--- | ---: | ---: |
| google/gemma-3-12b-it (원본) | 97/100 | 0 |
| mlabonne/gemma-3-12b-it-abliterated-v2 | 3/100 | 1.04 |
| huihui-ai/gemma-3-12b-it-abliterated | 3/100 | 0.45 |
| **p-e-w/gemma-3-12b-it-heretic (Heretic, 자동)** | **3/100** | **0.16** |

같은 거부 억제 효과에 **손상은 약 1/6**. 사람 손을 전혀 안 거친 자동 결과.

---

## 5. 가장 중요한 발견: 이건 "검열 제거 도구"가 아니다

저장소에 함께 들어 있는 설정 파일들이 결정적 증거다.

| 설정 파일 | 제거 대상 | good/bad 프롬프트 구성 |
| :--- | :--- | :--- |
| `config.default.toml` | 거부 | 무해 / 유해 프롬프트 |
| `config.noslop.toml` | **AI 특유 문체(슬롭)** | "클리셰 피해서 써라" / "클리셰 범벅으로 써라" |
| `config.nohumor.toml` | **유머 감각** | 진지한 프롬프트 / 농담 데이터셋 |

`noslop`의 키워드 목록은 `ethereal`, `tapestry`, `moonlit`, `whisper`, `crimson` 등
**AI 글 냄새 표현**으로 채워져 있다. 즉 이 도구의 본질은:

> **프롬프트 두 묶음으로 정의할 수 있는 어떤 행동 축이든, 모델 내부에서 찾아내
> 그 강도를 조절하는 범용 수술 도구.** 검열 제거는 그중 한 가지 프리셋일 뿐.

---

## 6. 설치 및 사용법

### 방법 A — 가장 간단
```sh
pip install -U heretic-llm
heretic Qwen/Qwen3-4B-Instruct-2507
```
(마지막 인자를 자동으로 `--model`로 삽입하는 로직이 `main.py:196`에 있음)

### 방법 B — 개발자 권장 (uv, 의존성 완전 일치)
```sh
git clone https://github.com/bmshin94/heretic
cd heretic
uv run heretic Qwen/Qwen3-4B-Instruct-2507
```

### 방법 C — 연구 기능 포함
```sh
pip install -U 'heretic-llm[research]'
heretic <model> --plot-residuals --print-residual-geometry
```
`geom-median`, `pacmap`, `matplotlib`, `scikit-learn` 추가 → 레이어별 PaCMAP PNG + 애니메이션 GIF,
잔차 기하학 정량 분석표(코사인 유사도, L2 노름, 실루엣 계수)

### VRAM/RAM 절감 설정
```toml
quantization = "bnb_4bit"       # 4비트 양자화
batch_size = 0                  # 0 = 자동 벤치마킹
offload_outputs_to_cpu = true   # 중간 텐서 CPU 이동
max_memory = { "0" = "20GB", "cpu" = "64GB" }
```
주의: 양자화 로드 후 **머지**하려면 전체 모델을 CPU RAM에 역양자화해야 함
(대략 파라미터 수 × 3 GB. 27B → 약 80GB RAM). 이럴 땐 **어댑터만 저장**을 선택.

### 소요 시간
RTX 3090 / 기본 설정 / Qwen3-4B-Instruct-2507 기준 **약 20~30분**

### 유용한 기능
- **체크포인트 재개** — `checkpoints/*.jsonl` (Optuna JournalStorage). 중단해도 이어서 실행
- **타인 모델 평가** — `heretic --model <원본> --evaluate-model <다른 abliterated 모델>`
- **재현** — `--reproduce reproduce.json` (스키마 v3), SHA256 해시 단위 검증
- **비대화형 자동 실행** — `checkpoint_action`, `trial_index`, `model_action`, `save_directory`,
  `export_strategy`를 설정해두면 질문 없이 완주 (= 웹 서비스 래핑이 쉬운 이유)

---

## 7. 플러그인? 스킬? MCP?

| 구분 | 해당 여부 | 근거 |
| :--- | :--- | :--- |
| Claude Code 플러그인 | 아님 | `.claude-plugin/` 없음 |
| Claude 스킬 | 아님 | `SKILL.md` 없음 |
| MCP 서버 | 아님 | MCP 프로토콜 흔적 없음 |
| **독립 Python CLI 애플리케이션** | **해당** | `[project.scripts] heretic = "heretic.main:main"` |

단, **Heretic 자체가 플러그인 시스템을 보유**하고 있다 (`plugin.py`, 305줄).

```toml
scorers = [
  { plugin = "heretic.scorers.keyword_rate.KeywordRate", optimization = "minimize" },
  { plugin = "./my_scorer.py:MyKoreanRefusalScorer",     optimization = "minimize" },
]
```

`Scorer`를 상속해 `get_score()`만 구현하면 자체 평가 지표를 주입 가능.
파일 경로(`path.py:ClassName`)와 임포트 경로 모두 지원 → **수익화의 핵심 확장점**.

MCP 서버로 감싸는 것은 가능하지만 원본에는 없음(직접 구현 대상).

---

## 8. API 토큰

**추론용 API 키는 전혀 불필요.** 모든 연산이 로컬 GPU에서 수행됨.

| 상황 | 토큰 필요 여부 |
| :--- | :--- |
| 공개 모델/데이터셋 다운로드 | 불필요 |
| abliteration 연산 | 불필요 (100% 로컬) |
| 게이트 모델(Llama, Gemma 등) 다운로드 | HF 토큰(읽기) 필요 |
| HF 업로드 | HF 토큰(쓰기) 필요 |
| OpenAI / Anthropic API | 무관 |

### 토큰 보안 설계 (주목할 점)
`main.py:1097` 주석 — `huggingface_hub.login()`을 **의도적으로 쓰지 않음**.
렌탈/공유 GPU 서버에서 실행되는 경우가 많아 토큰을 디스크에 남기지 않기 위함.
또한 **설정 파일로 토큰 지정 자체를 금지**(주석: "prevent exporting the token under all
circumstances") → `reproduce.json`에 설정이 그대로 실리기 때문.

설정 방법: `export HF_TOKEN=...` 또는 실행 중 비밀 입력 프롬프트.

---

## 9. GitHub에서 유명한 이유 (7가지)

1. **객관적 성능 우위** — KL 0.16 vs 경쟁 1.04/0.45. "자동화가 전문가 수동 작업을 이김"
2. **진입장벽 파괴** — 트랜스포머 내부 지식 없이 명령어 한 줄
3. **커뮤니티 수요 정점** — r/LocalLLaMA의 오래된 최대 불만(과잉 거부)을 정면 해결. README에 Reddit 호평 3건 인용
4. **생태계 플라이휠** — 생성 모델이 HF에 `heretic` 태그로 업로드 → 5,000개+ 모델이 상시 유입 경로가 됨
5. **연구 도구로서의 신뢰성** — PaCMAP 시각화, 잔차 기하학 정량표, BibTeX 인용 정보, 선행연구 6건 크레딧
6. **엔지니어링 완성도** — Python 3.10~3.13 매트릭스 CI, ruff + ty, `uv.lock` 고정,
   SHA256 단위 재현성 테스트(5모델 × 3플랫폼), 체크포인트 재개, 배치 자동 튜닝, 4bit 양자화
7. **브랜딩** — 도발적 이름, ASCII 로고, 전용 도메인, Discord/Matrix, Codeberg 미러,
   trendshift "Repository of the Day 1위"

---

## 10. 로컬 에이전트 구축에 도움이 되는가

### 실질적 도움
1. **과잉 거부 해소 (최대 가치)** — 보안 로그 분석, 의료 데이터 파싱, 게임 악당 대사,
   취약점 리뷰 등 **정당한 업무를 모델이 거부해 파이프라인이 멈추는 문제** 제거
2. **문체/포맷 제어 (`noslop` 응용)** — AI 냄새, 불필요한 면책 문구(`disclaimer`가 이미 기본 키워드),
   장황한 서론 제거. 프롬프트로 매번 지시하는 것보다 토큰 비용·안정성 모두 유리
3. **코드베이스 자체가 교보재**

| 파일 | 재사용 가치 |
| :--- | :--- |
| `config.py` | Pydantic-settings로 CLI + TOML + 환경변수 3중 통합 |
| `plugin.py` | 동적 플러그인 로더 (파일/임포트 경로, sys.modules 캐싱) |
| `system.py` | 하드웨어 감지 + OOM 없는 배치사이즈 자동 벤치마킹 |
| `main.py` | Optuna 다목적 최적화 + 체크포인트 재개 + questionary 대화형 UX |
| `model.py` | PEFT/LoRA 직접 조작, 이종 아키텍처 방어 코드 |
| `reproduce.py` | SHA256 기반 재현성 검증 (플랫폼별 복수 허용 해시) |

### 오해하면 안 되는 것
- 에이전트 프레임워크가 아니다 (LangChain/CrewAI 대체 아님)
- 추론 속도·지능 향상 도구가 아니다 (KL 발산 = 항상 약간의 손상 발생)
- 툴 호출 능력과 무관
- 실질적으로 GPU 필수

### 반드시 고려할 트레이드오프
안전 정렬을 제거하면 **프롬프트 인젝션 저항력도 함께 약해질 수 있다.**
웹 콘텐츠나 외부 사용자 입력을 읽는 에이전트에 무검열 모델을 투입하면 공격면이 커지므로,
**에이전트 레이어에서 별도 입력 검증·샌드박싱이 필요하다.**

---

## 11. React / PHP로 만들 수 있는가

### 코어 엔진: 불가능
필요한 것 — GPU 커널(CUDA/ROCm), PyTorch급 텐서 연산, transformers/PEFT 생태계,
수십 GB 메모리 관리, bitsandbytes 양자화, Optuna급 최적화 프레임워크.
JS는 추론 전용 런타임(transformers.js/ONNX)뿐이고 PHP는 수치 연산 생태계가 없다.
(WebGPU + 초소형 모델로 데모는 이론상 가능하나 실용성 없음)

### 그 위의 제품: 충분히 가능
```
React (프론트엔드)
  · 모델 선택, 데이터셋 업로더
  · 실시간 트라이얼 대시보드 (WebSocket)
  · 파레토 프론트 인터랙티브 산점도
  · Before/After 응답 비교 뷰
  · 프리셋 마켓 (noslop, nohumor, 커스텀)
        │  REST / WebSocket
PHP(Laravel) 또는 Node (오케스트레이션)
  · 인증, 결제(Stripe), 크레딧 과금
  · 작업 큐, GPU 인스턴스 스케줄링
  · 결과 스토리지, 로그, 집계
        │  작업 투입
Python Worker (Heretic 그대로)
  · config.toml 생성 후 CLI 실행 (비대화형)
  · GPU 서버 (RunPod / vast.ai / 자체)
```

난이도 중급. Heretic이 **설정 파일 + 비대화형 모드**를 완비했고 진행상황을
`checkpoints/*.jsonl`로 읽을 수 있어 래핑이 쉽다.
단 **AGPL §13(네트워크 사용 조항)** 대응이 필수.

---

## 12. 수익화 전략

### 3대 제약조건

**(1) AGPL-3.0-or-later — 가장 중요**

| 상황 | 소스 공개 의무 |
| :--- | :--- |
| 개인 로컬 사용 | 없음 |
| 사내 전용 사용 | 없음 (배포 아님) |
| 결과 모델 가중치만 배포 | 대체로 없음 (코드 아님) |
| 코드 수정 후 배포 | **있음** |
| **웹서비스 제공 (§13)** | **있음 — 이용자에게 소스 제공** |

대응 경로: 오픈코어 전략 / 별도 프로세스 호출(법적 논쟁 여지) /
코드를 건드리지 않는 사업(컨설팅·교육·데이터셋·플러그인) / 원저자와 별도 라이선스 협상.
**상업화 시 변호사 자문 필수.**

**(2) 모델 라이선스**
Gemma(Google), Llama(Meta)는 Prohibited Use Policy / AUP상 안전장치 제거가 위반 소지.
**Apache-2.0 / MIT 계열(Qwen, Mistral)만 사용하는 것이 안전선.**

**(3) 법적·평판 리스크**
오용 책임, EU AI Act 및 국내 AI 관련 규제 동향, 결제사(PG)의 카테고리 차단,
"검열 제거 업체" 라벨의 B2B 영업 치명성.

---

### 추천 순위

#### 1위. AI 문체 정제(De-slop) 서비스 — 리스크 낮음 / 수요 확실
`config.noslop.toml`의 상업화. 검열은 전혀 건드리지 않는다.

- **타겟**: 콘텐츠 에이전시, 웹소설/웹툰 스튜디오, 마케팅, 뉴스레터, 게임 시나리오 팀
- **문제**: AI 글 특유의 문체 → AI 탐지 감점, 독자 이탈, 교정 인건비
- **해결**: 고객 브랜드 톤 샘플로 good/bad 프롬프트 쌍 구성 → 전용 모델 튜닝
- **상품화**: Lite(공개 프리셋 구독) / Pro(브랜드 톤 커스텀 튜닝) / Enterprise(사내 배포 + 유지보수)
- **강점**: 법적·평판 리스크 거의 없음, B2B 설명이 쉬움, Apache 모델로 충분,
  **성과가 키워드 출현율 before/after로 정량 측정됨**, 컨설팅 형태면 AGPL 무관

#### 2위. 한국어 스코어러 + 데이터셋 — 리스크 낮음 / 자본 거의 0
**코드에서 발견한 실제 구멍:** `keyword_markers` 37개가 **전부 영어**.

```python
keyword_markers = ["sorry", "i cannot", "as an ai", "unethical", "illegal", ...]
```

→ "죄송하지만 도와드릴 수 없습니다"는 **탐지 실패**. 한국어 모델 튜닝 시 최적화가
잘못된 방향으로 수렴한다. `good_prompts`/`bad_prompts` 기본 데이터셋도 전부 영어.
일본어·중국어·스페인어도 동일 문제.

- **상품**: ① 한국어 거부 키워드 스코어러 플러그인 ② 검수된 한국어 harmless/harmful
  프롬프트 데이터셋 ③ 한국어 슬롭 키워드 사전(번역체, "결론적으로", "~라고 할 수 있습니다" 등)
  ④ **LLM 판정 스코어러**(키워드 대신 소형 모델로 거부 판정 → 언어 무관, 정확도 향상)
- **수익**: 데이터셋 유료 라이선스, 플러그인 오픈소스 + 컨설팅, 한국어 특화 튜닝 대행
- **강점**: 진입장벽(언어+도메인 지식), AGPL 무관(별도 플러그인 파일), 자본 거의 0,
  오픈소스 공개 시 upstream 유입 + 평판 획득

#### 3위. 교육 콘텐츠 & 강의 — 리스크 없음 / 복리 효과
- 유튜브 "LLM 내부 수술" 시리즈 (PaCMAP 애니메이션 GIF가 강력한 썸네일 소재)
- 온라인 강의(Inflearn/Udemy), 유료 뉴스레터·전자책, 기업 워크숍
- 주제: abliteration 수학, 잔차 스트림 해석, LoRA 원리, Optuna 다목적 최적화, PEFT 실전,
  재현성 엔지니어링, Pydantic 설정 아키텍처 — 전부 이 코드베이스에 실물로 존재
- **포지셔닝**: "검열 제거"가 아니라 "모델 내부 이해·해석"으로

#### 4위. 로컬 LLM 도입 컨설팅 — 단가 높음 / 고객 선정 중요
과잉 거부가 실제 업무 장애인 산업: 보안/펜테스트, 의료/제약, 법률, 게임/엔터,
학술 연구, 금융 사기 분석. 이들은 **데이터 외부 유출 금지** 때문에 클라우드 API를 못 쓰므로
로컬 모델이 필수 — 시장 적합성이 높다.

- 패키지: 진단(`--evaluate-model`로 현재 거부율 측정) → 도메인 데이터셋 구축 →
  튜닝 + 벤치마크 검증(지능 손상 정량 리포트) → 온프레미스 배포 + 가드레일 → 유지보수
- **프레이밍 필수**: "검열 제거"가 아니라 **"도메인 적합성 튜닝"**, **"업무 방해 거부 해소"**
- 계약서에 오용 방지 조항, 고객 심사 절차 포함
- 컨설팅은 배포가 아니므로 AGPL 안전 (툴 수정본을 고객에 배포하면 의무 발생)

#### 5위. 튜닝 파이프라인 SaaS (오픈코어) — 상한 높음 / 난이도 높음
11장 아키텍처의 제품화. 차별 기능: 실시간 대시보드, 파레토 프론트 시각화,
Before/After 비교, 프리셋 마켓플레이스, 팀 협업, HF 원클릭 배포.
과금은 GPU 시간 크레딧 + 구독.
**AGPL 대응**: 코어는 AGPL 공개, 결제·팀·멀티테넌시·마켓을 별도 레이어로(오픈코어).
포지셔닝은 "검열 제거"가 아닌 **"모델 행동 튜닝 플랫폼"**.

#### 6위. MCP 서버 / 에이전트 툴화 — 선점 가치
Heretic을 MCP 서버로 래핑해 에이전트가 직접 튜닝 작업을 오케스트레이션.
시장은 작지만 선점 및 포트폴리오 가치가 높다.

#### 7위. HF 모델 배포 + 후원 — 보조 수단
이미 5,000개+ 모델로 경쟁 포화. 직접 수익은 낮고 **브랜딩/유입 경로**로서 가치.

---

### 권장하지 않는 것

| 아이디어 | 이유 |
| :--- | :--- |
| 무검열 모델 유료 판매 | 모델 라이선스 위반, 오용 책임, 결제사 차단, 평판 리스크 |
| 무검열 추론 API 운영 | 위 전부 + 로그 책임 + 법적 노출 최대 |
| "검열 해제" 마케팅 | 포장과 무관하게 B2B 영업 불가, 언론 리스크 |
| 독점 SaaS (소스 비공개) | AGPL §13 위반 |

---

### 실행 로드맵

| 시기 | 실행 항목 |
| :--- | :--- |
| 1개월차 | 한국어 거부 키워드 스코어러 플러그인 제작(오픈소스), 한국어 슬롭 키워드 사전, 블로그/유튜브 콘텐츠 착수 |
| 2~3개월차 | 한국어 프롬프트 데이터셋 정제(유료 라이선스 준비), upstream PR 기여로 평판 획득, De-slop 데모 모델 before/after 사례 축적 |
| 4~6개월차 | De-slop 서비스 랜딩페이지 + 파일럿 고객 2~3곳, 강의/전자책 출시, 컨설팅 문의 접수 |
| 6개월+ | React + PHP/Node 웹 플랫폼(오픈코어) 구축, MCP 서버로 에이전트 생태계 진입 |

**핵심 전략:** "검열 제거"를 버리고 **"모델 행동 정밀 제어"**를 판다.
**한국어라는 미개척 영역에 먼저 깃발을 꽂는다.**

---

## 13. 요약 체크리스트

- [x] Heretic = LLM 행동 축 자동 탐색·제어 도구 (검열 제거는 프리셋 하나)
- [x] rank-1 LoRA로 절제를 표현 → 원본 보존, 즉시 리셋, 200회 실험 가능
- [x] 거부율 + KL 발산 동시 최소화 → 파레토 최적 해집합 제시
- [x] Claude 플러그인/스킬/MCP 아님. 독립 Python CLI (단, 자체 플러그인 시스템 보유)
- [x] 추론 API 키 불필요. HF 토큰은 게이트 모델/업로드에만
- [x] React/PHP로 코어는 불가, 상위 제품(UI·과금·큐)은 가능
- [x] AGPL-3.0 §13 + 모델 라이선스가 수익화의 실질적 제약
- [x] 가장 유망: De-slop 서비스, 한국어 스코어러/데이터셋, 교육, 컨설팅
- [x] 무검열 모델 사용 시 프롬프트 인젝션 저항력 저하 → 에이전트단 입력 검증 필수
