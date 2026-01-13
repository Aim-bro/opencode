너는 시니어 데이터 분석가 + 실무형 ML 엔지니어 역할로 행동해.
목표: Windows 로컬에서 opencode를 코드로 호출해 Kaggle Spaceship Titanic 파이프라인 구축.
모델: 기본 opencode/glm-4.7-free, 무료 모델 한계는 프롬프트 분해로 보완.

현재 상태(이미 완료):
- Repo 루트: C:\Users\wnsgu\Josh\Josh\opencode
- 프로젝트 폴더: C:\Users\wnsgu\Josh\Josh\opencode\spaceship-titanic
- kaggle.json 생성 위치: C:\Users\wnsgu\.kaggle\kaggle.json
- 데이터 다운로드 완료: C:\Users\wnsgu\Josh\Josh\opencode\spaceship-titanic\data\raw
  - train.csv, test.csv, sample_submission.csv 존재
- 스크립트:
  - C:\Users\wnsgu\Josh\Josh\opencode\spaceship-titanic\scripts\kaggle_auth.ps1
  - C:\Users\wnsgu\Josh\Josh\opencode\spaceship-titanic\scripts\kaggle_download.ps1

요청:
1) 다음 작업부터 진행해줘:
   - src/titanic/load.py (train.csv/test.csv 로더)
   - src/opencode_runs/run_prompt.py (subprocess로 opencode run 호출)
   - prompts/에 5개 프롬프트 (EDA/결측/피처/모델링/CV/리포트)
   - notebooks/01_eda.ipynb 최소 EDA 흐름
   - docs/decisions.md 템플릿
   - .gitignore에 **/.venv/, **/data/raw/, **/*.env, **/outputs/large 반영

작업 지침:
- Chain-of-Thought은 내부적으로만 사용하고, 출력은 핵심 논거/판단 근거 요약/선택·기각 이유를 구조화 bullet로 제공.
- 모든 LLM 제안은 가설로 취급하고 검증 가능한 형태로 변환.
