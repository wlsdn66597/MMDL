# 단일 호출 R1–R3, A 대조, R5–R9 학습과 평가

`server/`는 aix408-40 홈 디렉터리 기준의 실험 코드와 설정입니다. `pc/`는 MMDL 프로젝트 기준의 분석·검증 코드입니다. 새 환경에서 실행할 때는 데이터·설정에 기록된 경로를 복원 위치에 맞추고, 같은 실험 이름의 결과 묶음을 원래 상대 경로에 풉니다. 서로 다른 실험의 코드를 참조하는 스크립트는 해당 브랜치의 `server/` 폴더를 같은 위치에 합쳐 사용합니다.

## 관련 실험 코드

- [era-pilot-lora](https://github.com/wlsdn66597/MMDL/tree/sunjae/era-pilot-lora-20261111/experiments)
- [era-strength-last-block](https://github.com/wlsdn66597/MMDL/tree/sunjae/era-strength-last-block-20261111/experiments)
- [era-training-rounds](https://github.com/wlsdn66597/MMDL/tree/sunjae/era-training-rounds-20261111/experiments)
- [mmmu-evaluation](https://github.com/wlsdn66597/MMDL/tree/sunjae/mmmu-evaluation-20261111/experiments)

발표 모델의 기반 버전, 선행 어댑터, 병합 순서와 적용 강도는 [모델 저장소](https://huggingface.co/JsunKim/Qwen4B_ERA/tree/main/presentation-20261111)에 있습니다. PC 백업의 실험목록.csv에는 실험별 고정 커밋을 기록했습니다.
