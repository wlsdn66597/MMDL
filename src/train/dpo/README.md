# DPO 학습 기록

DPO2-E 재현 모델 자료에서 확보한 [학습·가중치 병합 코드](train_dpo_d.py), [공통 처리 코드](common.py), [실제 학습 설정](dpo2e_train_info.json)을 보관한다. 같은 스크립트에서 LoRA 병합과 DeepStack 게이트 반영도 처리한다.

## DPO2-E

기반 모델은 `Qwen/Qwen3-VL-4B-Instruct`다. 기록된 실행은 ERA·오답 선호 쌍, 반복 응답 쌍 4배 반영, 보존용 데이터를 포함한 선호 쌍 2,307개로 1 epoch(145 스텝) 학습했다. 학습률은 `1e-5`, LoRA rank는 16, alpha는 32다. LLM과 시각 연결부에 LoRA를 적용하고 DeepStack 게이트와 V·I 유형 이미지 조건 선호 손실을 사용했다.

원본 학습을 재현하려면 이 스크립트가 가져오는 팀원 `train_sft.py`의 공통 함수, 원본 `pairs_E.jsonl`과 연결 이미지, 해상도 설정 `training_settings_only.json`이 필요하다.

## DPO2F-CD

병합 체크포인트 평가를 [MMMU Test 비교](../../../reports/mmmu_test/comparison.md)에 기록했다. 학습 코드·설정과 입력 선호 쌍은 팀원 원본을 확보한 뒤 여기에 추가할 수 있다.
