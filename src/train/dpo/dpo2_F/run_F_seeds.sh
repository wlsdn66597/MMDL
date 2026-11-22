#!/usr/bin/env bash
# F 실험을 학습 seed 42, 72로 차례로 반복 (자기증류 데이터·DPO 쌍·평가 설정은 seed 3407과 같다)
for s in 42 72; do SEED=$s bash /shared_dir/MMDL_solo/repro/run_F_cd.sh || exit $?; done
