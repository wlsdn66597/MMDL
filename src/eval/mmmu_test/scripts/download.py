"""평가에 필요한 모델과 데이터를 고정 revision으로 Hugging Face 캐시에 받는다 (인터넷 필요, 한 번만).

받은 뒤에는 모든 스크립트가 HF_HUB_OFFLINE=1로 캐시만 읽는다.

  python scripts/download.py                 # 전부 (약 31GB)
  python scripts/download.py base judge mmmu # 일부만

  base   Qwen/Qwen3-VL-4B-Instruct   8.3GB  평가 대상 베이스 모델 (어댑터 병합에도 필요)
  judge  Qwen/Qwen3-8B               16GB   답 형식이 어긋난 응답만 판정하는 채점 보조 모델
  mmmu   MMMU/MMMU                   3.5GB  val 900문항, test 10,500문항
  pro    MMMU/MMMU_Pro               2.8GB  10지선다, 4지선다, 그림만(vision) 각 1,730문항
"""
import json
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

S = json.loads((Path(__file__).resolve().parent.parent / "eval_settings.json").read_text(encoding="utf-8"))["settings"]
TARGETS = {
    "base": (S["model"]["repo"], S["model"]["revision"], "model"),
    "judge": (S["judge"]["repo"], S["judge"]["revision"], "model"),
    "mmmu": (S["dataset"]["repo"], S["dataset"]["revision"], "dataset"),
    "pro": ("MMMU/MMMU_Pro", "563f3e84bb3b90893083a1f039cfa13077f2302b", "dataset"),
}

for name in sys.argv[1:] or list(TARGETS):
    repo, rev, kind = TARGETS[name]
    print(f"{name}: {repo}@{rev[:8]} 받는 중", flush=True)
    print("  ->", snapshot_download(repo, revision=rev, repo_type=kind))
