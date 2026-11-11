"""Finish the report once the official runner has recorded its terminal state."""
import ctypes
import json
import os
from pathlib import Path
import select
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from campaign import FULL, read, save, sha, timestamp


def terminal_state():
    libc = ctypes.CDLL(None, use_errno=True)
    fd = libc.inotify_init1(os.O_CLOEXEC | os.O_NONBLOCK)
    if fd < 0:
        raise OSError(ctypes.get_errno(), "inotify_init")
    try:
        if libc.inotify_add_watch(fd, os.fsencode(ROOT), 0x0080 | 0x0100) < 0:
            raise OSError(ctypes.get_errno(), str(ROOT))
        while True:
            status = read(ROOT / "status.json")
            if status["state"] in {"completed", "time_limit", "failed"}:
                return status
            if not select.select([fd], [], [], max(0, FULL + 120 - time.time()))[0]:
                raise TimeoutError("No terminal campaign status was written by the deadline.")
            os.read(fd, 65536)
    finally:
        os.close(fd)


def main():
    status = terminal_state()
    subprocess.run([sys.executable, str(ROOT / "audit/mmmu_report.py")], cwd=ROOT, check=True,
                   env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
    assert (ROOT / "mmmu_findings.md").is_file() and (ROOT / "audit/mmmu_comparison.json").is_file()
    summary = ROOT / "mmmu/reverse/full/summary.json"
    doc = ROOT / "final_report.md"
    body, heading, _ = doc.read_text(encoding="utf8").partition("## MMMU 공식 확인")
    assert heading
    if status["state"] == "completed":
        info = read(summary)
        assert info["n"] == 900 and info["baseline_correct"] == 575
        assert info["correct"] == 575 + info["paired_gained"] - info["paired_lost"]
        hits = info["correct"]
        delta = hits - 575
        conclusion = "유지했다" if hits >= 575 else "유지하지 못했다"
        result = (
            f"방향 반전 실험군은 **{hits}/900 ({100*hits/900:.2f}%)**으로 LoRA 전 "
            f"**575/900 (63.89%)**을 {conclusion}. 정답 수 차이는 **{delta:+d}개**다. "
            f"원본 정답 575개 중 {575-info['paired_lost']}개를 유지하고 {info['paired_lost']}개를 잃었으며, "
            f"원본 오답 325개 중 {info['paired_gained']}개를 회복했다. "
            f"개발 90개에서 51/90(원본 64/90), 나머지 810개에서 "
            f"{info['remaining810']['correct']}/810(원본 511/810)이었다. "
            f"출력 상한에 도달한 문항은 {info['length']}개, 공식 답 추출 저지를 거친 문항은 {info['judge_count']}개였다."
        )
    else:
        result = (
            f"서버가 {status['state']} 상태에서 MMMU를 종료했다. 공식 900문항 전체 점수는 미확정이다. "
            "완료한 문항, 확정 정답 수 및 남은 범위는 별도 MMMU 분석에 기록했다."
        )
    body += heading + "\n\n" + result + "\n\n"
    body += (f"2026-11-08 18:42 KST부터 aix408-40의 RTX 4090에서 방향 반전 모델만 평가했고, "
             f"전체 실행 종료는 {status['time_kst']}다. 각 문항에 한 번씩 답하도록 하고 "
             "첫 90개의 호출·채점 기록을 전체 평가에서 재사용했다. "
             "[공식 MMMU 비교 및 처리 경로](mmmu_findings.md) · "
             "[기준 점수와 채점 정책](../mmmu_first_baseline_20261009/analysis/README.md)\n")
    if (ROOT / "audit/screen_examples.md").is_file():
        body += "\n[첫 90문항에서 정오가 바뀐 세 사례](audit/screen_examples.md)\n"
    tmp_doc = doc.with_suffix(".tmp")
    tmp_doc.write_text(body, encoding="utf8")
    tmp_doc.replace(doc)
    members = [
        "status.json", "started.json", "decision.json", "steps", "signals", "eval/review", "eval/runs",
        "eval/findings.md", "eval/case_notes.md", "train/training_result.json", "audit/gate_replay.json",
        "mmmu/reverse", "mmmu/roster.json", "audit/mmmu_comparison.json", "mmmu_findings.md", "final_report.md",
    ]
    if (ROOT / "audit/screen_examples.md").is_file():
        members.append("audit/screen_examples.md")
    path = ROOT / "terminal.tgz"
    temporary = path.with_suffix(".building")
    with tarfile.open(temporary, "w:gz", compresslevel=3) as out:
        for item in members:
            out.add(ROOT / item, arcname=item)
    temporary.replace(path)
    receipt = dict(time_kst=timestamp(), archive_sha256=sha(path), report_sha256=sha(doc),
                   decision_sha256=sha(ROOT / "decision.json"), terminal=status)
    save(ROOT / "terminal_receipt.json", receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
