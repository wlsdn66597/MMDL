"""Verify the terminal recovery score snapshot and append the matched comparison."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def digest(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file,"sha256").hexdigest()


def save(path, content):
    temp=path.with_name(path.name+".publish.tmp")
    temp.write_text(content,encoding="utf8")
    temp.replace(path)


def publish():
    root=ROOT
    dest=root/"control_recovery/score"
    proof=read(dest/"coverage_seal.json")
    assert all(digest(dest/k)==v for k,v in proof["outputs"].items())
    x=read(dest/"summary.json")
    assert proof["n"]==x["n"] and proof["pending"]==len(x["pending_judge_ids"])
    assert x["official_full900"] is False and x["permanent_missing_id"] in x["uncommitted_ids"]
    mid=read(root/"mmmu_findings_at_12h.json")
    final=read(root/"candidate_findings.json")
    assert final["candidate"]["groups"]["all"]["right"]==511
    final["control_recovery"]=x
    final["control_recovery"]["coverage_proof_sha256"]=digest(dest/"coverage_seal.json")
    final["12h_control_vs_candidate"]=mid["candidate_vs_control"]
    save(root/"mmmu_findings.json",json.dumps(final,ensure_ascii=False,indent=2)+"\n")
    n=x["n"]
    low,high=x["full900_correct_range"]
    p=x["control_to_candidate"]
    base=x["baseline_to_control"]
    pval=f"{p['mcnemar_exact_twosided']:.6g}" if not p["pending"] else "미확정"
    section=[
        "## 같은 문항에서 정답 풀이만 학습한 대조군과 비교",
        "",
        f"대조군은 900개 중 {n}개의 단일 호출 답안이 저장됐고 잔여 미채점이 {p['left_pending']}개다. "
        f"기존 719개와 추가 {n-719}개의 응답을 같은 공식 파서·답 추출 규약으로 처리했다. "
        f"12시간 상한 도중 제출됐으나 답이 저장되지 않은 `{x['permanent_missing_id']}`의 재호출을 막았다.",
        "",
        "| 두 학습군 공통 문항 | 정답 풀이 학습 | A 대조 학습 | A 대조가 회복 / 손실 | 양쪽 정답 / 양쪽 오답 | 미확정 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key,group in (("개발",x["screen_control_to_candidate"]),("개발 외",x["outside_screen_control_to_candidate"]),("전체 공통",p)):
        left=str(group["left"]) if not group["left_pending"] else f"{group['left']}~{group['left']+group['left_pending']}"
        section.append(f"| {key} {group['n']} | {left} | {group['right']} | {group['gained']} / {group['lost']} | "
                       f"{group['both_right']} / {group['both_wrong']} | {group['pending']} |")
    section.extend([
        "",
        f"같은 {n}문항에서 LoRA 전 원본은 {base['left']}개를 맞혔다. 양쪽 학습군 모두의 판정이 확정된 "
        f"{n-p['pending']}개에서 A 대조 학습은 정답 풀이 대조군의 오답 {p['gained']}개를 회복하고 정답 {p['lost']}개를 잃었다. "
        f"공통 집합의 양측 정확 McNemar p는 {pval}이다. 같은 집합에서 완전한 ERA 형식은 "
        f"정답 풀이 학습 {x['strict_control']}/{n}, A 대조 학습 {x['strict_candidate_on_same_ids']}/{n}이었다.",
        "",
        f"대조군의 900문항 전체 정답 수가 가능한 범위는 {low}~{high}개다. 이 집계에서 900문항 대조군의 단일 공식 점수를 부여하지 않았다. "
        f"A 대조 모델의 전체 511개 및 원본 575개는 확정된 공식 점수다. [공통 문항과 미확정 범위](mmmu_findings.json)",
        "",
        f"12시간 중간 시점에는 대조군이 719개 응답했고, 392개 정답·60개 채점 대기였다. 동일 719개에서 A 대조 모델은 410개를 맞혔다. "
        f"이 시점의 자료와 최종 복구 자료를 분리해 보존했다. [12시간 시점](mmmu_findings_at_12h.json)",
    ])
    doc=root/"mmmu_findings.md"
    original=doc.read_text(encoding="utf8").split("\n## 같은 문항에서 정답 풀이만 학습한 대조군과 비교",1)[0].rstrip()
    save(doc,original+"\n\n"+"\n".join(section)+"\n")
    print(json.dumps({k:x[k] for k in ("n","control_to_candidate","full900_correct_range")}))


if __name__=="__main__":
    publish()
