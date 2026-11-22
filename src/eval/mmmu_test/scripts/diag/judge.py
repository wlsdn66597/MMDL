"""needs_judge 항목을 로컬 Qwen3-8B 저지로 판정 (베이스라인 저지 설정 사용).

grade.py에서 needs_judge로 남은 문항만 판정한다. 4B 생성과 동시에 GPU에 올리지 말고,
생성이 끝난 뒤 따로 실행한다.

사용 예
  python judge.py --baseline baseline_results.json --pred runs/seed1.jsonl --out runs/seed1.judge.jsonl
  python grade.py --baseline baseline_results.json --pred runs/seed1.jsonl \
                  --judge runs/seed1.judge.jsonl --out runs/seed1.graded.jsonl

베이스라인과의 차이: 베이스라인은 무효 출력 시 최대 25회 재시도했지만, temperature 0이라
재시도해도 같은 출력이 나오므로 여기서는 1회만 판정하고 무효면 오답으로 둔다.
"""
import argparse
import json
import re

from common import grade_prediction, load_baseline, read_jsonl


def options_text(rec):
    if rec["question_type"] == "multiple-choice":
        return " ".join(f"{k}. {v}" for k, v in rec["options"].items()), set(rec["options"])
    # 주관식: 공식 방식처럼 A=정답, B=Other Answers
    return f"A. {rec['reference_answer']} B. Other Answers", {"A", "B"}


def parse_judge(text, valid):
    t = text.split("</think>")[-1].strip()
    for pat in (r"your output\s*[:：]\s*\(?([A-Z])\)?", r"output\s*[:：]\s*\(?([A-Z])\)?",
                r"final answer\s*[:：]\s*\(?([A-Z])\)?"):
        ms = re.findall(pat, t, flags=re.I)
        if ms:
            return ms[-1].upper()
    m = re.fullmatch(r"\(?([A-Z])\)?\.?", t)
    if m:
        return m.group(1)
    ls = [x for x in re.findall(r"(?<![A-Za-z])([A-Z])(?![A-Za-z])", t) if x in valid | {"Z"}]
    return ls[-1] if ls else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu-mem", type=float, default=0.9)
    a = ap.parse_args()

    settings, records = load_baseline(a.baseline)
    jcfg = settings["judge"]
    preds = read_jsonl(a.pred)
    todo = []
    for p in preds:
        rec = records[p["id"]]
        if grade_prediction(p, rec)["verdict"] == "needs_judge":
            todo.append((p, rec))
    print(f"저지 대상 {len(todo)}개")
    if not todo:
        open(a.out, "w").close()
        return

    from vllm import LLM, SamplingParams
    llm = LLM(model=jcfg["repo"], revision=jcfg["revision"], dtype="bfloat16",
              max_model_len=jcfg["runtime"]["max_model_len"], gpu_memory_utilization=a.gpu_mem,
              seed=jcfg["sampling"]["seed"])
    sp = SamplingParams(temperature=0, top_p=1, top_k=-1, max_tokens=jcfg["max_new_tokens"],
                        seed=jcfg["sampling"]["seed"])
    convs, metas = [], []
    for p, rec in todo:
        opt, valid = options_text(rec)
        q = rec["question"]
        prompt = jcfg["prompt_template"].format(q, opt, p["model_response"]["raw"].split("</think>")[-1])
        convs.append([{"role": "user", "content": prompt}])
        metas.append((p["id"], valid))
    outs = llm.chat(convs, sp, chat_template_kwargs={"enable_thinking": False})
    with open(a.out, "w", encoding="utf-8") as f:
        for (rid, valid), o in zip(metas, outs):
            txt = o.outputs[0].text
            ans = parse_judge(txt, valid)
            f.write(json.dumps({"id": rid, "answer": ans, "judge_output": txt[-500:]}, ensure_ascii=False) + "\n")
    print("완료:", a.out)


if __name__ == "__main__":
    main()
