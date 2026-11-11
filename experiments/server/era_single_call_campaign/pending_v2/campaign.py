"""Finite, completion-driven controller with a durable 12-hour hard ceiling."""
import argparse
from datetime import datetime, timedelta
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parent
EVAL=Path.home()/"mmdl_test/.venv/bin/python"
TRAIN=Path.home()/"era_decoder_lora_20261030/.venv/bin/python"
BASE=Path.home()/"mmdl-cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17"
KST=ZoneInfo("Asia/Seoul")
CAP=12*3600


def timestamp(t=None):
    return datetime.fromtimestamp(time.time() if t is None else t,KST).isoformat(timespec="seconds")


def read(p,default=None):
    return json.loads(p.read_text(encoding="utf8")) if p.exists() else default


def save(p,data):
    p.parent.mkdir(parents=True,exist_ok=True)
    f=p.with_name(p.name+".tmp")
    with f.open("w",encoding="utf8") as out:
        out.write(json.dumps(data,ensure_ascii=False,indent=2)+"\n");out.flush();os.fsync(out.fileno())
    f.replace(p)


def digest(p):
    with p.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()


def strict_era(path):
    rows=read(path)["records"]
    ok=0
    for row in rows:
        response=row["model_response"]
        raw=response["raw"].strip()
        fields=list(re.finditer(r"(?im)^\s*(?:\*\*)?(Evidence|Rule|Application|Final answer)(?:\*\*)?\s*:",raw))
        if response["finish_reason"]!="stop" or [m[1].lower() for m in fields]!=["evidence","rule","application","final answer"]:
            continue
        spans=[raw[a.end():b.start()].strip(" \r\n*") for a,b in zip(fields,fields[1:])]
        ok+=bool(all(spans) and re.fullmatch(r"(?i)(?:\*\*)?Final answer(?:\*\*)?\s*:\s*\S[^\r\n]*",raw.splitlines()[-1].strip()))
    return ok


class EndBudget(Exception):
    pass


class Controller:
    def __init__(self):
        start_file=ROOT/"started.json"
        if not start_file.exists():
            t=time.time()
            save(start_file,dict(start_time=t,start_kst=timestamp(t),deadline=t+CAP,deadline_kst=timestamp(t+CAP),max_rounds=3))
        self.clock=read(start_file)
        self.completed=read(ROOT/"events.json",[])
        self.rounds=read(ROOT/"rounds.json",[])
        self.step_name="setup"
        self.full_result=read(ROOT/"full_result.json")

    def left(self):return self.clock["deadline"]-time.time()

    def sync(self,state,detail=""):
        save(ROOT/"status.json",dict(state=state,stage=self.step_name,time_kst=timestamp(),seconds_left=max(0,int(self.left())),
             deadline_kst=self.clock["deadline_kst"],rounds_completed=len(self.rounds),detail=detail))

    def step(self,name,env,script,*args):
        self.step_name=name
        command=[str(env),"-u",str(ROOT/script),*map(str,args)]
        receipt=ROOT/"steps"/(name+".json")
        signature={"command":command,"script_sha256":digest(ROOT/script)}
        if receipt.is_file():
            if read(receipt)["identity"]!=signature:raise RuntimeError("Changed stage identity: "+name)
            return
        if self.left()<=1:raise EndBudget(name)
        self.sync("running")
        print(timestamp(),name,flush=True)
        began=time.time()
        with (ROOT/"logs"/(name+".log")).open("a") as log:
            proc=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:
                rc=proc.wait(timeout=max(0.1,self.left()-1))
            except BaseException as exc:
                try:os.killpg(proc.pid,signal.SIGTERM)
                except ProcessLookupError:pass
                try:proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    try:os.killpg(proc.pid,signal.SIGKILL)
                    except ProcessLookupError:pass
                    proc.wait()
                if isinstance(exc,subprocess.TimeoutExpired):raise EndBudget(name) from exc
                raise
        if rc:raise RuntimeError(f"{name} exit={rc}; see logs/{name}.log")
        entry=dict(stage=name,identity=signature,started=timestamp(began),ended=timestamp(),elapsed_seconds=time.time()-began)
        save(receipt,entry);self.completed.append(entry);save(ROOT/"events.json",self.completed)

    def check_round(self,n,method,arm):
        name=f"r{n}-{arm}"; ckpt=ROOT/"models"/f"r{method}"/arm/"merged"
        script="train_format.py" if method==3 else "train_round.py"
        self.step(name+"-train",TRAIN,script,"--round",method,"--arm",arm)
        self.step(name+"-merge",TRAIN,script,"--round",method,"--arm",arm,"--merge")
        self.step(name+"-screen",EVAL,"mmmu_eval.py","--checkpoint",ckpt,"--name",name,"--scope","screen")
        self.step(name+"-chart",EVAL,"chart_infer.py","--checkpoint",ckpt,"--name",name)
        chart=ROOT/"reviews"/name/"chart"
        self.step(name+"-chart-score",EVAL,"connection.py","chart-score","--spec",ROOT/"chart_spec",
                  "--base-preds",ROOT/"charts/base/predictions.jsonl","--preds",ROOT/"charts"/name/"predictions.jsonl","--out",chart)
        mm=ROOT/"reviews"/name/"screen"
        self.semantic(name,"screen",mm)
        official=read(ROOT/"benchmark"/name/"screen/summary.json")
        formatted=strict_era(ROOT/"benchmark"/name/"screen/results.json")
        cc=read(chart/"connection.json")
        return dict(name=name,round=n,method=method,arm=arm,correct=official["correct"],baseline=official["baseline_correct"],
            n=official["n"],era_format=formatted,gained=official["paired_gained"],lost=official["paired_lost"],length=official["length"],
            chart_gate=cc["connection_gate"],chart_confirmed=cc["candidate"]["joint"]["yes"],
            chart_unknown=cc["candidate"]["joint"]["unresolved"],chart_baseline_confirmed=cc["baseline"]["joint"]["yes"],
            mmmu_gate=read(mm/"connection.json")["connection_gate"],screen_seconds=read(ROOT/"steps"/(name+"-screen.json"))["elapsed_seconds"])

    def semantic(self,name,scope,out):
        self.step(name+"-"+scope+"-semantics",EVAL,"connection.py","mmmu-score",
                  "--baseline",ROOT/"benchmark"/(scope+"_baseline.json"),
                  "--candidate",ROOT/"benchmark"/name/scope/"results.json","--out",out)

    def order(self,r):
        ok=r["correct"]>=r["baseline"] and r["era_format"]>=.90*r["n"]
        if ok:return (2,r["chart_confirmed"],r["correct"],-r["chart_unknown"],r["arm"]=="candidate")
        if r["era_format"]>=.90*r["n"]:return (1,r["correct"],r["chart_confirmed"],-r["chart_unknown"],r["arm"]=="candidate")
        return (0,r["era_format"],r["correct"],r["chart_confirmed"],r["arm"]=="candidate")

    def write_report(self,condition):
        lines=["# 단일 호출 ERA 실험 중간 보고","",f"{timestamp()} KST · {condition}",
               "",f"최대 3라운드. 서버 예산 상한 {self.clock['deadline_kst']}. MMMU seed 3407 baseline 575/900.",
               "각 모델의 각 문항은 한 번만 생성한다. 공식 채점기와 내용 연결 판정은 분리한다.",
               "", "| 라운드·모델 | MMMU 개발 90 | 같은 문항 baseline | 완전 ERA | 회복 / 손실 | 차트 연결 확정·미판정 / 96 | 차트 연결 게이트 | MMMU 내용 게이트 |",
               "|---|---:|---:|---:|---:|---:|---|---|"]
        for r in self.rounds:
            for a in r["arms"]:
                lines.append(f"| {a['name']} | {a['correct']}/{a['n']} | {a['baseline']} | {a['era_format']} | {a['gained']} / {a['lost']} | {a['chart_confirmed']} · {a['chart_unknown']} | {a['chart_gate']} | {a['mmmu_gate']} |")
        if self.rounds:
            lines+=["","## 라운드별 결정",""]
            for rnd in self.rounds:lines.append(f"- 라운드 {rnd['round']} · {rnd['decision']}")
        if self.full_result:
            f=self.full_result
            lines+=["","## 전체 benchmark", "",f"{f['name']}· 전체 {f['correct']}/900, baseline {f['baseline_correct']}/900, 기존 오답 회복 {f['paired_gained']}개, 기존 정답 손실 {f['paired_lost']}개.",
                    f"정확도 기준 충족 {f['correct']>=575}. 완전 ERA {f['era_format']}/900. MMMU 근거 연결 판정 {f['connection_gate']}. 전체 성공 확정 {f['success']}.",
                    "90개를 개발에 사용했고 나머지 810개에는 후보를 먼저 고정했다. MMMU validation 자체는 과거에도 여러 번 분석했다."]
        else:
            partial=[]
            for p in sorted((ROOT/"benchmark").glob("*/predictions.jsonl")):
                partial.append(f"{p.parent.name} {len(p.read_text(encoding='utf8').splitlines())}응답")
            lines+=["","전체 900문항과 의미 검수가 끝나기 전에는 baseline 충족 여부를 확정하지 않는다."," · ".join(partial)]
        lines+=["","근거가 있는 연결 성공과 자동 판독이 어려운 응답을 따로 보관했다. MMMU 내용은 검수 범위 및 정답 참조 카드의 한계를 함께 읽어야 한다.",
                "검수 자료는 reviews/, 응답·공식 채점은 benchmark/, 실행 이력은 events.json에 있다."]
        text="\n".join(lines)+"\n"
        f=ROOT/"report.tmp";f.write_text(text,encoding="utf8");f.replace(ROOT/"report.md")

    def run(self):
        self.step("freeze-mmmu",EVAL,"mmmu_eval.py","--plan")
        self.step("freeze-charts",EVAL,"connection.py","chart-plan","--gold",Path.home()/"era_self_1k_v1/holdout/gold.json","--out",ROOT/"chart_spec")
        self.step("chart-base",EVAL,"chart_infer.py","--checkpoint",BASE,"--name","base")
        best=None
        for n in (1,2,3):
            if len(self.rounds)>=n:
                rnd=self.rounds[n-1]
            else:
                methods={r["method"] for r in self.rounds}
                if not methods:method=1
                elif len(methods)==1:
                    last=max(self.rounds[-1]["arms"],key=self.order)
                    method=3 if last["era_format"]<.9*last["n"] or last["correct"]>=last["baseline"] else 2
                else:method=next(m for m in (2,3) if m not in methods)
                arms=[self.check_round(n,method,a) for a in ("control","candidate")]
                candidates=[a for r in self.rounds for a in r["arms"]]+arms
                best=max(candidates,key=self.order)
                spent=max(a["screen_seconds"] for a in arms)
                # Remaining full-scope runtime is estimated only from already completed development data.
                reserve=best["screen_seconds"]*9*1.15+15*60
                valid_format=best["era_format"]>=.9*best["n"]
                valid_screen=valid_format and best["correct"]>=best["baseline"]
                another_screen=2*spent*1.1+20*60
                if valid_screen and best["chart_gate"]=="improved":
                    nxt="full";decision="정확도 하한과 차트 연결 선별 기준 통과. 전체 확인 후보를 동결."
                elif not valid_format and (n==3 or self.left()<another_screen):
                    nxt="stop";decision="ERA 형식 유지 조건을 충족한 모델이 없어 전체 승격을 보류하고 완료분을 보고."
                elif not valid_format:
                    nxt="next";decision="ERA 형식 하한 미달. 아직 사용하지 않은 방법으로 대조 실험을 계속하고 810문항은 사용하지 않음."
                elif n==3 or self.left()<another_screen:
                    nxt="full";decision="ERA를 유지한 모델 중 사전 규칙으로 한 후보를 고정해 남은 시간 동안 전체 검증. 미완료이면 부분 결과로 보고."
                elif valid_screen and self.left()<reserve+another_screen:
                    nxt="full";decision="정확도·ERA 형식이 모두 하한 이상이며 남은 예산을 전체 검증에 우선 배정."
                else:
                    nxt="next";decision=("스크린 정확도 하한 미달. 다음 보존 중심 비교로 진행." if best["correct"]<best["baseline"]
                        else "정확도는 하한 이상이나 연결이 미확정. 다음 독립 ERA 비교로 진행.")
                rnd=dict(round=n,method=method,arms=arms,best=best["name"],next=nxt,decision=decision)
                self.rounds.append(rnd);save(ROOT/"rounds.json",self.rounds)
                self.write_report("라운드 완료")
                f=ROOT/f"round_{n}_report.md";f.write_bytes((ROOT/"report.md").read_bytes())
            if rnd["next"]=="stop":
                self.write_report("이번 예산 블록 종료 · ERA 후보 승격 보류")
                self.sync("completed","No format-eligible nominee; MMMU900 outcome not established.")
                return
            if rnd["next"]=="full":break
        best=next(a for rnd in self.rounds for a in rnd["arms"] if a["name"]==self.rounds[-1]["best"])
        name=best["name"];ckpt=ROOT/"models"/f"r{best['method']}"/best["arm"]/"merged"
        ticket=ROOT/"nominee.json";selection=dict(name=name,checkpoint=str(ckpt),rounds=len(self.rounds),
            rule="prefer passing screen floor, then confirmed grounded chart connections, then official screen answers; otherwise screen accuracy first",
            screen=best,partial_if_budget_expires=True)
        if ticket.exists() and read(ticket)!=selection:raise RuntimeError("Nominee changed")
        save(ticket,selection)
        self.step(name+"-full",EVAL,"mmmu_eval.py","--checkpoint",ckpt,"--name",name,"--scope","full")
        out=ROOT/"reviews"/name/"full"
        self.semantic(name,"full",out)
        summary=read(ROOT/"benchmark"/name/"full/summary.json")
        gate=read(out/"connection.json")["connection_gate"]
        formatted=strict_era(ROOT/"benchmark"/name/"full/results.json")
        self.full_result=dict(summary,name=name,connection_gate=gate,era_format=formatted,
            success=summary["n"]==900 and summary["correct"]>=575 and formatted>=810 and gate=="improved")
        save(ROOT/"full_result.json",self.full_result)
        self.write_report("예산 블록 종료");self.sync("completed")


def main():
    (ROOT/"logs").mkdir(exist_ok=True)
    ctl=Controller()
    try:ctl.run()
    except EndBudget as exc:
        ctl.write_report("12시간 한도 도달 · 완료분만 보고");ctl.sync("budget_limit",str(exc))
    except Exception as exc:
        ctl.write_report("기술 오류 · 완료분만 보고");ctl.sync("failed",str(exc))
        raise


if __name__=="__main__":main()
