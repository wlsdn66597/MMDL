"""Execute the conditional held-out comparison and its predeclared MMMU branch."""
import sys
from run import ROOT, jobs


def signal(stage, files):
    if stage == "confirm":
        files += ["confirm_run.py","confirm/eval/confirmation.py","confirm/eval/spec/inputs.jsonl",
                  "confirm/eval/spec/gold.jsonl","confirm/eval/spec/receipt.json"]
    else:
        files += [f"confirm/eval/review/manual_{i}.jsonl" for i in range(4)]
    target = ROOT/"signals"/(stage+".json")
    hashes = {name:jobs.sha(ROOT/name) for name in files}
    if target.exists():
        assert jobs.read(target)["hashes"]==hashes
    else:
        jobs.write(target,dict(hashes=hashes,time_kst=jobs.stamp()))


def main():
    assert jobs.read(ROOT/"eval/review/summary.json")["advance"] is True
    deadline = jobs.read(ROOT/"started.json")["deadline"]

    def stage(name, env, script, *args):
        jobs.step("confirm-"+name,env,script,args,deadline)

    infer = ROOT/"infer.py"
    scale = ROOT/"scale.py"
    stage("base",jobs.EVAL,infer,"base","--stage","confirm")
    stage("enhanced100",jobs.EVAL,infer,"enhanced100","--stage","confirm")
    for arm in ("ordinary","enhanced"):
        stage("scale-"+arm,jobs.TRAIN,scale,arm)
        stage("infer-"+arm,jobs.EVAL,infer,arm,"--stage","confirm")
        stage("retire-"+arm,jobs.TRAIN,scale,arm,"--retire","--stage","confirm")
    score = ROOT/"confirm/eval/confirmation.py"
    for mode in ("packet","judge","audit"):
        stage(mode,jobs.EVAL,score,mode)
    jobs.waiting("confirm_manual",deadline)
    for mode in ("merge","score"):
        stage(mode,jobs.EVAL,score,mode)
    result = jobs.read(ROOT/"confirm/decision.json")
    assert type(result["promote"]) is bool
    if result["promote"]:
        assert jobs.sha(jobs.OFFICIAL) == "ce72d13a3ac569fa2e67aafab3d54d5d2665f6960d356ea7212be0dfb54e3c3e"
        # Full MMMU reuses the first 90 items and reserves this one checkpoint.
        stage("rebuild-nominee",jobs.TRAIN,scale,"enhanced")
        for scope in ("screen","full"):
            stage("mmmu-"+scope,jobs.EVAL,jobs.OFFICIAL,"--scope",scope,"--name","enhanced50",
                  "--output-root",ROOT/"mmmu","--checkpoint",ROOT/"models/enhanced/merged")
        stage("archive-nominee",jobs.TRAIN,scale,"enhanced","--retire","--stage","confirm")


if __name__=="__main__":
    if len(sys.argv)>1:
        assert len(sys.argv)==3 and sys.argv[1]=="--seal" and sys.argv[2] in ("confirm","confirm_manual")
        signal(sys.argv[2],[])
    else:
        main()
