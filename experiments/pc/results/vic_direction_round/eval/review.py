"""Only request new blind grades for previously unseen (item ID, finish, raw) triples."""
import argparse
import json

from run import HERE, PREVIOUS, ARMS, INPUTS, GOLD, PACKET_HASH, SCORER_HASH, SUMMARY_HASH, freeze, load_tool, read, sha

DOMAINS=("motion","operations","inspection","science")
POLICY_FIELDS=("E","R","A","consistent","answer","first_E_correct","image_corrected")

def write_lines(dst,rows):
    data="".join(json.dumps(row,ensure_ascii=False)+"\n" for row in rows).encode()
    if dst.exists():
        assert dst.read_bytes()==data,str(dst)
    else:
        dst.parent.mkdir(parents=True,exist_ok=True)
        tmp=dst.with_suffix(".preparing")
        tmp.write_bytes(data)
        tmp.replace(dst)

def historical(sc):
    assert sha(PREVIOUS/"review/summary.json")==SUMMARY_HASH and sha(PREVIOUS/"review/packet.json")==PACKET_HASH
    packets=read(PREVIOUS/"review/packet.json")["domains"]
    labels=read(PREVIOUS/"review/summary.json")["grade_proof"]
    cache={}
    for domain in DOMAINS:
        pc=PREVIOUS/"review"/(domain+".jsonl")
        lc=PREVIOUS/"review"/(domain+"_labels.jsonl")
        assert sha(pc)==packets[domain]["sha256"] and sha(lc)==labels[lc.name]
        refs={row["key"]:row for row in sc.lines(pc)}
        for grade in sc.lines(lc):
            card=refs.pop(grade["key"])
            assert card["id"]==grade["id"] and card["key"] not in cache
            cache[card["key"]]=(card,grade)
        assert not refs
    return cache

def collect():
    sc=load_tool("score",SCORER_HASH)
    assert sc.INPUTS==INPUTS and sc.GOLD==GOLD and tuple(sc.FIELDS)==POLICY_FIELDS
    records=[];cards={}
    for dataset in sc.DATASETS:
        for row,card in sc.read_runs(dataset):
            card["image"]=card["image"].replace("\\","/")
            assert cards.setdefault(card["key"],card)==card
            records.append((row,card["key"]))
    assert len(records)==345
    return sc,records,cards

def packet():
    sc,records,cards=collect()
    history=historical(sc)
    new={k:[] for k in DOMAINS}
    reused=[]
    for key,card in sorted(cards.items()):
        if key in history:
            old,label=history[key]
            assert card==old,(card["id"],"Historic card differs despite same key")
            reused.append(label)
        else:
            new[card["domain"]].append(card)
    target=HERE/"review"
    write_lines(target/"reused_labels.jsonl",reused)
    shards={}
    for domain,items in new.items():
        file=target/(domain+".jsonl")
        write_lines(file,items)
        shards[domain]=dict(n=len(items),sha256=sha(file))
    freeze(target/"packet.json",dict(n_responses=len(records),unique=len(cards),cached=len(reused),unseen=len(cards)-len(reused),
        domains=shards,cache_sha256=sha(target/"reused_labels.jsonl"),scorer_sha256=SCORER_HASH,history_summary=SUMMARY_HASH))

def score(provisional=False):
    sc,records,cards=collect()
    target=HERE/"review"
    proof=read(target/"packet.json")
    cache_file=target/"reused_labels.jsonl"
    assert sha(cache_file)==proof["cache_sha256"]
    labels={r["key"]:r for r in sc.lines(cache_file)}
    grade_files={cache_file.name:sha(cache_file)}
    for domain in DOMAINS:
        pc=target/(domain+".jsonl")
        lc=target/(domain+"_labels.jsonl")
        assert sha(pc)==proof["domains"][domain]["sha256"]
        needed={r["key"] for r in sc.lines(pc)}
        if lc.exists():
            rows=sc.lines(lc)
            assert {r["key"] for r in rows}==needed and len(needed)==len(rows)
            grade_files[lc.name]=sha(lc)
            for r in rows:
                assert r["key"] not in labels
                labels[r["key"]]=r
        elif needed and not provisional:
            raise RuntimeError(f"Missing blind judgements: {lc}")
    unknown=set(cards)-set(labels)
    if unknown:
        assert provisional
        for key in unknown:
            card=cards[key]
            letter=sc.final_letter(card["raw"],card["reference"]["options"])
            ans=False if card["finish"]=="length" else None if letter is None else letter==card["reference"]["gold"]
            labels[key]=dict.fromkeys(POLICY_FIELDS,None) | dict(key=key,id=card["id"],answer=ans,reason="Blind semantic assessment pending.")
    assert cards.keys()==labels.keys()
    rows=[sc.grade(row,cards[key],labels[key]) for row,key in records]
    results=sc.summary(rows)
    results.update(n_responses=len(records),unique_reviews=len(cards),cached_unique=proof["cached"],unreviewed_unique=len(unknown),
                   public_arm_to_journal=ARMS,grade_proof=grade_files,scorer_sha256=SCORER_HASH)
    if provisional:
        # These bounds may only rule out promotion; never promote from unfinished review.
        p=results["arms"]["preserved"]
        c=results["arms"]["ordinary"]
        b=results["arms"]["base"]
        ret=results["science_retained_bounds"]
        left=results["scene_labels"]["ordinary"]
        right=results["scene_labels"]["preserved"]
        possible={s for s in left if left[s] is not True and right[s] is not False}
        domains={r["domain"] for r in rows if r["scene"] in possible}
        results["impossible_with_remaining_reviews"]=dict(
            natural_total=p["science"]["answer"]["bounds"][1] < b["science"]["answer"]["bounds"][0],
            retained_original=ret["preserved"][1] <= ret["ordinary"][0],
            joint_8_scene_increase=p["charts"]["scenes"]["bounds"][1] <= c["charts"]["scenes"]["bounds"][0],
            multi_scene_and_domain=len(possible)<2 or len(domains)<2,
            preserved_90pct_strict_format=p["strict_era_total"]<104)
        sc.jsonout(target/"partial_summary.json",results)
    else:
        write_lines(target/"scored_items.jsonl",rows)
        freeze(target/"summary.json",results)
        reasons=[k for k,ok in results["tests"].items() if not ok]
        freeze(HERE.parent/"decision.json",dict(promote=results["gate"],reason="Development gates passed." if not reasons else
              "Unmet gates: "+", ".join(reasons),tests=results["tests"],score_sha256=sha(target/"summary.json")))
    print(json.dumps(dict(provisional=provisional,unreviewed=len(unknown),gate=results["gate"],tests=results["tests"])))

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("mode",choices=["packet","score","partial"])
    args=p.parse_args()
    if args.mode=="packet":
        packet()
    else:
        score(args.mode=="partial")
