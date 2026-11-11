"""Freeze disjoint synthetic charts, gold traces, and matched draft checks before model outputs."""
from pathlib import Path
from decimal import Decimal as D
import hashlib,json,random,sys
R=Path(__file__).resolve().parent
sys.path.insert(0,str(R.parent/"era_visual_pairs_20261105"))
from build_pairs import paired_chart,fmt
def save(p,o):p.write_text(json.dumps(o,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    names={"train":"Alder Birch Cedar Elm Hazel Juniper Laurel Maple Oak Pine Rowan Willow".split(),
           "holdout":"Amber Bronze Coral Cyan Gold Indigo Ivory Jade Onyx Pearl Rose Silver".split()}
    for split,n,seed in [("train",24,182011),("holdout",12,840271)]:
        rng=random.Random(seed);root=R/split; (root/"images").mkdir(parents=True,exist_ok=True)
        items=[];requests=[]
        for k in range(n):
            family=("T" if split=="train" else "H")+f"{k+1:02}"
            m=5+(k%4); denominator=D(1 if k%3==0 else 10)
            seq=sorted(rng.sample(range(10,840),m));seq[3]=seq[2]+min(seq[4]-seq[2]-1,rng.choice([1,2,3,7]))
            if seq[3]<=seq[2]:seq[3]=seq[2]+1;seq[4]+=1
            if len(set(seq))!=m or seq!=sorted(seq):raise ValueError(seq)
            lab=rng.sample(names[split],m); vals={name:D(num)/denominator for name,num in zip(lab,seq)}
            triples=tuple(lab[:3]); wrong=(lab[0],lab[1],lab[3])
            def summed(keys):return fmt(sum(vals[v] for v in keys))
            true=summed(triples);lure=summed(wrong)
            options=list(dict.fromkeys([true,lure,summed(lab[:2]),summed(lab[:4])]))
            assert len(options)==4
            rng.shuffle(options)
            order=list(reversed(lab)); other=order.copy();other[-3],other[-4]=other[-4],other[-3]
            scatter=rng.sample(lab,m)
            while scatter==order or scatter==other or set(scatter[-3:])==set(triples):scatter=rng.sample(lab,m)
            title=f"{'Station observations' if split=='train' else 'District readings'} — series {k+1}"
            lim=(max(vals.values())/10).to_integral_value(rounding="ROUND_CEILING")*10
            for j,layout in enumerate((order,other,scatter)):
                rel=f"images/{family}_{j}.png"
                paired_chart(title,{v:vals[v] for v in layout},lim,"Value").save(root/rel)
                question="What is the sum of the three smallest values in the chart?"
                base=dict(chart=family,layout=j,image=rel,image_sha256=digest(root/rel),
                          rows={v:fmt(vals[v]) for v in layout},truth_names=triples,
                          truth_vals=[fmt(vals[v]) for v in triples],truth=true,options=options)
                def put(suff,mode,q,opts,target,extra=None):
                    rid=family+"_"+suff
                    items.append(dict(base,id=rid,mode=mode,question=q,options=opts,truth=target,
                        letter="ABCD"[opts.index(target)],**(extra or {})))
                    requests.append(dict(id=rid,image=rel,image_sha256=base["image_sha256"],question=q,options=opts))
                put(f"r{j}","rank",question,options,true)
                if j==0:
                    idx=k%m;v=lab[idx];pool=[fmt(vals[v])]+[fmt(vals[x]) for x in lab if x!=v][:3];rng.shuffle(pool)
                    put("read","direct",f"What value is printed beside {v}?",pool,fmt(vals[v]),dict(direct_name=v))
                if j==1:
                    # Correct and wrong drafts on exactly the same image: no implicit answer label in prompt.
                    for supplied,tri in (("keep",triples),("replace",wrong)):
                        draft=rng.sample(list(tri),3); sv=[fmt(vals[v]) for v in draft]
                        q=question+" A provisional selection is "+", ".join(sv)+"."
                        put("d"+supplied,"draft",q,options,true,dict(draft=sv))
        rng.shuffle(requests)
        save(root/"gold.json",items);save(root/"inputs.json",requests)
    train=json.loads((R/"train/gold.json").read_text())
    test=json.loads((R/"holdout/gold.json").read_text())
    assert not ({r["image_sha256"] for r in train}&{r["image_sha256"] for r in test})
    assert len(train)==144 and len(test)==72
    save(R/"freeze.json",{p:digest(R/p) for p in ("train/gold.json","train/inputs.json","holdout/gold.json","holdout/inputs.json")})
if __name__=="__main__": main()
