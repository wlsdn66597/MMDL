"""Export the trainer's own features with repository-relative image paths for local review."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
ROUND=HERE.parents[1]
PROJECT=ROUND.parents[1]
IMPORT_ROOTS=("results/era_a_only_round/train","results/vic_direction_round/train",
              "results/era_midstep_round/train","results/era_single_call_campaign",
              "code/training/era_decoder_lora",".runtime/tokenizers")
sys.path[:0]=[str(PROJECT/p) for p in IMPORT_ROOTS]
import train_a
from tokenizers import Tokenizer


def main():
    digest=train_a.engine.sha
    config=PROJECT/"code/training/era_decoder_lora/configs/era_4epoch_20261030.json"
    cfg=train_a.previous.load_config(config)
    rows=train_a.shared(cfg)+train_a.new_rows()
    r8_receipt=PROJECT/"results/era_r_only_round/train/data/receipt.json"
    r8=train_a.engine.read(r8_receipt)
    assert len(rows)==len({r["id"] for r in rows})==296
    assert len({r["image_sha256"] for r in rows[:248]} &
               {r["image_sha256"] for r in rows[248:]})==0
    tokenizer_path=PROJECT/".runtime/qwen_tokenizer/tokenizer.json"
    tok=Tokenizer.from_file(str(tokenizer_path))
    outputs={arm:[] for arm in train_a.ARMS}
    totals=defaultdict(Counter)
    positions=Counter();changed=[]
    for i,row in enumerate(rows):
        path=Path(row["image_path"]).resolve()
        assert path.is_relative_to(PROJECT) and path.is_file()
        relative_image=path.relative_to(PROJECT).as_posix()
        assert digest(path)==row["image_sha256"]
        positions[row["kind"]]+=1
        if i<248:
            prev=r8["samples"][i]
            assert (prev["id"],prev["group"],prev["kind"],prev["image_sha256"])==(
                row["id"],row["group"],row["kind"],row["image_sha256"])
        different=row["ordinary"]!=row["enhanced"]
        assert different==(row["kind"]=="A")
        if different:
            changed.append(row["id"])
            old=train_a.previous.TARGET.fullmatch(row["ordinary"]).groups()
            new=train_a.previous.TARGET.fullmatch(row["enhanced"]).groups()
            assert old[:2]==new[:2] and old[2]!=new[2] and old[3]==new[3]
        for arm in train_a.ARMS:
            feature=train_a.feature(row,arm)
            assert feature["image_path"]==str(path) and feature["sample_id"]==row["id"]
            feature["image_path"]=relative_image
            outputs[arm].append(feature)
            count=len(tok.encode(feature["response"],add_special_tokens=False).ids)+1
            totals["all"][arm]+=count
            totals[row["kind"]][arm]+=count
            if i<248:
                assert count==prev["target_tokens"][arm], row["id"]
    assert positions=={"shared200":200,"r5_direct":48,"A":24,"GENERAL":12,"SYMBOL":12}
    assert len(changed)==24 and totals["A"]=={"ordinary":2509,"enhanced":2547}
    files={}
    for arm,features in outputs.items():
        path=HERE/(arm+".jsonl")
        data="".join(json.dumps(r,ensure_ascii=False)+"\n" for r in features)
        path.write_bytes(data.encode("utf8"));files[arm]=digest(path)
    ordinary,enhanced=(outputs[a] for a in train_a.ARMS)
    for a,b in zip(ordinary,enhanced):
        assert all(a[k]==b[k] for k in a if k!="response")
    sources={"trainer":ROUND/"train/train_a.py","tokenizer":tokenizer_path,"config":config,
             "r8_preflight":r8_receipt,"new48":ROUND/"data/rows.jsonl",
             "common200":PROJECT/"code/training/era_decoder_lora/data/pilot200/exports/train_era.jsonl",
             "direct_source":PROJECT/"results/era_midstep_round/data/train_pairs.jsonl"}
    receipt=dict(features_per_arm=len(rows),expected_original_images=len({r["image_sha256"] for r in rows}),
                 counts=dict(positions),ordered_shared_rows_matching_r8=248,identical_inputs=296,
                 changed_only_A=changed,identical_complete_responses=272,
                 target_token_counts_per_epoch={g:dict(v) for g,v in totals.items()},
                 target_counts_method="Qwen tokenizer on exact assistant body plus one end token; all 248 common rows match the archived R8 loss-mask counts.",
                 output_sha256=files,sources_sha256={k:digest(p) for k,p in sources.items()},
                 paths_relative_to="C:/project/MMDL",image_copies=0,
                 trainer_readers_used=["train_a.shared","train_a.new_rows","train_a.feature"])
    (HERE/"receipt.json").write_bytes((json.dumps(receipt,ensure_ascii=False,indent=2)+"\n").encode("utf8"))
    print(json.dumps(receipt["target_token_counts_per_epoch"],ensure_ascii=False))


if __name__=="__main__":
    main()
