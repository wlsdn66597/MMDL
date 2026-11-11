"""Deterministic next-token-only probe: no answer supplied, no training."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from era_lora.common import load_config,resolve_path,write_json,canonical_json_hash,sha256_file
from era_lora.data import load_paired_records,load_processor
from era_lora.evaluate import validate_source,checkpoint_preflight
from train_format_diagnostic import selection,append


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    p.add_argument('--execute',action='store_true')
    args=p.parse_args();cfg=load_config(args.config)
    records,data=load_paired_records(cfg);ids=selection(records)
    source,protocol,configs,hashes=validate_source(cfg);c=configs[3407]
    mm_ids=json.loads((ROOT/'configs/era_diagnostic_ids.json').read_text())
    out=Path(args.output).resolve()
    if not args.execute:
        print(json.dumps({'training_ids':ids,'mmmu_ids':mm_ids,'training':False}));return
    if out.exists():raise ValueError('Use a fresh output directory')
    checkpoint=resolve_path(cfg,cfg['evaluation']['checkpoints']['era'])
    cp_hashes=checkpoint_preflight(checkpoint)
    import torch
    from PIL import Image
    from transformers import Qwen3VLForConditionalGeneration
    from qwen_vl_utils import process_vision_info
    sys.path.insert(0,str(source/'src'))
    import prepare
    paths=prepare.cache_paths(c,cfg['evaluation']['model_cache'],cfg['evaluation']['dataset_cache'])
    datasets,mm_rows=prepare.load_data(c,paths['dataset'],offline=True)
    old={r['id']:r for r in json.loads((resolve_path(cfg,cfg['evaluation']['output_root']).parent/'diagnostic8/report.json').read_text())['rows']}
    predictions={r['id']:r for r in prepare.read_jsonl(resolve_path(cfg,cfg['evaluation']['output_root']).parent/'diagnostic8/predictions.jsonl')}
    processor=load_processor(cfg)
    items=[]
    for row in records:
        if row['sample_id'] in ids:
            with Image.open(row['image_path']) as im:rgb=im.convert('RGB')
            items.append({'id':row['sample_id'],'domain':'train','prompt':row['input_prompt'],'images':[rgb]})
    for row in mm_rows:
        if row['id'] in mm_ids:
            src=datasets[row['subject']][row['dataset_index']]
            images=[src[f'image_{i}'].convert('RGB') for i in range(1,8) if src.get(f'image_{i}') is not None]
            items.append({'id':row['id'],'domain':'mmmu','prompt':predictions[row['id']]['input_prompt'],'images':images})
    evidence_ids=processor.tokenizer.encode('Evidence',add_special_tokens=False)
    assert len(evidence_ids)==1
    evidence=evidence_ids[0]
    model=Qwen3VLForConditionalGeneration.from_pretrained(str(checkpoint),dtype=torch.bfloat16,
        attn_implementation='sdpa',local_files_only=True).to('cuda').eval()
    out.mkdir(parents=True)
    write_json(out/'manifest.json',{'items':len(items),'training_ids':ids,'mmmu_ids':mm_ids,
        'model':'merged','checkpoint_hashes':cp_hashes,'script_sha256':sha256_file(Path(__file__)),
        'training':False,'answer_supplied':False,'sampling':False,'evidence_token_id':evidence})
    results=[]
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        for condition,vision in [('train_resolution',{k:cfg['training'][k] for k in ('min_pixels','max_pixels')}),('eval_resolution',c['vision'])]:
            for i,item in enumerate(items):
                messages=[{'role':'user','content':[dict(type='image',image=im,**vision) for im in item['images']]+[{'type':'text','text':item['prompt']}]}]
                rendered=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
                if item['domain']=='mmmu':assert rendered==predictions[item['id']]['rendered_chat']
                images,videos,kwargs=process_vision_info(messages,image_patch_size=processor.image_processor.patch_size,return_video_kwargs=True,return_video_metadata=True)
                assert videos is None
                kwargs['do_resize']=False
                encoded=processor(text=[rendered],images=images,return_tensors='pt',**kwargs)
                prefix=encoded['input_ids'][0].tolist()
                batch={k:v.to('cuda') for k,v in encoded.items()}
                output=model(**batch,use_cache=False)
                first=output.logits[0,-1].float()
                prob=first.softmax(-1);top=prob.topk(5)
                result={'id':item['id'],'domain':item['domain'],'condition':condition,
                    'p_evidence':float(prob[evidence]),'evidence_rank':int((first>first[evidence]).sum())+1,
                    'evidence_argmax':int(first.argmax())==evidence,
                    'top5':[{'token':processor.tokenizer.decode([int(t)]),'probability':float(v)} for t,v in zip(top.indices,top.values)],
                    'prefix_sha256':canonical_json_hash(prefix),'input_tokens':len(prefix)}
                results.append(result);append(out/'prefix_scores.jsonl',result)
                del output,first,prob,top,batch
                print(f'Prefix {condition} {i+1}/{len(items)} {item["id"]}: {result["p_evidence"]:.5f}',flush=True)
    summary={}
    for domain in ('train','mmmu'):
        for condition in ('train_resolution','eval_resolution'):
            rows=[r for r in results if r['domain']==domain and r['condition']==condition]
            summary[f'{domain}/{condition}']={'items':len(rows),'mean_p_evidence':sum(r['p_evidence'] for r in rows)/len(rows),'argmax_count':sum(r['evidence_argmax'] for r in rows)}
    write_json(out/'summary.json',summary);print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
