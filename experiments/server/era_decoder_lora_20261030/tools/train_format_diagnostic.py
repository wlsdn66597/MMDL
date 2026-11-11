"""Read-only format-learning diagnosis. Never creates an optimizer or trains.

Score all training targets under base, epoch-one, final adapter and merged model.
Generate a prespecified stratified subset with untouched prompts and baseline
sampling, separately at training and baseline evaluation image resolutions.
"""
import argparse
from collections import Counter
from contextlib import nullcontext
import gc
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from era_lora.common import load_config, resolve_path, write_json, canonical_json_hash, sha256_file
from era_lora.data import load_paired_records, load_processor, AssistantOnlyCollator
from era_lora.evaluate import validate_source, checkpoint_preflight, training_provenance


def selection(records):
    chosen = []
    for group in 'ABCD':
        rows = sorted((r for r in records if r['sample_id'][0] == group), key=lambda r: r['sample_id'])
        unique = list({r['visual_group_id']: r for r in reversed(rows)}.values())
        unique.sort(key=lambda r: r['sample_id'])
        indices = [round(i * (len(unique)-1)/4) for i in range(5)]
        chosen.extend(unique[i]['sample_id'] for i in indices)
    assert len(set(chosen)) == 20
    return chosen


def append(path, row):
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(row, ensure_ascii=False)+'\n')


def structure(text):
    strict = bool(re.search(r'(?ms)^Evidence:\s*\S.*?^Rule:\s*\S.*?^Application:\s*\S', text))
    flexible = bool(re.search(r'(?ims)^\s*(?:#+\s*)?(?:\*\*)?Evidence(?:\*\*)?\s*:.*?^\s*(?:#+\s*)?(?:\*\*)?Rule(?:\*\*)?\s*:.*?^\s*(?:#+\s*)?(?:\*\*)?Application(?:\*\*)?\s*:', text))
    return {'strict_era': strict, 'flexible_era': flexible,
            'starts_evidence': text.lstrip().startswith('Evidence')}


def score(cfg, records, output):
    import torch
    import torch.nn.functional as F
    from transformers import Qwen3VLForConditionalGeneration
    from peft import PeftModel
    processor = load_processor(cfg)
    collator = AssistantOnlyCollator(processor, cfg)
    run = resolve_path(cfg, cfg['training']['output_root']) / 'era'
    preflight = {r['sample_id']:r['era'] for r in json.loads((run/'encoded_pair_preflight.json').read_text())}
    # Cache CPU tensors to make the four model comparisons use identical inputs.
    batches = []
    for row in records:
        batch, audit = collator.encode(dict(row, response=row['response_era'], arm='era'))
        for key in ('prefix_token_ids_sha256', 'image_grid_thw', 'input_prompt_sha256'):
            assert audit[key] == preflight[row['sample_id']][key], (row['sample_id'], key)
        batches.append((row, batch, audit))
    base = Qwen3VLForConditionalGeneration.from_pretrained(cfg['model']['local_path'],
        dtype=torch.bfloat16, attn_implementation='sdpa', local_files_only=True).to('cuda')
    model = PeftModel.from_pretrained(base, str(run/'adapter'), is_trainable=False)
    model.load_adapter(str(run/'checkpoint-13'), adapter_name='epoch1', is_trainable=False)
    model.eval()
    summaries = {}
    for condition in ('base', 'epoch1', 'final', 'merged'):
        if condition == 'merged':
            del model, base
            gc.collect(); torch.cuda.empty_cache()
            model = Qwen3VLForConditionalGeneration.from_pretrained(str(run/'merged'),
                dtype=torch.bfloat16, attn_implementation='sdpa', local_files_only=True).to('cuda').eval()
        elif condition == 'epoch1':
            model.set_adapter('epoch1')
        else:
            model.set_adapter('default')
        context = model.disable_adapter() if condition == 'base' else nullcontext()
        results = []
        with context, torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            for i, (row, cpu, audit) in enumerate(batches):
                batch = {k:v.to('cuda') for k,v in cpu.items()}
                labels = batch.pop('labels')
                logits = model(**batch, use_cache=False).logits
                positions = (labels[0] != -100).nonzero().flatten()
                target = labels[0, positions]
                predicted = logits[0, positions-1, :].float()
                ce = F.cross_entropy(predicted, target, reduction='none')
                first = predicted[0]
                gold = int(target[0])
                probability = first.softmax(-1)
                top = torch.topk(probability, 5)
                result = {'id':row['sample_id'], 'condition':condition,
                    'assistant_mean_nll':float(ce.mean()), 'supervised_tokens':len(positions),
                    'first_target_token':processor.tokenizer.decode([gold]),
                    'first_target_token_id':gold, 'first_target_probability':float(probability[gold]),
                    'first_target_rank':int((first > first[gold]).sum())+1,
                    'first_target_is_argmax':int(first.argmax()) == gold,
                    'top5':[{'token':processor.tokenizer.decode([int(t)]),'probability':float(p)} for t,p in zip(top.indices,top.values)],
                    'prefix_sha256':audit['prefix_token_ids_sha256']}
                append(output/'scores.jsonl', result); results.append(result)
                del logits, predicted, batch, ce, first, probability, top
                if (i+1)%50 == 0:
                    print(f'Score {condition}: {i+1}/{len(records)}', flush=True)
        summaries[condition] = {
            'items':len(results),
            'mean_example_nll':sum(r['assistant_mean_nll'] for r in results)/len(results),
            'mean_first_target_probability':sum(r['first_target_probability'] for r in results)/len(results),
            'first_target_argmax_count':sum(r['first_target_is_argmax'] for r in results),
            'by_group':{g:{'mean_first_target_probability':sum(r['first_target_probability'] for r in results if r['id'][0]==g)/50,
                          'argmax_count':sum(r['first_target_is_argmax'] for r in results if r['id'][0]==g)} for g in 'ABCD'}}
        write_json(output/'score_summary.json', summaries)
        print(json.dumps({condition:summaries[condition]}), flush=True)


def generate(cfg, records, chosen, output, c):
    from PIL import Image
    from vllm import LLM, SamplingParams
    from qwen_vl_utils import process_vision_info
    processor = load_processor(cfg)
    checkpoint = resolve_path(cfg, cfg['evaluation']['checkpoints']['era'])
    rt = c['runtime']
    llm = LLM(model=str(checkpoint), dtype=rt['dtype'], quantization=None,
        tensor_parallel_size=rt['tensor_parallel_size'], max_model_len=rt['max_model_len'],
        gpu_memory_utilization=rt['gpu_memory_utilization'], max_num_seqs=rt['max_num_seqs'],
        max_num_batched_tokens=rt['max_num_batched_tokens'], enforce_eager=rt['enforce_eager'],
        limit_mm_per_prompt={'image':rt['max_images_per_prompt']}, seed=c['sampling']['seed'],
        mm_processor_kwargs={'do_resize':False})
    sample = SamplingParams(**c['sampling'], max_tokens=c['max_new_tokens'], stop_token_ids=[])
    lookup = {r['sample_id']:r for r in records}
    preflight = {r['sample_id']:r['era'] for r in json.loads((resolve_path(cfg,cfg['training']['output_root'])/'era/encoded_pair_preflight.json').read_text())}
    results = []
    for condition, vision in [('train_resolution',{k:cfg['training'][k] for k in ('min_pixels','max_pixels')}), ('eval_resolution',c['vision'])]:
        for i, sid in enumerate(chosen):
            row = lookup[sid]
            with Image.open(row['image_path']) as im:
                rgb = im.convert('RGB')
            messages = [{'role':'user','content':[dict(type='image',image=rgb,**vision),{'type':'text','text':row['input_prompt']}]}]
            rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            images, videos, kwargs = process_vision_info(messages,image_patch_size=processor.image_processor.patch_size,
                return_video_kwargs=True,return_video_metadata=True)
            assert videos is None
            kwargs['do_resize'] = False
            encoded = processor(text=[rendered],images=images,return_tensors='pt',**kwargs)
            prefix = encoded['input_ids'][0].tolist()
            assert canonical_json_hash(rendered) == preflight[sid]['rendered_prefix_sha256']
            if condition == 'train_resolution':
                assert canonical_json_hash(prefix) == preflight[sid]['prefix_token_ids_sha256']
                assert encoded['image_grid_thw'].tolist() == preflight[sid]['image_grid_thw']
            assert len(prefix)+c['max_new_tokens'] <= rt['max_model_len']
            response = llm.generate([{'prompt':rendered,'multi_modal_data':{'image':images},'mm_processor_kwargs':kwargs}],sample,use_tqdm=False)[0]
            assert list(response.prompt_token_ids) == prefix
            generated = response.outputs[0]
            final = re.findall(r'^Final answer:\s*([A-Z])\s*$',generated.text,re.M)
            result = {'id':sid,'condition':condition,'input_prompt':row['input_prompt'],
                'rendered_chat':rendered,'prefix_sha256':canonical_json_hash(prefix),
                'vision':vision,'image_sha256':row['image_sha256'],
                'processed_image_sizes_wh':[list(im.size) for im in images],
                'response':generated.text,'target':row['response_era'],
                'finish_reason':generated.finish_reason,'output_tokens':len(generated.token_ids),
                'answer_letter':row['answer_letter'],'extracted_answer':final[-1] if final else None,
                **structure(generated.text)}
            append(output/'generations.jsonl',result);results.append(result)
            summary = {cond:{'items':len(rr), 'strict_era':sum(x['strict_era'] for x in rr),
                'flexible_era':sum(x['flexible_era'] for x in rr),'starts_evidence':sum(x['starts_evidence'] for x in rr),
                'length_count':sum(x['finish_reason']=='length' for x in rr),
                'answer_matches':sum(x['extracted_answer']==x['answer_letter'] and x['finish_reason']=='stop' for x in rr)}
                for cond in ('train_resolution','eval_resolution') if (rr:=[x for x in results if x['condition']==cond])}
            write_json(output/'generation_summary.json',summary)
            print(f'Generate {condition} {i+1}/20 {sid}: ERA={result["strict_era"]}, tokens={len(generated.token_ids)}',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--mode',choices=['score','generate'],required=True)
    p.add_argument('--execute',action='store_true')
    args=p.parse_args()
    cfg=load_config(args.config)
    records,data=load_paired_records(cfg)
    chosen=selection(records)
    source,protocol,configs,hashes=validate_source(cfg)
    out=Path(args.output).resolve()
    manifest={'mode':args.mode,'all_score_items':len(records),'generation_ids':chosen,
        'selection':'Five evenly spaced sorted distinct visual groups per A/B/C/D, selected before generation',
        'prompt_changes':False,'training_started':False,'source_sha256':sha256_file(Path(__file__)),
        'dataset':data,'baseline_source':hashes,'sampling':configs[3407]['sampling'],
        'max_new_tokens':configs[3407]['max_new_tokens'],
        'limitations':'Training-set diagnostic, not held-out generalization or benchmark accuracy. Teacher-forced scores do not imply autonomous generation.'}
    if not args.execute:
        print(json.dumps(manifest,indent=2));return
    if out.exists():raise ValueError('Use a fresh output directory')
    checkpoint=resolve_path(cfg,cfg['evaluation']['checkpoints']['era'])
    manifest['checkpoint_hashes']=checkpoint_preflight(checkpoint)
    manifest['training_provenance']=training_provenance(cfg,'era',checkpoint)
    out.mkdir(parents=True)
    write_json(out/'manifest.json',manifest)
    os.environ['VLLM_ENABLE_V1_MULTIPROCESSING']='0'
    os.environ['VLLM_WORKER_MULTIPROC_METHOD']='spawn'
    os.environ.pop('VLLM_BATCH_INVARIANT',None)
    if args.mode=='score':score(cfg,records,out)
    else:generate(cfg,records,chosen,out,configs[3407])


if __name__=='__main__':main()
