"""Observed vLLM requests with atomic per-item completion, safe to resume after SIGKILL."""
import argparse
import fcntl
import os
from pathlib import Path
import sys
import time
import uuid

sys.dont_write_bytecode=True
from common import ROOT,read,write,event,now,stage,records,check_record,fingerprint,sha
import input_helpers as helper  # Sets batch invariant and isolated cache paths before vLLM import.


def run(name):
    if (ROOT/'PAUSE').exists(): raise SystemExit(75)
    spec=stage(name); bundle=read(ROOT/'inputs/frozen.json'); by_id={r['id']:r for r in bundle['items']}
    assert read(ROOT/'provenance/input_parity.json')['passed']
    output_dir=ROOT/'outputs'/name;output_dir.mkdir(parents=True,exist_ok=True)
    with (output_dir/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        done=records(name)
        for row in done.values(): check_record(row,spec,by_id)
        pending=[i for i in spec['ids'] if i not in done]
        if not pending:return
        attempt=uuid.uuid4().hex
        event('worker_started',stage=name,attempt=attempt,pid=os.getpid(),pending_ids=pending,already_completed_ids=sorted(done))
        selected=dict(bundle,items=[by_id[i] for i in pending])
        requests,checks=helper.prepared_vl(selected)
        assert all(c['passed'] for c in checks)
        from vllm import LLM,SamplingParams
        from vllm.sampling_params import RequestOutputKind
        import vllm.envs as envs
        import torch
        assert bool(envs.VLLM_BATCH_INVARIANT)
        c=bundle['config'];rt=c['runtime']
        kwargs=dict(model=bundle['model_path'],dtype='bfloat16',quantization=None,tensor_parallel_size=1,seed=spec['seed'],
                    max_model_len=rt['max_model_len'],gpu_memory_utilization=.9,max_num_seqs=spec['batch'],
                    max_num_batched_tokens=4096,enforce_eager=True,enable_prefix_caching=False,disable_log_stats=False,
                    limit_mm_per_prompt={'image':5},mm_processor_cache_gb=0,
                    mm_processor_kwargs={'size':{'shortest_edge':1003520,'longest_edge':4014080},'do_resize':False})
        sampling=dict(c['sampling'],seed=spec['seed'],max_tokens=spec['max_tokens'],stop_token_ids=[])
        if spec.get('force_full'):sampling.update(ignore_eos=True,min_tokens=32768)
        t0=time.perf_counter();started=now();llm=LLM(**kwargs)
        fp=fingerprint(spec)
        write(ROOT/'provenance/engines'/f'{name}_{attempt}.json',dict(stage=name,attempt=attempt,pid=os.getpid(),started_at=started,
              engine_init_seconds=time.perf_counter()-t0,llm_kwargs=kwargs,sampling=sampling,recognized_batch_invariant=bool(envs.VLLM_BATCH_INVARIANT),
              gpu=torch.cuda.get_device_name(0),capability=list(torch.cuda.get_device_capability()),torch_cuda=torch.version.cuda,
              engine_config=repr(llm.llm_engine.vllm_config),fingerprint=fp,runner_sha256=sha(ROOT/'runner.py'),
              output_observation='Engine add_request/step, RequestOutputKind.CUMULATIVE. api_smoke verifies equality to previous LLM.generate outputs.'))
        for offset in range(0,len(pending),spec['batch']):
            if (ROOT/'PAUSE').exists():raise SystemExit(75)
            group=pending[offset:offset+spec['batch']]
            active={i:dict(id=i,generated_tokens=0,finished=False) for i in group}
            began=time.perf_counter();last_progress=0
            event('batch_started',stage=name,attempt=attempt,ids=group,completed=len(done))
            for i in group:
                sample=SamplingParams(**sampling,output_kind=RequestOutputKind.CUMULATIVE)
                llm.llm_engine.add_request(request_id=i,prompt=requests[i],params=sample)
            while llm.llm_engine.has_unfinished_requests():
                if (ROOT/'PAUSE').exists():
                    event('user_pause_observed',stage=name,attempt=attempt,completed_ids=sorted(done),inflight=list(active.values()))
                    raise SystemExit(75)
                outputs=llm.llm_engine.step()
                for result in outputs:
                    i=result.request_id
                    assert i in active, (i,group)
                    o=result.outputs[0]
                    active[i]=dict(id=i,generated_tokens=len(o.token_ids),finished=result.finished)
                    if result.finished:
                        assert list(result.prompt_token_ids)==by_id[i]['prompt_token_ids'],i
                        row=dict(id=i,stage=name,seed=spec['seed'],max_tokens=spec['max_tokens'],fingerprint=fp,
                                 token_ids=list(o.token_ids),text=o.text,finish_reason=o.finish_reason,stop_reason=o.stop_reason,
                                 input_tokens=len(result.prompt_token_ids),output_tokens=len(o.token_ids),
                                 batch_ids=group,elapsed_since_batch_seconds=time.perf_counter()-began,created_at=now(),attempt=attempt)
                        check_record(row,spec,by_id)
                        dest=output_dir/f'{i}.json'
                        assert not dest.exists(), 'Never overwrite a committed result'
                        write(dest,row);done[i]=row
                        event('item_committed',stage=name,attempt=attempt,id=i,output_tokens=row['output_tokens'],completed=len(done),expected=len(spec['ids']))
                        print(name,len(done),'/',len(spec['ids']),i,row['output_tokens'],row['finish_reason'],flush=True)
                if time.perf_counter()-last_progress>=1 or any(r.finished for r in outputs):
                    write(ROOT/'progress'/f'{name}.json',dict(updated_at=now(),pid=os.getpid(),stage=name,attempt=attempt,
                          completed=len(done),expected=len(spec['ids']),active=list(active.values())))
                    last_progress=time.perf_counter()
            assert all(i in done for i in group)
        event('worker_completed',stage=name,attempt=attempt,pid=os.getpid(),completed=len(done))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',required=True);a=p.parse_args();run(a.stage)
