"""Independent audit of downloaded evidence; never imports campaign code."""
from pathlib import Path
from collections import Counter
import json, hashlib, datetime as dt, re, html
from PIL import Image

R = Path('C:/project/MMDL/results/mmmu_batch_invariant_extended_20261014')
def read(p): return json.loads((R/p).read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256((R/p).read_bytes()).hexdigest()
def objsha(x): return hashlib.sha256(json.dumps(x, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
def time(s): return dt.datetime.fromisoformat(s)
def kst(s): return time(s).astimezone(dt.timezone(dt.timedelta(hours=9))).strftime('%Y-%m-%d %H:%M:%S')
P=read('plan.json'); B=read('inputs/frozen.json'); S=read('status.json')
assert S['state']=='complete'
assert sha('inputs/frozen.json')==P['frozen_inputs_sha256']
assert sha('provenance/pip_freeze.txt')==P['environment_freeze_sha256']
assert sha('provenance/checkpoint_verification.json')==P['checkpoint_verification_sha256']
items={x['id']:x for x in B['items']}; assert len(items)==len(B['items'])==60
subjects=Counter(x['subject'] for x in items.values()); assert len(subjects)==30 and set(subjects.values())=={2}
subset=B['seed_subset_ids']; assert len(subset)==len(set(subset))==12 and set(subset)<=set(items)
assert {len(items[i]['image_paths']) for i in subset}=={1,2,3,4,5}
image_count=0
for item in items.values():
    for path, expected in zip(item['image_paths'],item['rgb_sha256'],strict=True):
        with Image.open(R/path) as im: assert hashlib.sha256(im.convert('RGB').tobytes()).hexdigest()==expected
        image_count+=1
parity=read('provenance/input_parity.json')
assert parity['passed'] and parity['recognized_batch_invariant']
assert len(parity['input_checks'])==60 and all(x['passed'] for x in parity['input_checks'])
assert len(parity['judge_input_checks'])==4 and all(x['passed'] for x in parity['judge_input_checks'])
specs={s['name']:s for s in P['stages']}; outputs={}; fps={}; comparisons=[]; stats=[]; lengths=[]
fields=['token_ids','text','finish_reason','stop_reason','input_tokens']
for name,s in specs.items():
    fp=objsha(dict(spec=s,plan_sha256=sha('plan.json'),runner_sha256=sha('runner.py'),helpers_sha256=sha('input_helpers.py')));fps[name]=fp
    rows={}
    for f in (R/'outputs'/name).glob('*.json'):
        x=json.loads(f.read_text(encoding='utf8'));assert f.stem==x['id'] and x['id'] not in rows;rows[x['id']]=x
    outputs[name]=rows
    if s.get('conditional'):
        assert not rows and (R/'skipped'/f'{name}.json').exists();continue
    assert set(rows)==set(s['ids']) and len(rows)==len(s['ids'])
    for i,x in rows.items():
        assert x['stage']==name and x['seed']==s['seed'] and x['max_tokens']==s['max_tokens']
        assert x['fingerprint']==fp
        assert x['output_tokens']==len(x['token_ids'])<=s['max_tokens']
        assert x['input_tokens']==len(items[i]['prompt_token_ids'])
        assert x['finish_reason'] in ['stop','length']
        assert x['finish_reason']!='length' or len(x['token_ids'])==s['max_tokens']
        assert i in x['batch_ids'] and len(x['batch_ids'])<=s['batch']
        if x['output_tokens']==32768:
            assert x['finish_reason']=='length';lengths.append(dict(stage=name,id=i,seed=s['seed']))
    stats.append(dict(stage=name,seed=s['seed'],outputs=len(rows),batch=s['batch'],cap=s['max_tokens'],max_output_tokens=max(x['output_tokens'] for x in rows.values()),length_32768=sum(x['output_tokens']==32768 for x in rows.values()),finish_reasons=dict(Counter(x['finish_reason'] for x in rows.values()))))
assert sum(len(x) for x in outputs.values())==234
assert len(list((R/'outputs').glob('*/*.json')))==234
assert len([x for x in lengths if x['stage']=='seed3407_reference'])>=1
assert specs['seed3407_batch4']['ids']==list(reversed(specs['seed3407_reference']['ids']))
for seed in [42,123,2026,9876]:
    assert specs[f'seed{seed}_reference']['ids']==subset
    assert specs[f'seed{seed}_batch4']['ids']==subset[::2]+subset[1::2]
parent={x['id']:x for x in map(json.loads,(R/'provenance/parent_evidence/outputs/vl_reference.jsonl').read_text(encoding='utf-8').splitlines())}
for name,s in specs.items():
    if not s['reference']:continue
    ref=parent if s['reference']=='parent_pilot' else outputs[s['reference']]
    for i,x in outputs[name].items():
        diff=[f for f in fields if x[f]!=ref[i][f]]
        comparisons.append(dict(stage=name,id=i,fields=fields,differences=diff,passed=not diff))
assert len(comparisons)==126 and all(x['passed'] for x in comparisons)
events=[json.loads(l) for l in (R/'events.jsonl').read_text(encoding='utf-8').splitlines()]
starts={x['attempt']:x for x in events if x['event']=='worker_started'}
commits=[x for x in events if x['event']=='item_committed'];assert len(commits)==234
assert len({(x['stage'],x['id']) for x in commits})==234
engines={}
for f in (R/'provenance/engines').glob('*.json'):
    e=json.loads(f.read_text(encoding='utf-8'));n=e['stage'];s=specs[n];kw=e['llm_kwargs'];sp=e['sampling']
    assert e['recognized_batch_invariant'] and e['fingerprint']==fps[n] and e['runner_sha256']==sha('runner.py')
    assert kw['dtype']=='bfloat16' and kw['seed']==sp['seed']==s['seed']
    assert kw['enforce_eager'] and kw['enable_prefix_caching'] is False and kw['mm_processor_cache_gb']==0
    assert kw['model']==B['model_path'] and kw['max_num_seqs']==s['batch']
    assert kw['max_model_len']==64000 and kw['max_num_batched_tokens']==4096
    assert kw['mm_processor_kwargs']=={'size':{'shortest_edge':1003520,'longest_edge':4014080},'do_resize':False}
    for key,val in B['config']['sampling'].items():
        if key!='seed':assert sp[key]==val
    assert sp['max_tokens']==s['max_tokens'] and not sp.get('ignore_eos',False) and sp.get('min_tokens',0)==0
    assert e['torch_cuda']=='13.0' and e['gpu']=='NVIDIA GeForce RTX 4090'
    engines[e['attempt']]=e
assert set(engines)==set(starts)
for name,rows in outputs.items():
    for i,x in rows.items():
        assert x['attempt'] in engines and i in starts[x['attempt']]['pending_ids']
        assert any(c['stage']==name and c['id']==i and c['attempt']==x['attempt'] and c['output_tokens']==x['output_tokens'] for c in commits)
helper=(R/'input_helpers.py').read_text(encoding='utf-8')
assert "os.environ['VLLM_BATCH_INVARIANT'] = '1'" in helper
assert "os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'" in helper
batch_evidence={}
for name,s in specs.items():
    if not outputs[name]:continue
    log=(R/'logs'/f'{name}.log').read_text(encoding='utf8')
    assert 'Using FLASH_ATTN attention backend' in log
    assert 'VLLM_ENABLE_V1_MULTIPROCESSING is set to False' in log
    running=sorted(set(map(int,re.findall(r'Running: (\d+) reqs',log))))
    batch_evidence[name]=running
    if s['batch']==4:assert 4 in running and 2 in running
fault=read('fault_injected.json'); rec=outputs['actual_resume_seed3407']; retained=fault['committed_sha256']
assert len(retained)>=2 and fault['signal']=='SIGKILL'
assert any(x['generated_tokens']>=128 and not x['finished'] for x in fault['observed_inflight'])
assert fault['process_cmdline'][-2:]==['--stage','actual_resume_seed3407']
assert fault['process_cmdline'][-3].endswith('/runner.py')
for i,h in retained.items():assert sha(f'outputs/actual_resume_seed3407/{i}.json')==h and rec[i]['attempt']==fault['attempt']
new_attempts={x['attempt'] for i,x in rec.items() if i not in retained}
assert len(new_attempts)==1 and fault['attempt'] not in new_attempts
new_attempt=next(iter(new_attempts));assert set(starts[new_attempt]['already_completed_ids'])==set(retained)
assert set(starts[new_attempt]['pending_ids'])==set(rec)-set(retained)
assert any(e['event']=='worker_exited' and e['pid']==fault['pid'] and e['returncode']==-9 and e['planned_injection'] for e in events)
assert len([e for e in events if e['event']=='planned_inflight_sigkill'])==1
assert not any(e['event'].startswith('user_pause') for e in events)
unplanned=[]
for f in (R/'interruption_history').glob('*/evidence.json'):
    ev=json.loads(f.read_text(encoding='utf-8'));assert not ev['planned_fault'] and not ev['user_pause']
    assert all(sha(p)==h for p,h in ev['completed_sha256'].items())
    old=ev['previous_progress'];new=[v for v in starts.values() if v['stage']==old['stage'] and time(v['time'])>time(old['updated_at'])]
    assert len(new)==1 and new[0]['attempt']!=old['attempt']
    old_ids={Path(p).stem for p in ev['completed_sha256'] if p.startswith('outputs/'+old['stage']+'/')}
    assert set(new[0]['already_completed_ids'])==old_ids
    assert set(new[0]['pending_ids'])==set(specs[old['stage']]['ids'])-old_ids
    unplanned.append(dict(evidence=str(f.relative_to(R)),cause=ev['cause'],preserved_outputs=len(ev['completed_sha256']),previous_progress=old,new_attempt=new[0]['attempt'],regenerated_ids=new[0]['pending_ids']))
assert len(unplanned)==1
env=read('provenance/environment_preservation.json');assert env['passed'] and all(env[k] for k in ['original_code_equal','original_packages_equal','new_environment_packages_equal'])
chk=read('provenance/checkpoint_final_verification.json');assert chk['passed'] and len(chk['checks'])==9
for c in chk['checks']:assert c['passed'] and c['actual']==c['expected']
versions=read('provenance/final_versions.json')['versions'];assert versions==parity['versions']
assert B['config']['judge']['sampling']['seed']==3407
judge_ref={x['id']:x for x in map(json.loads,(R/'provenance/parent_evidence/outputs/judge_reference.jsonl').read_text(encoding='utf-8').splitlines())}
for variant in ['judge_four_reverse','judge_repeat']:
    rows=list(map(json.loads,(R/f'provenance/parent_evidence/outputs/{variant}.jsonl').read_text(encoding='utf-8').splitlines()))
    assert len(rows)==4
    for x in rows:assert all(x[k]==judge_ref[x['id']][k] for k in fields)
durations=[]
for a,start in starts.items():
    end=next((x['time'] for x in events if x['event']=='worker_exited' and x['pid']==start['pid']),None)
    observed_only=False
    if end is None:
        old=unplanned[0]['previous_progress'];assert a==old['attempt'];end=old['updated_at'];observed_only=True
    durations.append(dict(stage=start['stage'],attempt=a,start=start['time'],end=end,seconds=(time(end)-time(start['time'])).total_seconds(),end_is_last_observation=observed_only))
elapsed=(time(S['updated_at'])-time(events[0]['time'])).total_seconds()
summary=read('summary.json');assert summary['complete'] and summary['passed'] and summary['comparison_count']==126 and summary['differences']==0
transfer=read('transfer_verification.json');assert transfer['passed']
result=dict(passed=True,checked_at=dt.datetime.now(dt.timezone.utc).isoformat(),method='Independent local parser, hashes, equality checks and event/log audit; campaign code not imported',output_count=234,comparison_count=126,differences=0,subject_counts=dict(subjects),image_count=image_count,stage_statistics=stats,comparisons=comparisons,natural_32768=lengths,forced_outputs=0,max_output_tokens=max(x['output_tokens'] for rows in outputs.values() for x in rows.values()),planned_recovery=dict(passed=True,preserved_ids=list(retained),regenerated_ids=starts[new_attempt]['pending_ids'],inflight=fault['observed_inflight'],old_attempt=fault['attempt'],new_attempt=new_attempt),unplanned_interruptions=unplanned,engine_attempt_count=len(engines),observed_running_requests=batch_evidence,versions=versions,environment_preservation=env,checkpoint_verification=chk,transfer=transfer,wall_seconds_including_outage=elapsed,observed_worker_seconds=sum(x['seconds'] for x in durations),worker_intervals=durations,completed_at=S['updated_at'],audit_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(R/'independent_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
esc=html.escape
rows=''.join(f"<tr><td>{esc(s['stage'])}</td><td>{s['seed']}</td><td>{s['outputs']}</td><td>{s['batch']}</td><td>{s['max_output_tokens']:,}</td><td>{s['length_32768']}</td></tr>" for s in stats)
interval_rows=''.join(f"<tr><td>{esc(x['stage'])}</td><td>{kst(x['start'])}</td><td>{kst(x['end'])}</td><td>{x['seconds']/60:.1f}</td><td>{'중단 전 마지막 관측' if x['end_is_last_observation'] else '프로세스 종료 이벤트'}</td></tr>" for x in durations)
natural_ref=sum(x['stage']=='seed3407_reference' for x in lengths)
report=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>MMMU 본평가 전 추가 검증</title>
<style>body{{font-family:system-ui,'Malgun Gothic',sans-serif;max-width:1120px;margin:40px auto;padding:0 24px;color:#172b3a;line-height:1.75;background:#f6f8fb}}h1,h2{{line-height:1.3}}section{{background:white;padding:24px;margin:20px 0;border-radius:12px;border:1px solid #dce3eb}}.pass{{color:#087347;font-size:22px;font-weight:700}}table{{width:100%;border-collapse:collapse;font-size:13px}}td,th{{text-align:left;border-bottom:1px solid #e0e6ee;padding:8px;overflow-wrap:anywhere}}th{{background:#edf3f8}}code{{overflow-wrap:anywhere}}.scroll{{overflow-x:auto}}a{{color:#0759a0}}</style>
<h1>MMMU 본평가 전 추가 검증</h1><p>실행 종료: {kst(S['updated_at'])} KST · 로컬 독립 검증 및 증거 회수 완료</p>
<section><p class="pass">234개 출력 · 126개 비교 완전 일치 · 차이 0건</p><p>같은 추론 seed에서 token_ids, text, finish_reason, stop_reason, input_tokens를 모두 비교했다. 요약 파일과 별개로 전체 원자적 출력 JSON의 ID, seed, 소스·설정 fingerprint, 토큰 수, 종료 사유와 이벤트를 재검사했다.</p><p>60개 고유 문항은 30과목당 2문항이다. seed 3407은 60문항의 단독 생성과 배치 4 역순을 비교했다. 다른 seed 42·123·2026·9876은 이미지 1~5개를 포함하는 동일한 12문항에서 단독 생성과 배치 4 동반 문항 변경을 비교했다. API smoke는 12문항·256토큰 상한으로 초기 pilot의 LLM.generate 결과와 일치했다.</p></section>
<section><h2>길이와 단계별 결과</h2><p>실제 최장 출력은 <b>32,768토큰</b>. seed 3407 단독 기준에서 {natural_ref}문항이 자연 생성으로 32,768토큰에 도달해 length 종료했다. 전체 실행에서는 반복 비교 출력을 포함해 {len(lengths)}개 출력이 도달했다. 강제 min_tokens/ignore_eos 진단 두 단계는 계획대로 생략했으며 강제 출력은 0개다. max_tokens 설정만으로 길이를 검증했다고 간주하지 않았다.</p><div class="scroll"><table><tr><th>단계</th><th>seed</th><th>출력</th><th>배치</th><th>실제 최장 토큰</th><th>32,768 도달</th></tr>{rows}</table></div></section>
<section><h2>실제 중단 후 복구</h2><p><b>계획된 SIGKILL:</b> {kst(fault['time'])} KST, Math_13 생성 중 132토큰을 관측한 뒤 제어기 자신의 자식 runner PID {fault['pid']}만 종료했다. 완료된 Agriculture_13·Music_21 두 파일의 SHA256과 기존 attempt가 그대로 유지됐다. 새 attempt는 Math_13·Psychology_17·Clinical_Medicine_13·Music_16 네 문항만 처음부터 재생성했다. 6문항 모두 무중단 기준과 완전 일치했다. 중간 토큰 상태를 이어 붙이는 복구가 아니다.</p>
<p><b>별도의 예기치 않은 중단:</b> seed 123 Accounting_5의 31,341토큰 기록이 10월 14일 11:50:27 KST에 남아 있었고, 서버 연결 복구 후 기존 제어기·작업자가 없음을 확인했다. 원인은 확인되지 않았다. 10월 15일 16:00에 재개하기 전 로그·상태와 완료 출력 169개 해시를 별도 보존했다. 최종 검사에서 이 169개 파일은 모두 동일했다. seed 123의 완료 7문항은 유지하고 미완료 5문항을 새 attempt로 재생성했다. 계획된 fault 검증과 이 사건을 구분한다.</p></section>
<section><h2>실측 시간</h2><p>첫 작업부터 실행 종료까지 {elapsed/3600:.2f}시간(서버 접근 불가 구간 포함). 로그로 관측된 worker 구간 합계는 {result['observed_worker_seconds']/3600:.2f}시간이다. 이 합계에는 모델 초기화와 계획된 중단 전 작업도 포함하며, 예기치 않은 중단 구간은 마지막 progress 시각까지만 계산했다. 순수 GPU 연산 시간이나 정확한 장애 지속시간으로 해석하면 안 된다.</p><div class="scroll"><table><tr><th>단계</th><th>시작 KST</th><th>종료/마지막 관측 KST</th><th>분</th><th>종료 근거</th></tr>{interval_rows}</table></div></section>
<section><h2>환경·체크포인트·입력 보존</h2><p>실제 버전: {esc(', '.join(k+' '+v for k,v in versions.items()))}. CUDA 런타임 13.0, RTX 4090, BF16. engine 및 SamplingParams seed를 함께 고정했고 eager=True, prefix cache=False, MM processor cache=0, VLLM_BATCH_INVARIANT=1, VLLM_ENABLE_V1_MULTIPROCESSING=0을 확인했다. 로그에서 FLASH_ATTN backend 및 배치 실행 중 실제 2·4개 요청의 동시 처리를 확인했다.</p><p>원래 입력 해상도·프롬프트·체크포인트를 유지했다. 60개 모델 입력 및 4개 Judge 입력 정합성 검사가 통과했고, 원본 RGB 이미지 {image_count}개 해시를 로컬에서 재확인했다. 구환경 코드·pip freeze와 새환경 pip freeze 보존 검사도 통과했다. 모델과 Judge 가중치 7개 및 config 2개의 실제 파일을 스트리밍 SHA256으로 다시 읽어 기존 값과 대조했으며 모두 일치했다.</p><p>Judge seed는 3407 유지. 이번 실행에서 새 Judge 추론이나 전체 재채점은 하지 않았다. 초기 pilot의 Judge 기준·역순·반복 증거를 보존하고 8개 비교를 별도로 재확인했다(이번 126개 비교에 포함하지 않음).</p></section>
<section><h2>회수 증거와 해석 범위</h2><p>ZIP SHA256: <code>{transfer['sha256']}</code>. ZIP의 418개 항목과 manifest에 기록된 417개 파일 해시를 검증했다. 경로 탈출·심볼릭 링크를 차단해 해제했으며, 기존 로컬 plan.json과 input_parity.json은 동일함을 확인해 보존했다. 가상환경·캐시·모델 가중치는 ZIP에 포함하지 않았다.</p><p><a href="independent_verification.json">독립 검증 JSON</a> · <a href="transfer_verification.json">다운로드 검증</a> · <a href="summary.json">실행기 요약</a> · <a href="plan.json">고정 계획</a> · <a href="provenance/checkpoint_final_verification.json">최종 가중치 해시</a></p><p>이 결과는 목적 표집에 대한 생성 재현성 검증이다. 정확도 평가, 학습 seed 실험, 모든 900문항·모든 장비·모든 배치 구성의 일치를 보장하지 않는다. 새 환경 5개 seed의 최종 정확도는 별도 본평가가 필요하다. 기존 vLLM 0.11.0의 5-seed 결과는 그대로 보존했고 900문항 본평가는 시작하지 않았다. 기존 환경과 새 환경 사이의 출력 동일성도 주장하지 않는다.</p></section></html>'''
(R/'report.html').write_text(report,encoding='utf8')
print(json.dumps({k:result[k] for k in ['passed','output_count','comparison_count','differences','max_output_tokens','engine_attempt_count','wall_seconds_including_outage','observed_worker_seconds']},indent=2))
print('NATURAL_32768',len(lengths),'REFERENCE3407',natural_ref)
