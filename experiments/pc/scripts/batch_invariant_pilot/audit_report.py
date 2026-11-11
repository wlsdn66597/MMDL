"""Independently audit the downloaded pilot evidence and build a Korean HTML report."""
import ast
import datetime as dt
import hashlib
import html
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2] / 'results/mmmu_batch_invariant_pilot_20261014'


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def rows(p):
    return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines() if s.strip()]


def stamp(s):
    return dt.datetime.fromisoformat(s)


def main():
    manifest = read(ROOT/'evidence_manifest.json')
    for rel, digest in manifest['files'].items():
        p = (ROOT/rel).resolve()
        assert p.is_relative_to(ROOT.resolve())
        assert hashlib.sha256(p.read_bytes()).hexdigest() == digest, rel
    archive = ROOT.with_suffix('.zip')
    archive_hash = hashlib.sha256(archive.read_bytes()).hexdigest()
    assert archive_hash == archive.with_suffix('.zip.sha256').read_text().split()[0]
    frozen = read(ROOT/'inputs/frozen.json')
    summary = read(ROOT/'summary.json')
    status = read(ROOT/'status.json')
    parity = read(ROOT/'provenance/input_parity.json')
    preservation = read(ROOT/'provenance/original_environment_preserved.json')
    checkpoint = read(ROOT/'provenance/checkpoint_verification.json')
    assert status['state'] == 'complete' and summary['complete'] and summary['passed']
    assert parity['passed'] and preservation['passed'] and checkpoint['target_matches_original_baseline']
    vl_ids = [r['id'] for r in frozen['items']]
    judge_ids = [r['id'] for r in frozen['judges']]
    extended_ids = ['validation_Math_13', 'validation_Music_21', 'validation_Psychology_17']
    # Explicit independent stage specification; do not import the producer's comparator.
    specs = [
        ('vl_reference', '단독 기준', vl_ids, None, 1, 256),
        ('vl_pair', '배치 2', vl_ids, 'vl_reference', 2, 256),
        ('vl_four_reverse', '배치 4 · 역순', vl_ids, 'vl_reference', 4, 256),
        ('vl_four_mix', '배치 4 · 동반 문항 변경', vl_ids, 'vl_reference', 4, 256),
        ('vl_repeat', '새 프로세스 단독 반복', vl_ids, 'vl_reference', 1, 256),
        ('vl_resume_subset', '완료 문항 제외 부분 집합', vl_ids[1::2], 'vl_reference', 2, 256),
        ('vl_extended_reference', '2,048토큰 단독 기준', extended_ids, None, 1, 2048),
        ('vl_extended_pair', '2,048토큰 배치 2 · 역순', extended_ids, 'vl_extended_reference', 2, 2048),
        ('judge_reference', 'Judge 단독 기준', judge_ids, None, 1, 4096),
        ('judge_four_reverse', 'Judge 배치 4 · 역순', judge_ids, 'judge_reference', 4, 4096),
        ('judge_repeat', 'Judge 새 프로세스 반복', judge_ids, 'judge_reference', 1, 4096),
    ]
    all_rows = {name:rows(ROOT/'outputs'/f'{name}.jsonl') for name,*_ in specs}
    frozen_by_id = {r['id']:r for r in frozen['items'] + frozen['judges']}
    comparisons, stages = [], []
    for name,label,expected,reference,batch,cap in specs:
        data = all_rows[name]
        assert len(data) == len(expected) == len({r['id'] for r in data})
        assert {r['id'] for r in data} == set(expected)
        assert summary['stage_counts'][name] == dict(completed=len(data),expected=len(expected))
        engine = read(ROOT/'provenance'/f'{name}_engine.json')
        assert engine['environment_flag']=='1' and engine['recognized_flag'] is True
        assert engine['llm_kwargs']['max_num_seqs']==batch and engine['llm_kwargs']['enforce_eager']
        assert engine['sampling']['seed']==3407 and engine['sampling']['max_tokens']==cap
        assert engine['llm_kwargs']['model']==frozen['judge_path' if name.startswith('judge') else 'model_path']
        groups = {}
        for r in data:
            assert r['stage']==name and r['seed']==3407 and r['max_tokens']==cap
            assert len(r['token_ids'])==r['output_tokens']<=cap
            assert r['input_tokens']==len(frozen_by_id[r['id']]['prompt_token_ids'])
            assert r['finish_reason'] in ('stop','length')
            if r['finish_reason']=='length':
                assert r['output_tokens']==cap
            assert r['id'] in r['batch_ids'] and 1<=len(r['batch_ids'])<=batch
            groups.setdefault(tuple(r['batch_ids']), r['batch_elapsed_seconds'])
        if reference:
            ref = {r['id']:r for r in all_rows[reference]}
            for r in data:
                differences = [k for k in ('token_ids','text','finish_reason','stop_reason','input_tokens') if r[k]!=ref[r['id']][k]]
                comparisons.append(dict(stage=name,id=r['id'],passed=not differences,differences=differences))
        log = (ROOT/f'{name}.log').read_text(encoding='utf-8')
        batch_values = []
        for line in log.splitlines():
            marker='Batchsize forward time stats (batchsize, count, median_time(ms)): '
            if marker in line:
                batch_values += [int(x[0]) for x in ast.literal_eval(line.split(marker,1)[1])]
        running = [int(x) for x in re.findall(r'Running: (\d+) reqs',log)]
        assert 'Using FLASH_ATTN attention backend' in log
        if name in ('vl_pair','vl_four_reverse','vl_four_mix','judge_four_reverse'):
            assert batch in batch_values, (name,batch_values)
        stages.append(dict(stage=name,label=label,outputs=len(data),comparisons=len(data) if reference else 0,
                           max_num_seqs=batch,max_tokens=cap,stop=sum(r['finish_reason']=='stop' for r in data),
                           length=sum(r['finish_reason']=='length' for r in data),
                           output_tokens=sum(r['output_tokens'] for r in data),
                           generation_seconds=sum(groups.values()),init_seconds=engine['init_seconds'],
                           elapsed_seconds=(max(stamp(r['created_at']) for r in data)-stamp(engine['started_at'])).total_seconds(),
                           observed_forward_token_batch_sizes=sorted(set(batch_values)),
                           logged_max_running_requests=max(running,default=None)))
    assert len(comparisons)==65 and all(r['passed'] for r in comparisons)
    assert comparisons==summary['comparisons']
    short = {r['id']:r for r in all_rows['vl_reference']}
    prefix_checks = {r['id']:r['token_ids'][:len(short[r['id']]['token_ids'])]==short[r['id']]['token_ids'] for r in all_rows['vl_extended_reference']}
    events = rows(ROOT/'events.jsonl')
    actual_control_events = [r for r in events if r['event'] in ('pause_requested','resume_requested')]
    first = re.search(r'START (\S+)',(ROOT/'vl_reference.log').read_text(encoding='utf-8')).group(1)
    elapsed = (stamp(status['updated_at'])-stamp(first)).total_seconds()
    final = dict(checked_at=dt.datetime.now(dt.timezone.utc).isoformat(),passed=True,archive_sha256=archive_hash,
                 manifest_files_verified=len(manifest['files']),output_records=sum(len(x) for x in all_rows.values()),
                 exact_comparisons=len(comparisons),differences=0,stages=stages,input_parity=parity,
                 original_environment=preservation,checkpoint_matches_original=True,
                 controller_wall_seconds=elapsed,extended_prefix_matches=prefix_checks,
                 actual_pause_resume_events=actual_control_events,comparisons=comparisons)
    (ROOT/'independent_verification.json').write_text(json.dumps(final,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    def esc(x): return html.escape(str(x))
    def table(headers, body):
        return '<div class="table"><table><thead><tr>'+''.join('<th>'+esc(h)+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+esc(c)+'</td>' for c in row)+'</tr>' for row in body)+'</tbody></table></div>'
    stage_table=table(['조건','출력 수','정확 일치 비교','상한','자연 종료 / 상한 종료','생성 호출 시간'],[
        [s['label'],s['outputs'],str(s['comparisons'])+'/'+str(s['comparisons']) if s['comparisons'] else '기준',s['max_tokens'],f"{s['stop']} / {s['length']}",f"{s['generation_seconds']:.1f}초"] for s in stages])
    selected_table=table(['문항 ID','선정 근거','이미지 수','입력 토큰','256토큰 기준 출력 길이 / 종료'],[
        [r['id'],r['reason'],len(r['image_paths']),len(r['prompt_token_ids']),f"{short[r['id']]['output_tokens']} / {short[r['id']]['finish_reason']}"] for r in frozen['items']])
    environment_table=table(['항목','실제 검증 환경'],[
        ['vLLM / PyTorch / CUDA runtime','0.31.0 / 2.13.0 / 13.0'],['Transformers / qwen-vl-utils','5.17.0 / 0.0.14'],
        ['GPU / 정밀도','RTX 4090 24GB, compute capability 8.9 / BF16, 비양자화'],
        ['모델','Qwen3-VL-4B-Instruct · '+frozen['config']['model']['revision']],
        ['Judge','Qwen3-8B · '+frozen['config']['judge']['revision']],
        ['주요 설정','VLLM_BATCH_INVARIANT=1, VLLM_ENABLE_V1_MULTIPROCESSING=0, eager=True'],
        ['배치 / 캐시','max_num_seqs=1·2·4, prefix cache=False, 이미지 processor cache=0'],
        ['Attention','FLASH_ATTN / FlashAttention 2 (실행 로그)'],
        ['추론 seed / sampling','3407 / temperature 0.7, top_p 0.8, top_k 20, presence_penalty 1.5'],
        ['Judge sampling','temperature 0, seed 3407, enable_thinking=False'],
        ['입력 해상도·문맥','원본과 동일: min_pixels 1,003,520, max_pixels 4,014,080; context 64,000 (Judge 32,768)']])
    report=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>MMMU 배치 불변성 소규모 검증</title>
<style>body{{font-family:"Malgun Gothic",Arial,sans-serif;color:#182b3c;background:#f2f5f7;margin:0;line-height:1.7}}main{{max-width:1100px;margin:30px auto;background:white;padding:42px;border-radius:12px}}h1{{font-size:30px;line-height:1.4}}h2{{font-size:22px;margin-top:38px;border-bottom:2px solid #dce8ed;padding-bottom:8px}}h3{{font-size:17px}}.eyebrow{{color:#486576;font-size:13px}}.lead{{font-size:18px}}.cards{{display:flex;gap:14px;flex-wrap:wrap;margin:22px 0}}.card{{background:#eef7f4;padding:18px;flex:1;min-width:170px}}.card b{{font-size:27px;color:#14725b;display:block}}.note{{background:#fff5dd;border-left:4px solid #d4a53a;padding:14px 18px}}table{{width:100%;border-collapse:collapse;font-size:13px}}th,td{{padding:10px;text-align:left;border-bottom:1px solid #dce4e9;vertical-align:top;overflow-wrap:anywhere}}th{{background:#edf2f5}}.table{{overflow-x:auto}}code{{overflow-wrap:anywhere;font-size:12px}}a{{color:#176783}}footer{{font-size:12px;color:#5f7180;margin-top:35px}}@media(max-width:700px){{main{{padding:20px;margin:0}}h1{{font-size:25px}}}}@media print{{body{{background:white}}main{{margin:0;padding:0}}h2{{break-after:avoid}}tr{{break-inside:avoid}}}}</style>
<main><div class="eyebrow">MMMU · 별도 환경 사전 검증 · 2026년 10월 14일</div><h1>VLLM_BATCH_INVARIANT=1<br>소규모 검증 결과</h1>
<p class="lead"><strong>지정한 11단계와 65개 출력 비교를 모두 통과했다.</strong> 같은 체크포인트·seed·입력으로 배치 크기와 순서, 동반 문항, 새 프로세스 반복 조건을 바꿔도 출력 토큰과 종료 정보가 일치했다.</p>
<div class="cards"><div class="card"><b>65 / 65</b>기준 대비 정확 일치 비교</div><div class="card"><b>84개</b>모델 72회 + Judge 12회 출력</div><div class="card"><b>12 + 4</b>고유 이미지 문항 + Judge 입력</div><div class="card"><b>{elapsed/60:.2f}분</b>검증 제어기 실측 경과 시간</div></div>
<p>실행: 2026-10-14 01:22:49~01:33:49 KST. 새 환경 설치·다운로드 시간은 위 경과 시간에서 제외했다. 기존 baseline 환경의 패키지 목록과 평가 코드 6개 파일은 준비 전후가 같았고, 대상 모델 가중치·config 해시는 기존 seed3407 baseline과 일치했다.</p>
<div class="note"><strong>결론의 범위:</strong> 이 결과는 RTX 4090의 별도 환경에서 수행한 작은 사전 검증이다. 900문항 전체, 최대 32,768토큰 생성, 모든 seed, 실제 강제 중단 복구를 보장하지 않는다. 정확도 향상이나 flag의 속도·정확도 효과를 측정한 실험도 아니다.</div>
<h2>1. 실행 환경과 적용 근거</h2>{environment_table}
<p>단순한 환경변수 저장 외에 11개 엔진의 <code>recognized_flag=true</code>, 실제 backend 로그, vLLM 설치 소스의 linear·RMSNorm·attention 분기와 정확한 출력 일치를 함께 확인했다. 배치 2·4의 forward 로그와 요청 수 로그도 보존했다. 다만 개별 CUDA 커널을 독립 profiler로 추적한 것은 아니다.</p>
<p><a href="https://raw.githubusercontent.com/vllm-project/vllm/v0.31.0/docs/features/batch_invariance.md">vLLM 0.31.0 공식 문서</a>는 기능이 beta이며 NVIDIA compute capability 8.0 이상을 지원한다고 설명한다. Qwen3-VL-4B의 공식 검증 범위는 단일 이미지·비디오다. 이번 파일럿에서는 실제 데이터의 1~5장 이미지 입력을 추가로 확인했다.</p>
<h2>2. 입력 검증과 검사 방법</h2>
<p>구환경에서 MMMU 원본 이미지와 seed42의 기록을 대조해 12문항을 고정했다. 새 환경에서 프롬프트 문자열·토큰 ID, image_grid_thw, 처리 후 이미지 크기·RGB 해시, pixel_values tensor 해시가 모두 일치했다. Judge 입력 4개도 문자열·토큰 ID가 일치했다. 원본 해상도, 프롬프트, 체크포인트를 축소·교체하지 않았다.</p>
<p>주 검사 출력 상한은 256토큰, Math_13·Music_21·Psychology_17의 확장 검사는 2,048토큰이다. Judge 상한은 4,096토큰이다. 각 조건의 <code>token_ids, text, finish_reason, stop_reason, input_tokens</code>를 기준 실행과 정확히 비교했다. 서로 다른 길이 제한의 결과를 같은 기준으로 억지로 비교하지 않았다.</p>
{stage_table}
<p>표의 시간은 각 생성 호출의 실측 시간을 배치별 한 번씩 합한 값이다. 엔진 초기화 시간은 제외했으며, 전체 경과 시간에는 모델 재로딩·입력 준비·제어기 대기 등이 포함된다. cache를 켠 운영 환경의 성능이나 flag=0 대비 오버헤드로 해석할 수 없다.</p>
<h2>3. 표본과 출력 범위</h2>{selected_table}
<p>표본은 이미지 수, 긴 입력, 기존 잘림 사례, 과목 다양성을 고려한 목적 표본이다. 전체 30과목을 대표하지 않는다. 2,048토큰 확장 결과의 기존 256토큰 구간 일치도는 {sum(prefix_checks.values())}/{len(prefix_checks)}였으며, 이는 별도 보조 점검이다. Judge 답의 의미적 정답 여부를 수동 인증하거나 이 파일럿의 MMMU 점수를 산출하지 않았다.</p>
<h2>4. 발견한 호환 문제와 보존한 증거</h2>
<p>초기 Judge 입력 검사는 Transformers 5.17의 <code>apply_chat_template(tokenize=True)</code>가 기본적으로 <code>BatchEncoding</code>을 반환해 실패했다. 검사기가 이를 토큰 목록처럼 취급했다. 새 검증 코드에 <code>return_dict=False</code>를 명시한 뒤 4개 입력 모두 원본 토큰과 완전히 일치했다. 입력을 바꾸거나 비교 기준을 완화하지 않았다. 초기 실패 로그·코드·JSON은 <code>provenance/initial_preflight/</code>에 남겼다.</p>
<p>본 GPU 검증 중 오류나 출력 불일치는 없었다. 실제 pause/resume 요청 이벤트는 {len(actual_control_events)}개다. 이번의 ‘완료 문항 제외’ 조건은 부분 집합 재실행 검증이며, 실행 중 프로세스를 강제로 끊은 복구 실험은 아니다.</p>
<p>증거 ZIP의 SHA256 및 내부 원본 파일 {len(manifest['files'])}개의 해시를 로컬에서 검증했다. 실행기와 독립된 집계 코드가 JSONL 84개 기록, 문항 ID·토큰 수·종료 조건·설정, 원격 summary의 65개 비교를 재확인했다.</p>
<h2>5. 다음 본평가로 넘어가기 위한 판단</h2>
<p><strong>vLLM 0.31.0은 이 장비·모델 조합에서 별도 평가 환경의 사용 후보로 확인됐다.</strong> 기존 vLLM 0.11.0의 5-seed 결과는 기존 조건의 결과로 보존한다. 이번 검증 때문에 그 결과를 폐기하거나 학습을 다시 할 필요는 없다. 다만 ‘batch-invariant 환경의 최종 5-seed 점수’를 보고하려면 새 환경에서 본평가 5개 seed를 별도로 실행해야 하며, 구·신환경 결과를 같은 평균에 섞으면 안 된다.</p>
<ol><li>30과목을 포함한 약 60~100문항으로 범위를 넓히고, 1~5 이미지와 기존 length 사례를 포함해 실제 32,768토큰 상한에서 검사한다.</li><li>seed 3407 이외의 seed도 고정값별로 반복한다. 최종 사용할 캐시·eager/graph·batch 설정을 잠그고 설정이 달라지면 다시 검증한다.</li><li>실제 중단 후 남은 문항만 재개해 끊기지 않은 기준 실행과 비교한다. 타 GPU·드라이버·환경으로 옮기면 해당 조합을 별도로 확인한다.</li><li>본평가 전 Judge 파서·잘림 강제 오답 규칙을 유지하고, 새 환경의 seed3407 전체 반복 일치 및 900개 고유 ID·30과목 구성을 확인한다.</li></ol>
<p>위 항목은 후속 검증 제안이며 이번 작업에서 실행하지 않았다. flag=0 대조군을 돌리지 않았으므로 관측된 일치가 오직 flag 덕분이라고 주장하지 않는다. 이전의 같은 조건 3407 반복 통과와 이번의 배치 변경 검증은 서로 다른 근거다.</p>
<h2>6. 원본과 검증 파일</h2><p><a href="summary.json">원격 요약</a> · <a href="independent_verification.json">독립 검증 JSON</a> · <a href="provenance/input_parity.json">입력 정합성</a> · <a href="provenance/original_environment_preserved.json">기존 환경 보존</a> · <a href="provenance/checkpoint_verification.json">가중치 해시</a> · <a href="evidence_manifest.json">증거 파일 목록</a> · <a href="../mmmu_batch_invariant_pilot_20261014.zip">전체 원본 ZIP</a></p>
<p><code>ZIP SHA256: {archive_hash}</code></p><footer>로컬 보고서·독립 검증은 다운로드 후 작성된 파생 산출물이다. ZIP은 서버 원본 91개 파일을 담고 있으며 이 HTML을 포함하지 않는다. 원본 출력·이미지·설정·로그·실패 이력은 보존했다.</footer></main></html>'''
    (ROOT/'report.html').write_text(report,encoding='utf-8')
    print(json.dumps({k:final[k] for k in ('passed','output_records','exact_comparisons','differences','controller_wall_seconds','archive_sha256')},ensure_ascii=False))


if __name__ == '__main__':
    main()
