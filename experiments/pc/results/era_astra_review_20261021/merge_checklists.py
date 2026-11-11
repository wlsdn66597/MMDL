"""Merge actual human entries; never fabricate decisions or approve a release."""
import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def main():
    expected = {r['sample_id']: r for r in read_csv(ROOT/'human_review_tasks.csv')}
    rows = []
    errors = []
    for reviewer in 'ABCD':
        path = ROOT/f'reviewer_{reviewer}'/'checklist.csv'
        if not path.exists():
            errors.append(f'missing_file:{path.relative_to(ROOT)}')
            continue
        for r in read_csv(path):
            if r.get('reviewer') != reviewer:
                errors.append(f'wrong_reviewer_file:{r.get("sample_id")}')
            rows.append(r)
    counts = Counter(r.get('sample_id') for r in rows)
    errors += [f'duplicate:{sid}' for sid,n in counts.items() if n>1]
    errors += [f'missing:{sid}' for sid in expected if sid not in counts]
    errors += [f'unexpected:{sid}' for sid in counts if sid not in expected]
    pending=[]; completed=[]; sample_issues=[]; actions=Counter()
    immutable=['sample_id','owner','reviewer','record_sha256','review_reason','astra_verdict','astra_issues']
    for r in rows:
        sid=r.get('sample_id'); old=expected.get(sid)
        if old is None:continue
        row_errors=[]
        for key in immutable:
            if r.get(key)!=old[key]:row_errors.append(f'changed_locked_field:{key}')
        if r.get('status')=='pending':
            pending.append(sid)
        elif r.get('status')=='reviewed':
            for key in ['notes','reviewer_name','reviewed_at','human_answer']:
                if not (r.get(key) or '').strip():row_errors.append(f'missing:{key}')
            if r.get('human_assessment') not in {'confirmed_issue','no_issue','uncertain'}:row_errors.append('invalid_assessment')
            if r.get('action') not in {'keep','revise','replace','hold'}:row_errors.append('invalid_action')
            if r.get('human_assessment')=='uncertain' and r.get('action')!='hold':row_errors.append('uncertain_requires_hold')
            if r.get('human_answer')!='uncertain' and r.get('human_answer') not in list('ABCDEFGHIJ'):row_errors.append('invalid_answer')
            if 'repeated_visual_family' in r['review_reason'] and not r.get('family_decision','').strip():row_errors.append('missing_family_decision')
            try:
                stamp=datetime.fromisoformat(r.get('reviewed_at',''))
                if stamp.tzinfo is None:row_errors.append('timestamp_requires_timezone')
            except ValueError:row_errors.append('invalid_timestamp')
            if not row_errors:
                completed.append(sid);actions[r['action']]+=1
                if r['human_assessment']=='confirmed_issue' and 'random_sample_20pct' in r['review_reason']:sample_issues.append(sid)
        else:row_errors.append('invalid_status')
        errors.extend(f'{sid}:{e}' for e in row_errors)
    if len(sample_issues)>=2:expansion='full_remaining_model_pass_review_required'
    elif sample_issues:expansion='same_owner_task_family_plus_additional_sample_required'
    else:expansion='manual_cross_owner_error_pattern_check_still_required'
    audit=dict(expected=len(expected),rows_received=len(rows),valid_completed=len(completed),pending_ids=pending,errors=errors,actions=dict(actions),confirmed_sample_issue_ids=sample_issues,expansion=expansion,all_initial_tasks_recorded=not errors and len(completed)==len(expected),final_training_release=False,note='CSV completeness only. Revisions, source-family decisions, same-error cross-owner expansion, re-review and release gates require separate adjudication.')
    (ROOT/'human_merge_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    with (ROOT/'human_review_merged.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(next(iter(expected.values()))),extrasaction='ignore')
        w.writeheader();w.writerows(rows)
    print(json.dumps({k:v for k,v in audit.items() if k!='pending_ids'},ensure_ascii=True,indent=2))
    return 0 if audit['all_initial_tasks_recorded'] else 1

if __name__=='__main__':
    raise SystemExit(main())
