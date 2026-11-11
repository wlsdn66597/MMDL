"""Resolve pilot parser failures using the unchanged baseline local judge."""
import argparse
import copy
import os
from pathlib import Path
import traceback

import evaluate
import judge
import math13_occlusion as experiment_code
import parsers
import prepare


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    root = Path(args.output).resolve()
    lock = evaluate.run_lock(root / '.experiment.lock')
    try:
        exp = prepare.read(root / 'experiment.json')
        c = exp['configuration']
        records = prepare.read_jsonl(root / 'predictions.jsonl')
        assert len(records) == exp['expected_generations']
        assert len({(r['condition'], r['seed']) for r in records}) == len(records)
        pending = []
        for r in records:
            assert r['fingerprint'] == exp['fingerprint']
            r['grading_target_parser'] = copy.deepcopy(r['grading'])
            if r['grading']['status'] == 'pending_judge':
                row = copy.deepcopy(r)
                row['id'] = f"{r['id']}__{r['condition']}__seed{r['seed']}"
                r['judge_record_id'] = row['id']
                pending.append(row)
        run_dir = root / 'judge'
        run_dir.mkdir(exist_ok=True)
        path = run_dir / 'predictions.jsonl'
        if path.exists():
            assert prepare.read_jsonl(path) == pending, 'Judge inputs changed'
        else:
            for r in pending:
                prepare.append(path, r)
        prepare.write(root / 'status.json', {'state': 'running', 'phase': 'frozen_local_judge',
            'completed': len(records), 'expected': len(records), 'judge_pending': len(pending), 'updated_at': prepare.now()})
        last = evaluate.saved_judges(pending, prepare.read_jsonl(run_dir / 'judge_attempts.jsonl'), c)
        remaining = []
        for r in pending:
            a = last.get(r['id'])
            if not a or (not parsers.grade_judge(r, a['raw'])['judge_parser_decision']['decision_parsed']
                         and a['attempt'] < c['judge']['max_attempts']):
                remaining.append(r['id'])
        if remaining:
            prepare.write(run_dir / 'pending_judge.json', remaining)
            judge.run(c, run_dir, {'cache_paths': exp['cache_paths']})
        last = evaluate.saved_judges(pending, prepare.read_jsonl(run_dir / 'judge_attempts.jsonl'), c)
        for r in records:
            if 'judge_record_id' in r:
                a = last[r['judge_record_id']]
                r['grading'] = parsers.grade_judge(r, a['raw'])
                r['judge_final_output'] = a['raw']
                r['judge_attempt'] = a['attempt']
        assert all(r['grading']['status'] == 'graded' for r in records)
        summary = experiment_code.summarize(exp, records)
        summary['judge_resolved_records'] = len(pending)
        summary['judge_attempts'] = len(prepare.read_jsonl(run_dir / 'judge_attempts.jsonl'))
        prepare.write(root / 'summary.json', summary)
        prepare.write(root / 'results.json', {'records': records, 'summary': summary,
            'predictions_sha256': prepare.digest(root / 'predictions.jsonl'),
            'experiment_fingerprint': exp['fingerprint'], 'scored_at': prepare.now(),
            'scoring_code_sha256': {name: prepare.digest(prepare.ROOT / 'src' / name)
                for name in ('parsers.py', 'judge.py', 'evaluate.py', 'score_math13_occlusion.py')}})
        prepare.write(root / 'status.json', {'state': 'complete', 'phase': 'all_scored',
            'completed': len(records), 'expected': len(records), 'all_scored': True,
            'judge_resolved_records': len(pending), 'updated_at': prepare.now()})
        print(summary, flush=True)
    except Exception:
        prepare.write(root / 'status.json', {'state': 'failed', 'phase': 'scoring',
            'error': traceback.format_exc(), 'updated_at': prepare.now()})
        raise


if __name__ == '__main__':
    os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
    main()
