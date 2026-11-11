"""Exercise the gate using real completed fixtures and deliberate corruptions."""
import copy
from pathlib import Path
import tempfile
from unittest.mock import patch
import mmmu_multiseed_report as report

ROOT = Path(__file__).resolve().parents[1]
FIRST = ROOT / 'results/mmmu_first_baseline_20261009/baseline_seed3407'
SECOND = ROOT / 'results/mmmu_visual_tokens_seed42_20261010/baseline_seed42'
original_read = report.read


def main():
    with tempfile.TemporaryDirectory(dir=ROOT / 'tmp', prefix='multiseed-check-') as temporary:
        out = Path(temporary) / 'comparison.json'
        assert report.compare_repeat(FIRST, FIRST, out)['passed']
        changed_seed = report.compare_repeat(FIRST, SECOND, out)
        assert not changed_seed['passed'] and len(changed_seed['record_differences']) > 0
        data = original_read(FIRST / 'results.json')
        fake = Path(temporary) / 'fake'
        def altered_read(path):
            return mutated if Path(path) == fake / 'results.json' else original_read(path)
        attempts = report.jsonl(FIRST / 'judge_attempts.jsonl')
        for field, should_pass in [('elapsed_seconds', True), ('raw', False), ('output_tokens', False)]:
            mutated = copy.deepcopy(data)
            value = mutated['records'][0]['model_response'][field]
            mutated['records'][0]['model_response'][field] = value + (' altered' if isinstance(value, str) else 1)
            with patch.object(report, 'read', side_effect=altered_read), patch.object(report, 'jsonl', return_value=attempts):
                assert report.compare_repeat(FIRST, fake, out)['passed'] == should_pass, field
        mutated = copy.deepcopy(data)
        calls = 0
        changed_attempts = copy.deepcopy(attempts)
        changed_attempts[0]['raw'] += ' altered'
        def altered_attempts(path):
            return changed_attempts if Path(path).parent == fake else attempts
        with patch.object(report, 'read', side_effect=altered_read), patch.object(report, 'jsonl', side_effect=altered_attempts):
            assert not report.compare_repeat(FIRST, fake, out)['passed']
        items = original_read(ROOT / 'reports/mmmu_eval_protocol/evaluation_items.json')
        ids = {r['id'] for r in items}
        config = original_read(ROOT / 'reports/mmmu_eval_protocol/config.seed3407.json')
        summary = original_read(FIRST / 'summary.json')
        def bad_summary(path):
            return {**summary, 'correct': 900} if Path(path) == FIRST / 'summary.json' else original_read(path)
        with patch.object(report, 'read', side_effect=bad_summary):
            try:
                report.validate(FIRST, ids, config)
            except AssertionError:
                pass
            else:
                raise AssertionError('Corrupt summary was accepted')
        print('PASS: identical baseline, changed seed, timing-only change, raw/token/attempt mismatch, corrupt summary')


if __name__ == '__main__':
    main()
