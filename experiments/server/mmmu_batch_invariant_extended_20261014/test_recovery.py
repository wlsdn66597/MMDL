"""CPU-only checks of durable results, protocol drift, and recovery verification."""
from pathlib import Path
import tempfile
import unittest
import common as c
import control as ctl


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.root_before=c.ROOT;self.control_before=ctl.ROOT
        base=Path(__file__).resolve().parent/'test_tmp';base.mkdir(exist_ok=True)
        self.base=base;self.tmp=tempfile.TemporaryDirectory(dir=base)
        c.ROOT=ctl.ROOT=Path(self.tmp.name)
        for name in ('runner.py','input_helpers.py'): (c.ROOT/name).write_text('# frozen test source\n')
        ids=[f'item_{i}' for i in range(6)]
        self.specs=[dict(name='seed3407_reference',ids=ids,reference=None,seed=3407,max_tokens=32768,batch=1),
                    dict(name='actual_resume_seed3407',ids=ids,reference='seed3407_reference',seed=3407,max_tokens=32768,batch=1,inject_interrupt=True)]
        c.write(c.ROOT/'plan.json',dict(stages=self.specs))
        c.write(c.ROOT/'inputs/frozen.json',dict(items=[dict(id=i,prompt_token_ids=[1,2]) for i in ids]))
        for spec in self.specs:
            for j,i in enumerate(ids):
                row=dict(id=i,stage=spec['name'],seed=3407,max_tokens=32768,fingerprint=c.fingerprint(spec),
                         token_ids=[3,4],output_tokens=2,text='same',input_tokens=2,finish_reason='stop',stop_reason=None,
                         attempt='original' if j<2 else 'restarted')
                c.write(c.ROOT/'outputs'/spec['name']/f'{i}.json',row)
        hashes={i:c.sha(c.ROOT/'outputs/actual_resume_seed3407'/f'{i}.json') for i in ids[:2]}
        c.write(c.ROOT/'fault_injected.json',dict(stage='actual_resume_seed3407',attempt='original',committed_sha256=hashes,
                                                observed_inflight=[dict(id=ids[2],generated_tokens=128,finished=False)]))

    def tearDown(self):
        c.ROOT=self.root_before;ctl.ROOT=self.control_before
        assert Path(self.tmp.name).resolve().is_relative_to(self.base.resolve())
        self.tmp.cleanup()

    def test_recovery_retains_committed_files(self):
        ctl.verify_recovery();self.assertTrue(c.read(c.ROOT/'recovery_verification.json')['passed'])

    def test_changed_saved_file_fails(self):
        p=c.ROOT/'outputs/actual_resume_seed3407/item_0.json';r=c.read(p);r['attempt']='regenerated';c.write(p,r)
        with self.assertRaises(AssertionError):ctl.verify_recovery()

    def test_output_difference_fails(self):
        p=c.ROOT/'outputs/actual_resume_seed3407/item_3.json';r=c.read(p);r['token_ids']=[9,4];c.write(p,r)
        with self.assertRaises(AssertionError):ctl.verify_recovery()

    def test_temporary_incomplete_file_is_not_a_completion(self):
        p=c.ROOT/'outputs/actual_resume_seed3407/not_finished.999.tmp';p.write_text('{partial')
        self.assertEqual(len(c.records('actual_resume_seed3407')),6)

    def test_protocol_drift_rejected(self):
        row=c.records('actual_resume_seed3407')['item_0'];(c.ROOT/'runner.py').write_text('# changed')
        with self.assertRaises(AssertionError):c.check_record(row,self.specs[1],{'item_0':dict(prompt_token_ids=[1,2])})

    def test_event_can_preserve_fault_time(self):
        c.event('planned_inflight_sigkill',time='observed time',pid=123)
        self.assertIn('observed time',(c.ROOT/'events.jsonl').read_text())


if __name__=='__main__':unittest.main()
