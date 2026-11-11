"""Meaningful regression checks for leakage, false merging and stale metadata."""
import contextlib
import copy
import io
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
import visual_groups as v


class VisualGroupsTests(unittest.TestCase):
    def setUp(self):
        test_root=Path(__file__).resolve().parent/'.test_runs'
        test_root.mkdir(exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=test_root)
        # Verify the concrete recursive-cleanup target remains inside this test workspace.
        self.assertTrue(Path(self.temp.name).resolve().is_relative_to(test_root))
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        rng=np.random.default_rng(3407)
        self.array=rng.integers(0,256,(96,96,3),dtype=np.uint8)
        self.image('a',self.array)
        changed=self.array.copy();changed[10,10,0]^=1
        self.image('b',changed)
        self.image('c',rng.integers(0,256,(96,96,3),dtype=np.uint8))

    def image(self,name,array):
        path=self.root/f'{name}.png';Image.fromarray(array).save(path)
        return path

    def info(self,name):
        path=self.root/f'{name}.png'
        return dict(sample_id=name,package='C',source_id=name,image_group_no=9,image_path=str(path),**v.fingerprint(path))

    def decision(self,a,b,verdict):
        return dict(left=a,right=b,left_sha256=self.info(a)['file_sha256'],right_sha256=self.info(b)['file_sha256'],verdict=verdict,reviewer='test',reason='synthetic fixture')

    def registry(self,names,decisions=()):
        data=[self.info(n) for n in names];decs=v.validated_decisions(data,decisions);groups=v.assign(data,decs)
        return dict(hash_profile=v.PROFILE,images=data,groups=groups,decisions=decs)

    def manifest(self,name,ids,registry=None):
        records=[]
        for sid in ids:
            r=dict(sample_id=sid,image_path=f'{sid}.png')
            if registry:r['visual_group_id']=next(x['visual_group_id'] for x in registry['images'] if x['sample_id']==sid)
            records.append(r)
        path=self.root/f'{name}.jsonl';v.jsonl(path,records);return path

    def test_exact_pixels_merge_despite_different_png_bytes(self):
        Image.fromarray(self.array).save(self.root/'d.png',compress_level=0)
        self.assertNotEqual(self.info('a')['file_sha256'],self.info('d')['file_sha256'])
        self.assertEqual(self.info('a')['pixel_sha256'],self.info('d')['pixel_sha256'])
        self.assertEqual(len(self.registry(['a','d'])['groups']),1)

    def test_near_hash_does_not_automatically_merge(self):
        self.assertTrue(v.is_candidate(self.info('a'),self.info('b')))
        self.assertEqual(len(self.registry(['a','b'])['groups']),2)

    def test_reviewed_near_image_across_splits_is_leak(self):
        reg=self.registry(['a','b'],[self.decision('a','b','same')])
        result=v.check_records({'train':self.manifest('train',['a'],reg),'test':self.manifest('test',['b'],reg)},reg)
        self.assertEqual(result['exit_code'],1)
        self.assertEqual(len(result['leaks']),1)

    def test_unreviewed_cross_split_similarity_requires_review(self):
        reg=self.registry(['a','b'])
        result=v.check_records({'train':self.manifest('train',['a']),'test':self.manifest('test',['b'])},reg)
        self.assertEqual(result['exit_code'],2)

    def test_reviewed_different_content_can_cross_splits(self):
        reg=self.registry(['a','b'],[self.decision('a','b','different')])
        result=v.check_records({'train':self.manifest('train',['a'],reg),'test':self.manifest('test',['b'],reg)},reg)
        self.assertEqual(result['exit_code'],0)

    def test_unregistered_exact_duplicate_still_leaks(self):
        Image.fromarray(self.array).save(self.root/'d.png',compress_level=0)
        reg=self.registry(['c'])
        result=v.check_records({'train':self.manifest('train',['a']),'test':self.manifest('test',['d'])},reg)
        self.assertEqual(result['exit_code'],1)

    def test_repeated_training_image_is_allowed(self):
        reg=self.registry(['a','b','c'],[self.decision('a','b','same')])
        result=v.check_records({'train':self.manifest('train',['a','b'],reg),'test':self.manifest('test',['c'],reg)},reg)
        self.assertEqual(result['exit_code'],0)

    def test_actual_file_change_invalidates_group_claim(self):
        reg=self.registry(['a','c'])
        train=self.manifest('train',['a'],reg);test=self.manifest('test',['c'],reg)
        Image.fromarray(np.zeros_like(self.array)).save(self.root/'a.png')
        self.assertEqual(v.check_records({'train':train,'test':test},reg)['exit_code'],3)

    def test_missing_image_fails_closed(self):
        reg=self.registry(['a','c']);train=self.manifest('train',['a'],reg);test=self.manifest('test',['c'],reg)
        (self.root/'a.png').unlink()
        self.assertEqual(v.check_records({'train':train,'test':test},reg)['exit_code'],3)

    def test_stale_decision_rejected(self):
        decision=self.decision('a','b','same');decision['left_sha256']='wrong'
        with self.assertRaises(ValueError):self.registry(['a','b'],[decision])

    def test_transitive_conflicting_decisions_rejected(self):
        decs=[self.decision('a','b','same'),self.decision('b','c','same'),self.decision('a','c','different')]
        with self.assertRaises(ValueError):self.registry(['a','b','c'],decs)

    def test_existing_group_id_survives_new_member(self):
        reg=self.registry(['a']);data=[self.info('a'),self.info('b')]
        decs=v.validated_decisions(data,[self.decision('a','b','same')]);groups=v.assign(data,decs,reg)
        self.assertEqual(groups[0]['visual_group_id'],reg['groups'][0]['visual_group_id'])

    def test_existing_groups_cannot_be_silently_renumbered(self):
        reg=self.registry(['a','b']);data=[self.info('a'),self.info('b')]
        decs=v.validated_decisions(data,[self.decision('a','b','same')])
        with self.assertRaises(ValueError):v.assign(data,decs,reg)

    def test_registry_profile_mismatch_rejected(self):
        reg=self.registry(['a']);reg['hash_profile']={'version':'other'}
        with self.assertRaises(ValueError):v.check_records({},reg)

    def test_cli_checks_all_three_partition_pairs(self):
        reg=self.registry(['a','b','c'],[self.decision('a','b','same')]);v.dump(self.root/'reg.json',reg)
        args=['check','--registry',str(self.root/'reg.json'),'--report',str(self.root/'report.json')]
        for split,sid in [('train','c'),('dev','a'),('test','b')]:args+=['--partition',split+'='+str(self.manifest(split,[sid],reg))]
        with contextlib.redirect_stdout(io.StringIO()):code=v.main(args)
        self.assertEqual(code,1)
        self.assertEqual({v.load(self.root/'report.json')['leaks'][0][k] for k in ['left_split','right_split']},{'dev','test'})

    def test_build_preserves_assignments_and_model_text(self):
        row=dict(sample_id='a',source_id='123',owner='C',image_path='a.png',image_group_no=47,
                 input_prompt='Question unchanged',response_general='General unchanged',response_era='ERA unchanged')
        path=self.root/'input.jsonl';v.jsonl(path,[row]);out=self.root/'out'
        with contextlib.redirect_stdout(io.StringIO()):code=v.main(['build','--package','C='+str(path),'--output',str(out)])
        self.assertEqual(code,0);item=v.rows(out/'C/paired_train.draft.jsonl')[0]
        for k in ['sample_id','owner','image_group_no','input_prompt','response_general','response_era']:self.assertEqual(item[k],row[k])
        self.assertTrue(v.image_path(item,out/'C/paired_train.draft.jsonl').exists())
        self.assertEqual(v.rows(path),[row])
        self.assertNotIn('visual_group_id',v.rows(out/'C/train_plain.draft.jsonl')[0]['messages'][0])

    def test_solid_image_phash_is_numerically_stable(self):
        self.image('white',np.full((96,96,3),255,dtype=np.uint8))
        self.assertEqual(self.info('white')['phash'],'8000000000000000')


if __name__=='__main__':unittest.main(verbosity=2)
