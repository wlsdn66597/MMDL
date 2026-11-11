"""Protocol-drift guards, executable on CPU/Windows without model packages."""
import ast
import copy
from pathlib import Path
import unittest

source=Path(__file__).resolve().parents[1]/'tools/plain_mmmu900_campaign_20261102.py'
tree=ast.parse(source.read_text(encoding='utf8'))
module=ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('verify_pair_manifest','learning_rate_history')],type_ignores=[])
namespace={};exec(compile(module,str(source),'exec'),namespace)

class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.era={'status':'complete','identity':{'arm':'era','source_sha256':{'train.py':'frozen'},'data_sha256':'paired'},'training':{'epochs':4,'lr_scheduler_type':'cosine'},'lora':{'r':16},'initial_adapter_sha256':'init','final_adapter_sha256':'era_final','base_files_sha256':{'base':'same'},'data_manifest':{'items':200},'processor_provenance':{'vision':'same'},'loss_normalization':'same','actual_optimizer_steps':52,'frozen_weights_unchanged':True,'environment':{'versions':{'torch':'2.8.0'},'gpu':'4090','cuda':'12.8'}}
        self.plain=copy.deepcopy(self.era);self.plain['identity']['arm']='plain';self.plain['final_adapter_sha256']='plain_final'
    def test_only_target_arm_may_differ(self):namespace['verify_pair_manifest'](self.era,self.plain)
    def test_rejects_data_code_initialization_budget_drift(self):
        for key,value in [('identity',{'arm':'plain','data_sha256':'different'}),('initial_adapter_sha256','different'),('actual_optimizer_steps',26),('final_adapter_sha256','era_final'),('frozen_weights_unchanged',False),('training',{'epochs':2})]:
            with self.subTest(key=key):
                candidate=copy.deepcopy(self.plain);candidate[key]=value
                with self.assertRaises(AssertionError):namespace['verify_pair_manifest'](self.era,candidate)
    def test_schedule_comparison_keeps_step_and_lr(self):
        lr=namespace['learning_rate_history']
        self.assertEqual(lr({'log_history':[{'step':1,'loss':2,'learning_rate':0.0},{'train_runtime':1},{'step':2,'learning_rate':0.0001}]}),[(1,0.0),(2,0.0001)])
        self.assertNotEqual(lr({'log_history':[{'step':2,'learning_rate':0.0001}]}),lr({'log_history':[{'step':2,'learning_rate':0.00005}]}))

if __name__=='__main__':unittest.main()
