"""Tiny CPU train/save/resume/evaluate test. Synthetic tokens, not research evidence."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]

class TrainingEntry(unittest.TestCase):
 def test_train_resume_complete(self):
  with tempfile.TemporaryDirectory(prefix='rms-release-smoke-') as td:
   d=Path(td);source=d/'native';source.mkdir();prepared=d/'data';prepared.mkdir()
   for name in ['modeling_brujula_v2.py','configuration_brujula_v2.py']:shutil.copy2(ROOT/'vendor/brujula'/name,source/name)
   (source/'tokenizer.json').write_text('{}')
   sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
   (prepared/'config.json').write_text(json.dumps({'tokenizer_sha256':{'tokenizer.json':sha(source/'tokenizer.json')}}))
   entries={}
   for split,n in [('train',12),('validation',2),('test',2)]:
    toks=(np.arange(n*9,dtype=np.uint32)%251+1).astype('<u4');toks.tofile(prepared/(split+'.bin'))
    np.save(prepared/(split+'.npy'),np.array([[0,i*9,8] for i in range(n)],dtype=np.int64))
    entries[split]=dict(labels=n*8,bytes=n*9*4,index_sha256=sha(prepared/(split+'.npy')),fingerprint='synthetic-'+split)
   (prepared/'indexed.json').write_text(json.dumps(dict(seq_len=8,splits=entries)))
   (prepared/'COMPLETE.json').write_text('{}')
   cmd=[sys.executable,str(ROOT/'small/train_pretrain.py'),'--prepared',str(prepared),'--native-source',str(source),
        '--out',str(d/'run'),'--events',str(d/'events'),'--variant','full_rms','--device','cpu',
        '--width','64','--layers','4','--heads','4','--seq-len','8','--batch-size','2','--accumulation','2',
        '--train-tokens','96','--eval-tokens','16','--save-every','1','--eval-every','1']
   subprocess.run(cmd+['--stop-after-steps','2'],check=True,capture_output=True,text=True)
   self.assertTrue((d/'run/PAUSED.json').exists());self.assertFalse((d/'run/COMPLETE.json').exists())
   subprocess.run(cmd+['--resume'],check=True,capture_output=True,text=True)
   complete=json.loads((d/'run/COMPLETE.json').read_text())
   self.assertEqual(complete['steps'],3);self.assertEqual(complete['tokens'],96);self.assertEqual(complete['test']['tokens'],16)

if __name__=='__main__':unittest.main()
