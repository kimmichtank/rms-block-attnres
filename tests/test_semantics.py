"""CPU semantic checks; not a substitute for a distributed convergence replication."""
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'small'))
sys.path.insert(0,str(ROOT/'vendor/brujula'))
from pretrain_core import HiddenNormalizedSum, DocumentChunks, make_model, full_rms_aggregate
from common import digest
from modeling_brujula_v2 import AttnResAggregator


class Semantics(unittest.TestCase):
 def setUp(self):
  torch.manual_seed(42);torch.set_num_threads(2)

 def test_norm_is_per_token_hidden_dimension(self):
  n=HiddenNormalizedSum(1,'norm_rms',42)
  x=torch.randn(3,7,16)*5
  z=n.transform(x)
  self.assertTrue(torch.allclose(z.square().mean(-1),torch.ones(3,7),atol=1e-5))
  changed=x.clone();changed[2,4]=100
  self.assertTrue(torch.equal(z[0],n.transform(changed)[0]))

 def test_full_raw_and_normalized_values_scale_behavior(self):
  agg=AttnResAggregator(16).double();agg.key_norm.eps=0.
  agg.w.data.normal_();agg.key_norm.weight.data.uniform_(.5,1.5)
  src=[torch.randn(2,5,16,dtype=torch.double) for _ in range(4)]
  new=[x.clone() for x in src];new[2]*=4
  def weights(s):return torch.softmax(torch.einsum('d,lbtd->lbt',agg.w,agg.key_norm(torch.stack(s))),0)
  a=weights(src);self.assertTrue(torch.allclose(a,weights(new),atol=1e-12))
  self.assertTrue(torch.allclose(agg(new)-agg(src),3*a[2].unsqueeze(-1)*src[2],atol=1e-12))
  self.assertTrue(torch.allclose(full_rms_aggregate(agg,src),full_rms_aggregate(agg,new),atol=1e-12))

 def test_completed_source_normalized_but_pending_raw(self):
  n=HiddenNormalizedSum(1,'norm_rms',42)
  v=[torch.randn(2,4,16)*3 for _ in range(4)];w=[x.clone() for x in v];w[1]*=4
  self.assertTrue(torch.allclose(n(v,0,None)[0],n(w,0,None)[0],atol=1e-6))
  self.assertFalse(torch.allclose(sum(v[:2]),sum(w[:2])))

 def test_four_models_forward_backward_and_no_extra_parameters(self):
  counts=[]
  for variant in ('full','full_rms','block4','norm_rms'):
   torch.manual_seed(42)
   m=make_model(ROOT/'vendor/brujula',width=64,layers=4,heads=4,seq_len=8,vocab=257,
       residual='attnres_full' if variant.startswith('full') else 'attnres_block',
       summary_mode=variant if variant in ('full_rms','norm_rms') else None,checkpointing=True)
   x=torch.randint(0,257,(2,8));y=torch.randint(0,257,(2,8));y[1,5:]=-100
   loss=m(x,y);self.assertTrue(torch.isfinite(loss));loss.backward()
   self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()))
   counts.append(sum(p.numel() for p in m.parameters()))
  self.assertEqual(len(set(counts)),1)

 def test_checkpoint_recomputation_parity(self):
  for variant in ('full','full_rms','norm_rms'):
   kw=dict(width=64,layers=4,heads=4,seq_len=8,vocab=257,
      residual='attnres_full' if variant.startswith('full') else 'attnres_block',summary_mode=variant if variant!='full' else None)
   torch.manual_seed(7);a=make_model(ROOT/'vendor/brujula',checkpointing=False,**kw)
   torch.manual_seed(7);b=make_model(ROOT/'vendor/brujula',checkpointing=True,**kw)
   x=torch.randint(0,257,(2,8));y=torch.randint(0,257,(2,8))
   la=a(x,y);lb=b(x,y);la.backward();lb.backward()
   self.assertTrue(torch.equal(la,lb))
   for pa,pb in zip(a.parameters(),b.parameters()):self.assertTrue(torch.allclose(pa.grad,pb.grad,rtol=1e-5,atol=1e-6))

 def test_document_boundaries_and_padding(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);a=np.array([1,2,3,4,7,8],dtype='<u4');a.tofile(p/'test.bin')
   np.save(p/'test.npy',np.array([[0,0,3],[0,4,1]],dtype=np.int64))
   (p/'COMPLETE.json').write_text('{}')
   (p/'indexed.json').write_text(json.dumps(dict(seq_len=4,splits={'test':dict(labels=4,bytes=24,index_sha256=digest(p/'test.npy'),fingerprint='test')})))
   ds=DocumentChunks(p,'test',4,4);x,y=ds.batch([0,1,-1])
   self.assertEqual(x[0].tolist(),[1,2,3,0]);self.assertEqual(y[0].tolist(),[2,3,4,-100])
   self.assertEqual(y[1].tolist(),[8,-100,-100,-100]);self.assertEqual(y[2].tolist(),[-100]*4)

if __name__=='__main__':unittest.main()
