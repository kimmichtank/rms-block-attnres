#!/usr/bin/env python3
"""Portable four-variant launch entry. No scheduler, credentials or cluster defaults."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--size',choices=['57m','157m','450m'],required=True)
 p.add_argument('--variant',choices=['full','full_rms','block','rms_block'],required=True)
 p.add_argument('--prepared',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
 p.add_argument('--world-size',type=int,default=4)
 p.add_argument('--micro-batch',type=int);p.add_argument('--global-batch',type=int)
 p.add_argument('--seed',type=int,default=42);p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
 p.add_argument('--resume',action='store_true');p.add_argument('--stop-after-steps',type=int,default=0)
 p.add_argument('--dry-run',action='store_true')
 a=p.parse_args()
 settings={
  '57m':dict(width=512,layers=12,heads=8,seq=512,kv=64,q=128,
             micro=64,global_batch=256,tokens=1_000_000_000,publisher=False),
  '157m':dict(width=816,layers=18,heads=6,seq=1024,kv=64,q=192,
              micro=16,global_batch=64,tokens=3_200_000_000,publisher=True),
  '450m':dict(width=1280,layers=28,heads=8,seq=1024,kv=112,q=320,
              micro=16,global_batch=64,tokens=10_000_000_000,publisher=False),
 }[a.size]
 if a.world_size < 1 or (a.micro_batch is not None and a.micro_batch < 1) or (a.global_batch is not None and a.global_batch < 1):
  p.error('World size and batch sizes must be positive')
 if a.device=='cpu' and a.world_size!=1:p.error('CPU smoke tests require --world-size 1; distributed training uses NCCL')
 mbs=a.micro_batch or settings['micro'];gbs=a.global_batch or settings['global_batch']
 if mbs*a.world_size>gbs or gbs%(mbs*a.world_size):p.error('Global batch must be divisible by micro-batch × world-size')
 mode={'full':'full','full_rms':'full_rms','block':'block4','rms_block':'norm_rms'}[a.variant]
 cmd=[sys.executable]
 if a.world_size>1:cmd+=['-m','torch.distributed.run','--standalone',f'--nproc_per_node={a.world_size}']
 cmd+=[str(ROOT/'small/train_pretrain.py'),'--prepared',str(a.prepared.resolve()),'--out',str(a.out.resolve()),
       '--events',str(a.out.resolve()/'events'),'--native-source',str(ROOT/'vendor/brujula'),'--variant',mode,
       '--width',str(settings['width']),'--layers',str(settings['layers']),'--heads',str(settings['heads']),
       '--seq-len',str(settings['seq']),'--kv-compression-dim',str(settings['kv']),
       '--q-compression-dim',str(settings['q']),'--batch-size',str(mbs),
       '--accumulation',str(gbs//(mbs*a.world_size)),
       '--train-tokens',str(settings['tokens']),'--eval-tokens','5000000',
       '--seed',str(a.seed),'--lr','0.0003','--warmup-fraction','0.02','--device',a.device,
       '--save-every','250','--eval-every','500']
 if settings['publisher']:cmd+=['--publisher-config']
 if a.resume:cmd+=['--resume']
 if a.stop_after_steps:cmd+=['--stop-after-steps',str(a.stop_after_steps)]
 import shlex
 print(shlex.join(cmd),flush=True)
 if not a.dry_run:subprocess.run(cmd,check=True)

if __name__=='__main__':main()
