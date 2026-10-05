#!/usr/bin/env python3
"""Generic Megatron command generator; prints by default, executes only on request."""
import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--variant',choices=['full','block','rms_block'],required=True)
 p.add_argument('--megatron-root',type=Path,required=True)
 p.add_argument('--data-prefix',type=Path,required=True)
 p.add_argument('--tokenizer-dir',type=Path,required=True)
 p.add_argument('--out',type=Path,required=True)
 p.add_argument('--master-addr',required=True);p.add_argument('--master-port',type=int,default=29500)
 p.add_argument('--node-rank',type=int,required=True);p.add_argument('--nodes',type=int,default=8)
 p.add_argument('--devices-per-node',type=int,default=8)
 p.add_argument('--allow-topology-change',action='store_true')
 p.add_argument('--execute',action='store_true')
 a=p.parse_args()
 # Resolve relative paths before changing cwd to the external Megatron checkout.
 for key in ('megatron_root','data_prefix','tokenizer_dir','out'):
  setattr(a,key,getattr(a,key).resolve())
 if (a.nodes,a.devices_per_node)!=(8,8) and not a.allow_topology_change:
  p.error('Historical topology is8×8/TP2/DP32. Explicitly acknowledge any execution change.')
 if not 0<=a.node_rank<a.nodes:p.error('Invalid node rank')
 if (a.nodes*a.devices_per_node)%2 or 256%((a.nodes*a.devices_per_node)//2):p.error('Global batch256 must be divisible by DP size')
 tokens=(ROOT/'evidence/4b/full/arguments.txt').read_text().splitlines()
 replace={'--attention-residual-type':a.variant,'--data-path':a.data_prefix,
          '--tokenizer-model':a.tokenizer_dir,'--save':a.out/'checkpoints',
          '--tensorboard-dir':a.out/'tensorboard','--data-cache-path':a.out.parent/'data-cache'}
 for flag,value in replace.items():tokens[tokens.index(flag)+1]=str(value)
 if any('<LOCAL_PATH' in s for s in tokens):p.error('An unresolved private-path placeholder remains')
 cmd=[sys.executable,'-m','torch.distributed.run',f'--nnodes={a.nodes}',f'--nproc_per_node={a.devices_per_node}',
      f'--node_rank={a.node_rank}',f'--master_addr={a.master_addr}',f'--master_port={a.master_port}',
      'pretrain_gpt.py']+tokens
 env=os.environ.copy()
 for key in ['SYNC_TIED_WGRAD','CHUNKED_TIED_WGRAD','VALIDATE_TIED_WGRAD','TIED_WEIGHT_PROXY','VALIDATE_TIED_OUTPUT_WGRAD','GRAD_DIAGNOSTICS']:
  env['ATTNRES_'+key]='0'
 env.update(ATTNRES_GRAD_BUCKET_DIAGNOSTICS='1',ATTNRES_GRAD_DIAGNOSTIC_THRESHOLD='1000',OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false')
 print(shlex.join(cmd))
 print('Bucket diagnostics enabled; experimental tied-gradient workarounds disabled. Full+RMS is not implemented in this audited4B patch.')
 if a.execute:
  cp=a.out/'checkpoints'
  if cp.exists() and any(cp.iterdir()):p.error('Refusing to start over existing checkpoint contents; resume must be designed explicitly')
  subprocess.run(cmd,cwd=a.megatron_root,env=env,check=True)

if __name__=='__main__':main()
