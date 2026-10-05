"""Single-node DDP, fully trainable random-init native Brújula; no weight download."""
import argparse
import contextlib
from datetime import timedelta
import fcntl
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from common import atomic_json,digest,output_path
from pretrain_core import DocumentChunks,make_model

def save_checkpoint(path,value):
    temp=path.with_suffix('.pt.tmp');torch.save(value,temp);os.replace(temp,path)

def amp(device):
    return torch.autocast('cuda',dtype=torch.bfloat16) if device.type=='cuda' else contextlib.nullcontext()

@torch.no_grad()
def evaluate(model,data,device,rank,world,batch):
    model.eval();loss=0.;count=0
    indices=list(range(rank,len(data),world))
    for start in range(0,len(indices),batch):
        x,y=data.batch(indices[start:start+batch]);x=x.to(device);y=y.to(device)
        with amp(device):value=model(x,y)
        loss+=float(value);count+=int((y!=-100).sum())
    totals=torch.tensor([loss,count],dtype=torch.float64,device=device)
    if world>1:dist.all_reduce(totals)
    model.train();nll=float(totals[0]/totals[1])
    if not math.isfinite(nll):raise FloatingPointError('Nonfinite evaluation')
    return dict(nll=nll,tokens=int(totals[1]),ppl=math.exp(min(nll,80)))

def parser():
    p=argparse.ArgumentParser()
    p.add_argument('--prepared',required=True);p.add_argument('--out',required=True)
    p.add_argument('--native-source',default=str(Path(__file__).resolve().parents[1]/'vendor/brujula'))
    p.add_argument('--events',required=True)
    p.add_argument('--variant',choices=['block4','full','full_rms','norm_rms'],default='block4')
    p.add_argument('--width',type=int,default=512);p.add_argument('--layers',type=int,default=12)
    p.add_argument('--heads',type=int,default=8);p.add_argument('--seq-len',type=int,default=512)
    p.add_argument('--kv-compression-dim',type=int,default=64)
    p.add_argument('--q-compression-dim',type=int,default=128)
    p.add_argument('--batch-size',type=int,default=8);p.add_argument('--accumulation',type=int,default=8)
    p.add_argument('--train-tokens',type=int,default=1_000_000_000)
    p.add_argument('--eval-tokens',type=int,default=5_000_000)
    p.add_argument('--lr',type=float,default=3e-4);p.add_argument('--warmup-fraction',type=float,default=.02)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--save-every',type=int,default=250)
    p.add_argument('--eval-every',type=int,default=500)
    p.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--stop-after-steps',type=int,default=0,help='Profile/pause, not successful training completion')
    p.add_argument('--no-checkpointing',action='store_true')
    p.add_argument('--publisher-config',action='store_true')
    p.add_argument('--tokenizer-source',help='Explicit tokenizer used to prepare the corpus; native vocabulary must match')
    return p

def initialize_run_config(config_path,config,resume):
    """Only rank zero checks preexistence; broadcast its decision to all ranks."""
    payload=[None]
    if not dist.is_initialized() or dist.get_rank()==0:
        try:
            if config_path.exists():
                if not resume:raise ValueError('Output already exists; use explicit --resume or a new directory')
                if json.loads(config_path.read_text())!=config:raise ValueError('Resume configuration mismatch')
            elif resume:raise ValueError('No existing run to resume')
            else:atomic_json(config_path,config)
            payload[0]={'config':config,'error':None}
        except Exception as e:
            payload[0]={'error':f'{type(e).__name__}: {e}'}
    if dist.is_initialized():dist.broadcast_object_list(payload,src=0)
    if payload[0]['error']:raise ValueError(payload[0]['error'])
    if payload[0]['config']!=config:raise ValueError('Configuration differs between ranks')

def main():
    a=parser().parse_args()
    if min(a.batch_size,a.accumulation,a.seq_len,a.train_tokens,a.eval_tokens,a.save_every,a.eval_every)<1:
        raise ValueError('Invalid positive budget')
    if not 0<=a.warmup_fraction<1 or a.lr<=0:raise ValueError('Invalid LR schedule')
    rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1));local=int(os.environ.get('LOCAL_RANK',0))
    torch.set_num_threads(2)
    if a.device=='cuda':torch.cuda.set_device(local)
    device=torch.device('cuda',local) if a.device=='cuda' else torch.device('cpu')
    if world>1:dist.init_process_group('nccl' if a.device=='cuda' else 'gloo',timeout=timedelta(minutes=15))
    out=output_path(a.out);events=output_path(a.events);out.mkdir(parents=True,exist_ok=True);events.mkdir(parents=True,exist_ok=True)
    lock=None
    if rank==0:
        lock=(out/'LOCK').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.manual_seed(a.seed)
    train=DocumentChunks(a.prepared,'train',a.seq_len,a.train_tokens)
    val=DocumentChunks(a.prepared,'validation',a.seq_len,a.eval_tokens)
    test=DocumentChunks(a.prepared,'test',a.seq_len,a.eval_tokens)
    source=Path(a.native_source)
    prepared_config=json.loads((Path(a.prepared)/'config.json').read_text())
    tokenizer_source=Path(a.tokenizer_source) if a.tokenizer_source else source
    if digest(source/'tokenizer.json')!=digest(tokenizer_source/'tokenizer.json'):
        raise ValueError('Native and corpus tokenizer vocabularies differ')
    for name,h in prepared_config['tokenizer_sha256'].items():
        if digest(tokenizer_source/name)!=h:raise ValueError('Tokenizer fingerprint does not match prepared data')
    global_batch=world*a.batch_size*a.accumulation
    steps=math.ceil(len(train)/global_batch);warmup=max(1,int(steps*a.warmup_fraction))
    config={k:v for k,v in vars(a).items() if k not in ('events','out','resume','stop_after_steps')}
    config.update(world=world,global_batch=global_batch,steps=steps,effective_train_labels=train.tokens,
                  data_fingerprint={'train':train.fingerprint,'validation':val.fingerprint,'test':test.fingerprint},prepared_config_sha256=digest(Path(a.prepared)/'config.json'),
                  native_code={n:digest(source/n) for n in ['configuration_brujula_v2.py','modeling_brujula_v2.py']},
                  training_code={n:digest(Path(__file__).with_name(n)) for n in ['train_pretrain.py','pretrain_core.py','common.py']})
    config_path=out/'config.json'
    initialize_run_config(config_path,config,a.resume)
    model=make_model(source,a.width,a.layers,a.heads,a.seq_len,
                     'attnres_full' if a.variant in ('full','full_rms') else 'attnres_block',
                     4,checkpointing=not a.no_checkpointing,
                     summary_mode=a.variant if a.variant in ('full_rms','norm_rms') else None,seed=a.seed,
                     publisher_config=a.publisher_config,
                     kv_compression_dim=a.kv_compression_dim,
                     q_compression_dim=a.q_compression_dim).to(device)
    assert all(p.requires_grad for p in model.parameters())
    params=sum(p.numel() for p in model.parameters())
    groups=[{'params':[p for p in model.parameters() if p.ndim>=2],'weight_decay':.1},
            {'params':[p for p in model.parameters() if p.ndim<2],'weight_decay':0.}]
    optimizer=torch.optim.AdamW(groups,lr=a.lr,betas=(.9,.95),eps=1e-8,
                               fused=device.type=='cuda')
    start_step=0;consumed=0
    if a.resume:
        ckpt=torch.load(out/'last.pt',map_location=device,weights_only=False)
        if ckpt['config']!=config:raise ValueError('Checkpoint/config mismatch')
        model.load_state_dict(ckpt['model']);optimizer.load_state_dict(ckpt['optimizer'])
        start_step=ckpt['step'];consumed=ckpt['tokens']
        torch.set_rng_state(ckpt['rng'].cpu())
    wrapped=DDP(model,device_ids=[local] if device.type=='cuda' else None,broadcast_buffers=False) if world>1 else model
    if rank==0:
        atomic_json(out/'native-config.json',model.native.config.to_dict())
        atomic_json(out/'runtime.json',dict(parameters=params,trainable_parameters=params,
            initialization='random_all_parameters' if not a.resume else 'resume_own_pretraining_checkpoint',
            pretrained_weights_loaded=False,torch=torch.__version__,cuda=torch.version.cuda,
            device=torch.cuda.get_device_name() if device.type=='cuda' else 'cpu',
            width=a.width,layers=a.layers,heads=a.heads))
        print(json.dumps(dict(stage='initialized',parameters=params,steps=steps,train_tokens=train.tokens)),flush=True)
    order=np.random.default_rng(a.seed).permutation(len(train));model.train();times=[];started=time.monotonic()
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats()
    ready_at=min(steps,start_step+3)
    for step in range(start_step,steps):
        tick=time.monotonic()
        selected=order[step*global_batch:(step+1)*global_batch].tolist()
        selected += [-1]*(global_batch-len(selected))
        ntokens=sum(int(train.records[i,2]) for i in selected if i>=0)
        if ntokens<=0:raise ValueError('Empty global batch')
        if step<warmup:lr=a.lr*(step+1)/warmup
        else:lr=a.lr*(.1+.9*.5*(1+math.cos(math.pi*(step-warmup)/max(1,steps-warmup-1))))
        for g in optimizer.param_groups:g['lr']=lr
        optimizer.zero_grad(set_to_none=True);local_loss=0.
        for micro in range(a.accumulation):
            base=(micro*world+rank)*a.batch_size
            ids,labels=train.batch(selected[base:base+a.batch_size]);ids=ids.to(device);labels=labels.to(device)
            sync=wrapped.no_sync() if world>1 and micro<a.accumulation-1 else contextlib.nullcontext()
            with sync,amp(device):
                loss=wrapped(ids,labels)
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
                scaled=loss*(world/ntokens)
            scaled.backward();local_loss+=float(loss.detach())
        grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        optimizer.step();consumed+=ntokens
        if hasattr(model.native,'spectral_summary'):model.native.spectral_summary.update()
        total_loss=torch.tensor(local_loss,dtype=torch.float64,device=device)
        if world>1:dist.all_reduce(total_loss)
        if device.type=='cuda':torch.cuda.synchronize()
        duration=time.monotonic()-tick;times.append(duration)
        if rank==0:
            record=dict(step=step+1,tokens=consumed,nll=float(total_loss)/ntokens,lr=lr,
                        grad_norm=float(grad),seconds=duration,tokens_per_second=ntokens/duration,
                        peak_allocated=torch.cuda.max_memory_allocated() if device.type=='cuda' else 0,
                        peak_reserved=torch.cuda.max_memory_reserved() if device.type=='cuda' else 0)
            with (out/'train.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            atomic_json(out/'status.json',record)
            if step+1==ready_at:atomic_json(events/'TRAIN_READY.json',record)
        stop=a.stop_after_steps and step+1>=a.stop_after_steps
        if (step+1)%a.save_every==0 or step+1==steps or stop:
            if rank==0:save_checkpoint(out/'last.pt',dict(model=model.state_dict(),optimizer=optimizer.state_dict(),
                config=config,step=step+1,tokens=consumed,rng=torch.get_rng_state()))
            if world>1:dist.barrier()
        if stop:
            if rank==0:atomic_json(out/'PAUSED.json',dict(step=step+1,tokens=consumed))
            break
        if (step+1)%a.eval_every==0 or step+1==steps:
            metric=evaluate(model,val,device,rank,world,a.batch_size)
            if rank==0:atomic_json(out/f'validation-{step+1:06d}.json',metric)
    else:
        metric=evaluate(model,test,device,rank,world,a.batch_size)
        if rank==0:
            atomic_json(out/'test.json',metric)
            atomic_json(out/'COMPLETE.json',dict(steps=steps,tokens=consumed,test=metric,parameters=params))
            atomic_json(events/'TRAIN_READY.json',dict(completed=True,steps=steps))
    if rank==0:
        atomic_json(out/'profile.json',dict(step_seconds=times,elapsed_seconds=time.monotonic()-started,
            steady_p50=float(np.median(times[2:])) if len(times)>2 else None,
            steady_p95=float(np.percentile(times[2:],95)) if len(times)>2 else None,
            peak_allocated=torch.cuda.max_memory_allocated() if device.type=='cuda' else 0,
            peak_reserved=torch.cuda.max_memory_reserved() if device.type=='cuda' else 0))
    if world>1:dist.destroy_process_group()
    if lock:lock.close()

if __name__=='__main__':main()
