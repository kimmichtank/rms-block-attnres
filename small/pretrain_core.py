"""Native randomly initialized Brújula and document-isolated token chunks."""
import collections
import functools
import json
from pathlib import Path
import sys
import types

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from common import digest

def load_native(source):
    # Read native architecture code ONLY, never pretrained weights or model config.
    sys.path.insert(0,str(Path(source).resolve()))
    from configuration_brujula_v2 import BrujulaConfig
    from modeling_brujula_v2 import BrujulaForCausalLM
    return BrujulaConfig,BrujulaForCausalLM

def make_model(source, width=512,layers=12,heads=8,seq_len=512,
               residual='attnres_block',block_size=4,vocab=50257,checkpointing=True,
               summary_mode=None,seed=42,publisher_config=False,
               kv_compression_dim=64,q_compression_dim=128):
    C,M=load_native(source)
    if width%heads or (width//heads)%2:raise ValueError('Invalid head dimensions')
    config=C(vocab_size=vocab,n_embd=width,n_layer=layers,n_head=heads,block_size=seq_len,
             kv_compression_dim=kv_compression_dim,q_compression_dim=q_compression_dim,
             dropout=0.,residual=residual,
             attnres_block_size=block_size,rope_scaling_method=None,
             rope_scale_len=seq_len,rope_trained_len=seq_len,tie_word_embeddings=True,
             bos_token_id=min(50256,vocab-1),eos_token_id=min(50256,vocab-1))
    if publisher_config:
        config=C.from_dict(json.loads((Path(source)/'config.json').read_text()))
        config.residual=residual
        config.attnres_block_size=block_size
        config.block_size=seq_len
        config.max_position_embeddings=seq_len
        config.rope_scaling_method=None
        config.rope_scale_len=seq_len
        config.rope_trained_len=seq_len
        config.dropout=0.
        width,layers,heads=config.n_embd,config.n_layer,config.n_head
    native=M(config)
    if summary_mode=='full_rms':
        if residual!='attnres_full':raise ValueError('Full+RMS requires attnres_full residual routing')
        for module in native.modules():
            if type(module).__name__=='AttnResAggregator':
                module.forward=types.MethodType(full_rms_aggregate,module)
    elif summary_mode:
        if 2*layers%4:raise ValueError('RMS Block requires complete groups of four sublayer outputs')
        if summary_mode != 'norm_rms': raise ValueError('Only RMS Block is released here')
        cls=HiddenNormalizedSum
        native.spectral_summary=cls(2*layers//4,summary_mode,seed)
        native._run_backbone=types.MethodType(two_summary_backbone,native)
    if checkpointing:
        def wrapped(fn,*args):
            # Full AttnRes appends to its source list after each read. Snapshot it
            # so backward recomputation sees the exact original source set.
            args=tuple(tuple(v) if isinstance(v,list) else v for v in args)
            return checkpoint(fn,*args,use_reentrant=False) if torch.is_grad_enabled() else fn(*args)
        for module in native.modules():
            if type(module).__name__ in ('MultiHeadLatentAttention','SquaredReLU','AttnResAggregator'):
                module.forward=functools.partial(wrapped,module.forward)
    return TrainLM(native)

def full_rms_aggregate(module,prior_values):
    """Full AttnRes routing with the existing normalized keys also used as values."""
    values=torch.stack(prior_values,dim=0)
    normalized_values=module.key_norm(values)
    logits=torch.einsum('d,lbtd->lbt',module.w,normalized_values)
    weights=F.softmax(logits,dim=0)
    return (weights.unsqueeze(-1)*normalized_values).sum(dim=0)

class HiddenNormalizedSum(nn.Module):
    """No affine gain; normalize each raw output only when four writes complete."""
    def __init__(self, blocks, mode, seed):
        super().__init__()
        if mode != 'norm_rms': raise ValueError(mode)
        self.mode = mode

    def transform(self, value):
        x = value.float()
        x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)
        return x.to(value.dtype)

    def forward(self, values, index, mask):
        return (sum(self.transform(v) for v in values),)

    def update(self):
        pass

def two_summary_backbone(m,x,cos,sin):
    emb=x;completed=[];pending=[];index=0
    def sources():
        return [emb]+completed+([sum(pending)] if pending else [])
    def append(value):
        nonlocal index
        pending.append(value)
        if len(pending)==4:
            completed.extend(m.spectral_summary(pending,index,m.summary_mask));pending.clear();index+=1
    for block in m.blocks:
        append(block.sa(block.ln1(block.attn_res_attn(sources())),cos,sin))
        append(block.ffwd(block.ln2(block.attn_res_ffn(sources()))))
    if pending:raise AssertionError('Incomplete summary block')
    return m.attn_res_final([emb]+completed)

class TrainLM(nn.Module):
    def __init__(self,native):
        super().__init__();self.native=native

    def forward(self,ids,targets):
        m=self.native
        x=m.token_embedding_table(ids)
        if hasattr(m,'spectral_summary'):m.summary_mask=targets!=-100
        cos,sin=m.rope(ids.shape[1],ids.device,x.dtype)
        hidden=m.ln_f(m._run_backbone(x,cos,sin)).reshape(-1,x.shape[-1])
        y=targets.reshape(-1)
        keep=y!=-100; hidden=hidden[keep]; y=y[keep]
        # Small recomputed vocabulary projections avoid B*T*V peak storage.
        loss=hidden.sum()*0+m.lm_head.weight.sum()*0
        def ce(h,t):
            return F.cross_entropy(F.linear(h,m.lm_head.weight).float(),t,reduction='sum')
        for start in range(0,len(y),256):
            h,t=hidden[start:start+256],y[start:start+256]
            loss=loss+(checkpoint(ce,h,t,use_reentrant=False) if torch.is_grad_enabled() else ce(h,t))
        return loss

class DocumentChunks:
    """Index one document at a time, never permit attention across documents.

    A length L document supplies L-1 labels, EOS included. Consecutive chunks
    share one context token. Right padding cannot affect preceding causal tokens.
    """
    def __init__(self,root,split,seq_len,max_tokens=0):
        root=Path(root);self.seq_len=seq_len;self.paths=[];records=[];self.maps=collections.OrderedDict()
        if not (root/'COMPLETE.json').exists():raise ValueError('Tokenization incomplete')
        if (root/'indexed.json').exists():
            index=json.loads((root/'indexed.json').read_text())
            if index['seq_len']!=seq_len:raise ValueError('Indexed sequence length mismatch')
            entry=index['splits'][split]
            if max_tokens!=entry['labels']:raise ValueError('Indexed token budget mismatch')
            self.paths=[root/(split+'.bin')]
            if self.paths[0].stat().st_size!=entry['bytes']:raise ValueError('Indexed token size mismatch')
            self.records=np.load(root/(split+'.npy'),mmap_mode='r')
            if digest(root/(split+'.npy'))!=entry['index_sha256']:raise ValueError('Index hash mismatch')
            self.tokens=entry['labels'];self.fingerprint=entry['fingerprint']
            return
        total=0;manifest_hashes=[];full_counts=collections.Counter();seen=set()
        for part in sorted(root.glob('part-*')):
            if part.name.endswith('.pending'):continue
            m=json.loads((part/'manifest.json').read_text());full_counts.update(m['counts'])
            path=part/(split+'.bin'); size=path.stat().st_size
            if size!=m['counts'].get(split,0)*4:raise ValueError('Token file size mismatch')
            manifest_hashes.append((part.name,digest(part/'manifest.json'),digest(part/'docs.jsonl'),digest(path)))
            file_id=len(self.paths);self.paths.append(path)
            with (part/'docs.jsonl').open() as f:
                for line in f:
                    d=json.loads(line)
                    if d['split']!=split:continue
                    if d['hash'] in seen:raise ValueError('Repeated document hash')
                    seen.add(d['hash'])
                    offset,length=d['offset'],d['length']
                    if offset<0 or length<1 or (offset+length)*4>size:raise ValueError('Invalid document offsets')
                    for pos in range(0,length-1,seq_len):
                        n=min(seq_len,length-1-pos)
                        if max_tokens:n=min(n,max(0,max_tokens-total))
                        if n:records.append((file_id,offset+pos,n));total+=n
        counts=json.loads((root/'COMPLETE.json').read_text())['counts']
        if dict(full_counts)!=counts:raise ValueError('Completion counts mismatch')
        self.records=np.asarray(records,dtype=np.int64).reshape(-1,3);self.tokens=total
        import hashlib
        self.fingerprint=hashlib.sha256(json.dumps(manifest_hashes).encode()).hexdigest()
        if not len(records):raise ValueError('No usable labels')

    def __len__(self):return len(self.records)

    def get(self,index):
        file_id,start,n=map(int,self.records[index]);path=self.paths[file_id]
        if file_id not in self.maps:
            self.maps[file_id]=np.memmap(path,dtype='<u4',mode='r')
            if len(self.maps)>16:self.maps.popitem(last=False)
        self.maps.move_to_end(file_id)
        tokens=np.asarray(self.maps[file_id][start:start+n+1],dtype=np.int64)
        return tokens

    def batch(self,indices,vocab=50257):
        x=torch.zeros((len(indices),self.seq_len),dtype=torch.long)
        y=torch.full_like(x,-100)
        for row,i in enumerate(indices):
            if i<0:continue
            ids=self.get(i)
            if ids.min()<0 or ids.max()>=vocab:raise ValueError('Token outside vocabulary')
            n=len(ids)-1;x[row,:n]=torch.from_numpy(ids[:-1].copy());y[row,:n]=torch.from_numpy(ids[1:].copy())
        return x,y
