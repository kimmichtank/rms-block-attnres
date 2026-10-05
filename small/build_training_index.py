"""Audit full corpus metadata, then materialize bounded, document-isolated inputs."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import pyarrow.parquet as pq
from common import atomic_json, digest, output_path

def build(source, out, train_tokens=3200000000, eval_tokens=5000000, seq_len=1024, require_full=True):
    source=Path(source);out=output_path(out)
    if out.exists():raise ValueError('Index output exists; do not overwrite')
    if not (source/'COMPLETE.json').exists():raise ValueError('Tokenization incomplete')
    complete=json.loads((source/'COMPLETE.json').read_text())
    cfg=json.loads((source/'config.json').read_text())
    counts=Counter();groups=defaultdict(set);parts=[]
    for part in sorted(source.glob('part-*')):
        if part.name.endswith('.pending'):continue
        m=json.loads((part/'manifest.json').read_text())
        if m['row_group'] in groups[m['file']]:raise ValueError('Duplicate row group')
        groups[m['file']].add(m['row_group']);counts.update(m['counts'])
        for s in ('train','validation','test'):
            if (part/(s+'.bin')).stat().st_size!=m['counts'].get(s,0)*4:raise ValueError('Shard size mismatch')
        parts.append(part)
    if dict(counts)!=complete['counts'] or len(parts)!=complete['parts']:raise ValueError('Completion counts mismatch')
    if require_full:
        if complete['files']!=140 or len(groups)!=140:raise ValueError('Not all 140 files complete')
        for f in cfg['files']:
            path=Path(cfg['data'])/f['name']
            if path.stat().st_size!=f['bytes']:raise ValueError('Raw input changed')
            n=pq.ParquetFile(path).num_row_groups
            if groups[f['name']]!=set(range(n)):raise ValueError('Missing input row group')
    out.mkdir(parents=True)
    atomic_json(out/'audit.json',dict(source=str(source),counts=dict(counts),parts=len(parts),files=len(groups)))
    targets=dict(train=train_tokens,validation=eval_tokens,test=eval_tokens)
    totals=Counter();offsets=Counter();records={s:[] for s in targets};seen=set()
    handles={s:(out/(s+'.bin')).open('xb') for s in targets}
    selected=hashlib.sha256()
    try:
        for part in parts:
            maps={s:np.memmap(part/(s+'.bin'),dtype='<u4',mode='r')
                  for s in targets if totals[s]<targets[s] and (part/(s+'.bin')).stat().st_size}
            with (part/'docs.jsonl').open() as f:
                for line in f:
                    d=json.loads(line);s=d['split']
                    if totals[s]>=targets[s]:continue
                    key=d['hash']
                    if key in seen:raise ValueError('Duplicate selected document')
                    seen.add(key)
                    bucket=int(key[:16],16)%1000
                    if s!=('validation' if bucket<10 else 'test' if bucket<20 else 'train'):raise ValueError('Split mismatch')
                    start,length=d['offset'],d['length']
                    if start<0 or length<1 or start+length>len(maps[s]):raise ValueError('Document bounds invalid')
                    labels=min(length-1,targets[s]-totals[s])
                    if not labels:continue
                    tokens=maps[s][start:start+labels+1]
                    if int(tokens.max())>=50257:raise ValueError('Invalid token ID')
                    handles[s].write(tokens.tobytes())
                    selected.update((part.name+line).encode())
                    for pos in range(0,labels,seq_len):
                        records[s].append((0,offsets[s]+pos,min(seq_len,labels-pos)))
                    offsets[s]+=labels+1;totals[s]+=labels
            if all(totals[s]==targets[s] for s in targets):break
    finally:
        for f in handles.values():f.close()
    if dict(totals)!=targets:raise ValueError('Insufficient usable labels')
    entries={}
    for s in targets:
        p=out/(s+'.npy');np.save(p,np.asarray(records[s],dtype=np.int64))
        token_hash=digest(out/(s+'.bin'));index_hash=digest(p)
        entries[s]=dict(labels=totals[s],bytes=offsets[s]*4,token_sha256=token_hash,index_sha256=index_hash,
                        fingerprint=hashlib.sha256((token_hash+index_hash).encode()).hexdigest(),chunks=len(records[s]))
    shutil.copy2(source/'config.json',out/'config.json')
    atomic_json(out/'indexed.json',dict(seq_len=seq_len,splits=entries,selection='sorted part names, document order',
        selected_documents_sha256=selected.hexdigest(),source_config_sha256=digest(source/'config.json')))
    atomic_json(out/'COMPLETE.json',dict(counts=dict(offsets),labels=dict(totals)))
    return entries

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--out',required=True)
    p.add_argument('--train-tokens',type=int,default=3200000000);p.add_argument('--eval-tokens',type=int,default=5000000)
    p.add_argument('--seq-len',type=int,default=1024);p.add_argument('--allow-partial',action='store_true')
    a=p.parse_args();print(json.dumps(build(a.source,a.out,a.train_tokens,a.eval_tokens,a.seq_len,not a.allow_partial)),flush=True)
