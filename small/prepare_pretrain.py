"""Download a bounded prefix and tokenize whole documents; row-group atomic resume.

Output is little-endian uint32 token shards and JSONL document offsets. It is
not yet a training loader. Every document gets one EOS, no chat template/BOS.
Hash buckets 0..9 validation, 10..19 test, 20..999 train. Buckets remain reserved
even after a split reaches its target. Exact normalized duplicates are removed;
near-duplicate filtering is NOT implemented and must precede final claims.
"""
import argparse
import collections
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import unicodedata

import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer

REPO = 'HuggingFaceFW/fineweb-edu'
REVISION = '87f09149ef4734204d70ed1d046ddc9ca3f2b8f9'
ROOT = Path('.')


def doc_hash(text):
    normalized = ' '.join(unicodedata.normalize('NFKC', text).split())
    return hashlib.sha256(normalized.encode()).hexdigest()


def split_for(key):
    bucket = int(key[:16], 16) % 1000
    return 'validation' if bucket < 10 else 'test' if bucket < 20 else 'train'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def save(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--tokenizer', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--train-tokens', type=int, default=1_000_000_000)
    ap.add_argument('--heldout-tokens', type=int, default=5_000_000)
    ap.add_argument('--max-files', type=int, default=4)
    ap.add_argument('--local-only', action='store_true')
    ap.add_argument('--max-groups', type=int, default=0, help='Bounded profiling/resume test')
    ap.add_argument('--exclusion-hashes', type=Path, default=Path(__file__).resolve().parents[1]/'evidence/data/exclusion-hashes.json')
    args = ap.parse_args()
    if min(args.train_tokens, args.heldout_tokens, args.max_files) <= 0:
        raise ValueError('Budgets must be positive')
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out/'LOCK').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args)


def run(a):
    targets = dict(train=a.train_tokens, validation=a.heldout_tokens, test=a.heldout_tokens)
    tokenizer_files = {p.name: sha(p) for p in a.tokenizer.iterdir()
                       if p.name in ('tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json')}
    exclusions = json.loads(a.exclusion_hashes.read_text())
    if exclusions['normalization'] != 'nfkc-whitespace-sha256-v1': raise ValueError('Exclusion format mismatch')
    config = dict(repo=REPO, revision=REVISION, tokenizer=str(a.tokenizer.resolve()),
                  tokenizer_sha256=tokenizer_files, targets=targets, max_files=a.max_files,
                  data=str(a.data.resolve()), code_sha256=sha(Path(__file__)),
                  exclusions={'hash_manifest': sha(a.exclusion_hashes)}, split='nfkc-whitespace-sha256-v1')
    cp = a.out/'config.json'
    if cp.exists() and json.loads(cp.read_text()) != config:
        raise ValueError('Resume configuration mismatch; use a new output directory')
    save(cp, config)
    seen = set(exclusions['hashes'])
    totals = collections.Counter()
    completed = set()
    for part in sorted(a.out.glob('part-*')):
        if part.name.endswith('.pending'):
            continue
        m = json.loads((part/'manifest.json').read_text())
        for s, count in m['counts'].items():
            if (part/(s+'.bin')).stat().st_size != count*4:
                raise ValueError('Committed shard size mismatch')
        with (part/'docs.jsonl').open() as f:
            for line in f:
                key=json.loads(line)['hash']
                if key in seen:
                    raise ValueError('Duplicate committed document')
                seen.add(key)
        totals.update(m['counts']); completed.add((m['file'], m['row_group']))
    done = lambda: all(totals[s] >= targets[s] for s in targets)
    if done():
        save(a.out/'COMPLETE.json', dict(counts=dict(totals), parts=len(completed)))
        print('Already complete', flush=True); return
    tok = AutoTokenizer.from_pretrained(a.tokenizer, local_files_only=True, trust_remote_code=False)
    if not tok.is_fast or tok.eos_token_id is None:
        raise ValueError('Fast tokenizer with EOS required')
    if len(tok) >= 2**32:
        raise ValueError('Token vocabulary exceeds uint32')
    if a.local_only:
        files = sorted(p.relative_to(a.data).as_posix() for p in (a.data/'sample/100BT').glob('*.parquet'))[:a.max_files]
    else:
        from huggingface_hub import HfApi, get_token
        token = get_token()
        if not token:
            raise ValueError('Configured HF token unavailable')
        files = sorted(x.path for x in HfApi(token=token).list_repo_tree(
            REPO, path_in_repo='sample/100BT', repo_type='dataset', revision=REVISION)
            if x.path.endswith('.parquet'))[:a.max_files]
    save(a.out/'input-plan.json', dict(files=files))
    began=time.monotonic(); new_groups=0
    for name in files:
        path=a.data/name
        if not a.local_only:
            from huggingface_hub import hf_hub_download
            print(json.dumps(dict(stage='download', file=name)), flush=True)
            path=Path(hf_hub_download(REPO, name, repo_type='dataset', revision=REVISION,
                                     local_dir=a.data, token=token))
        pf=pq.ParquetFile(path)  # A valid footer is necessary, not a full integrity proof.
        fingerprint=sha(path)
        for rg in range(pf.num_row_groups):
            if (name,rg) in completed:
                part=a.out/('part-'+hashlib.sha256(name.encode()).hexdigest()[:12]+f'-{rg:05d}')
                if json.loads((part/'manifest.json').read_text())['input_sha256'] != fingerprint:
                    raise ValueError('Committed input changed')
                continue
            t=time.monotonic(); rejected=collections.Counter()
            texts=pf.read_row_group(rg, columns=['text']).column('text').to_pylist()
            rows=[]; buffers={s:[] for s in targets}; counts=collections.Counter()
            for offset in range(0,len(texts),128):
                candidates=[]
                for row, text in enumerate(texts[offset:offset+128],offset):
                    if not text or not text.strip(): rejected['empty']+=1; continue
                    key=doc_hash(text); s=split_for(key)
                    if key in seen: rejected['duplicate_or_old']+=1; continue
                    if totals[s]+counts[s]>=targets[s]: rejected['split_budget_met']+=1; continue
                    seen.add(key); candidates.append((row,text,key,s))
                ids=tok([c[1] for c in candidates], add_special_tokens=False,
                        truncation=False, return_attention_mask=False)['input_ids'] if candidates else []
                for (row,text,key,s), tokens in zip(candidates,ids):
                    if totals[s]+counts[s]>=targets[s]:
                        seen.remove(key); rejected['split_budget_met']+=1; continue
                    tokens.append(tok.eos_token_id)
                    rows.append(dict(hash=key, split=s, row=row, offset=counts[s], length=len(tokens)))
                    buffers[s].extend(tokens); counts[s]+=len(tokens)
            part=a.out/('part-'+hashlib.sha256(name.encode()).hexdigest()[:12]+f'-{rg:05d}')
            temp=a.out/(part.name+'.pending')
            # Only this program's uncommitted staging directory may be replaced.
            if temp.exists(): shutil.rmtree(temp)
            temp.mkdir()
            for s in targets:
                np.asarray(buffers[s],dtype='<u4').tofile(temp/(s+'.bin'))
            with (temp/'docs.jsonl').open('w') as f:
                for row in rows: f.write(json.dumps(row)+'\n')
            m=dict(file=name,row_group=rg,input_sha256=fingerprint,counts=dict(counts),
                   documents=len(rows), input_documents=len(texts), rejected=dict(rejected),
                   seconds=time.monotonic()-t)
            save(temp/'manifest.json',m); os.replace(temp,part)
            totals.update(counts); completed.add((name,rg)); new_groups+=1
            status=dict(stage='tokenize',counts=dict(totals),parts=len(completed),
                        last_group=m, elapsed_seconds=time.monotonic()-began)
            save(a.out/'status.json',status); print(json.dumps(status),flush=True)
            if done():
                save(a.out/'COMPLETE.json',dict(counts=dict(totals),parts=len(completed)))
                return
            if a.max_groups and new_groups>=a.max_groups:
                print('Profiling limit reached; safe to resume',flush=True); return
    save(a.out/'INSUFFICIENT_DATA.json',dict(counts=dict(totals),targets=targets,files=files))
    raise SystemExit('Input budget exhausted before token targets; no COMPLETE marker')


if __name__ == '__main__':
    main()
