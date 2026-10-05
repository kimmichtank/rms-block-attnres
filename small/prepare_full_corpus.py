"""Full local corpus preparation, using the established document/split format.

No token-budget truncation. Committed row groups are immutable and resumable.
Input SHA256 is checked on each pass; the original 1B dataset is not modified.
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

import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer
from prepare_pretrain import ROOT, REPO, REVISION, doc_hash, split_for, sha, save


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--tokenizer', type=Path, required=True)
    p.add_argument('--expected-files', type=int, default=140)
    p.add_argument('--max-groups', type=int, default=0)
    p.add_argument('--exclusion-hashes', type=Path, default=Path(__file__).resolve().parents[1]/'evidence/data/exclusion-hashes.json', help='Normalized SHA256 exclusions')
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    with (a.out/'LOCK').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(a)


def run(a):
    paths = sorted((a.data/'sample/100BT').glob('*.parquet'))
    if len(paths) != a.expected_files:
        raise ValueError(f'Expected {a.expected_files} files, found {len(paths)}')
    config = dict(repo=REPO, revision=REVISION, mode='all-documents',
                  data=str(a.data.resolve()), tokenizer=str(a.tokenizer.resolve()),
                  tokenizer_sha256={p.name: sha(p) for p in a.tokenizer.iterdir()
                      if p.name in ('tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json')},
                  code_sha256=sha(Path(__file__)), helper_sha256=sha(Path(__file__).with_name('prepare_pretrain.py')),
                  exclusions={'hash_manifest': sha(a.exclusion_hashes)},
                  files=[dict(name=p.relative_to(a.data).as_posix(), bytes=p.stat().st_size) for p in paths],
                  split='nfkc-whitespace-sha256-v1; buckets 0-9 validation, 10-19 test, 20-999 train')
    cp = a.out/'config.json'
    if cp.exists() and json.loads(cp.read_text()) != config:
        raise ValueError('Resume configuration mismatch')
    save(cp, config)
    splits = ('train', 'validation', 'test')
    seen = set()
    if a.exclusion_hashes:
        manifest = json.loads(a.exclusion_hashes.read_text())
        if manifest['normalization'] != 'nfkc-whitespace-sha256-v1':
            raise ValueError('Unexpected exclusion normalization')
        seen.update(bytes.fromhex(h) for h in manifest['hashes'])
        if any(len(h) != 32 for h in seen):
            raise ValueError('Invalid exclusion SHA256')
    completed = {}
    totals = collections.Counter()
    for part in sorted(a.out.glob('part-*')):
        if part.name.endswith('.pending'):
            continue
        m = json.loads((part/'manifest.json').read_text())
        for s in splits:
            if (part/(s+'.bin')).stat().st_size != m['counts'].get(s, 0)*4:
                raise ValueError('Committed token shard size mismatch')
        with (part/'docs.jsonl').open() as f:
            for line in f:
                key = bytes.fromhex(json.loads(line)['hash'])
                if key in seen:
                    raise ValueError('Duplicate committed document')
                seen.add(key)
        completed[(m['file'], m['row_group'])] = m['input_sha256']
        totals.update(m['counts'])
    tok = AutoTokenizer.from_pretrained(a.tokenizer, local_files_only=True, trust_remote_code=False)
    if not tok.is_fast or tok.eos_token_id is None or len(tok) >= 2**32:
        raise ValueError('Fast uint32-compatible tokenizer with EOS required')
    started = time.monotonic()
    new_groups = 0
    for path in paths:
        name = path.relative_to(a.data).as_posix()
        print(json.dumps(dict(stage='checking_input', file=name)), flush=True)
        fingerprint = sha(path)
        pf = pq.ParquetFile(path)
        for rg in range(pf.num_row_groups):
            if (name, rg) in completed:
                if completed[(name, rg)] != fingerprint:
                    raise ValueError('Committed input changed')
                continue
            began = time.monotonic()
            texts = pf.read_row_group(rg, columns=['text']).column('text').to_pylist()
            counts = collections.Counter()
            rejected = collections.Counter()
            buffers = {s: [] for s in splits}
            rows = []
            for offset in range(0, len(texts), 128):
                candidates = []
                for row, text in enumerate(texts[offset:offset+128], offset):
                    if not text or not text.strip():
                        rejected['empty'] += 1
                        continue
                    key = doc_hash(text)
                    raw_key = bytes.fromhex(key)
                    if raw_key in seen:
                        rejected['duplicate_or_old'] += 1
                        continue
                    seen.add(raw_key)
                    candidates.append((row, text, key, split_for(key)))
                ids = tok([c[1] for c in candidates], add_special_tokens=False,
                          truncation=False, return_attention_mask=False)['input_ids'] if candidates else []
                for (row, text, key, s), tokens in zip(candidates, ids):
                    tokens.append(tok.eos_token_id)
                    rows.append(dict(hash=key, split=s, row=row, offset=counts[s], length=len(tokens)))
                    buffers[s].extend(tokens)
                    counts[s] += len(tokens)
            part = a.out/('part-'+hashlib.sha256(name.encode()).hexdigest()[:12]+f'-{rg:05d}')
            temp = a.out/(part.name+'.pending')
            if temp.exists():
                shutil.rmtree(temp)  # Only this program's uncommitted exact staging path.
            temp.mkdir()
            for s in splits:
                np.asarray(buffers[s], dtype='<u4').tofile(temp/(s+'.bin'))
            with (temp/'docs.jsonl').open('w') as f:
                for row in rows:
                    f.write(json.dumps(row)+'\n')
            m = dict(file=name, row_group=rg, input_sha256=fingerprint, counts=dict(counts),
                     documents=len(rows), input_documents=len(texts), rejected=dict(rejected),
                     seconds=time.monotonic()-began)
            save(temp/'manifest.json', m)
            os.replace(temp, part)
            completed[(name, rg)] = fingerprint
            totals.update(counts)
            new_groups += 1
            status = dict(stage='tokenize', counts=dict(totals), parts=len(completed),
                          last_group=m, elapsed_seconds=time.monotonic()-started)
            save(a.out/'status.json', status)
            print(json.dumps(status), flush=True)
            if a.max_groups and new_groups >= a.max_groups:
                print('Bounded test complete; safe to resume', flush=True)
                return
    result = dict(counts=dict(totals), parts=len(completed), files=len(paths))
    save(a.out/'COMPLETE.json', result)
    print(json.dumps(dict(stage='complete', **result)), flush=True)


if __name__ == '__main__':
    main()
