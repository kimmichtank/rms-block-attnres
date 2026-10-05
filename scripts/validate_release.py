#!/usr/bin/env python3
"""Offline traceability, numerical consistency and accidental-private-data checks."""
import ast
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[1]
errors=[]
def require(ok,message):
 if not ok:errors.append(message)

manifest=json.loads((ROOT/'evidence/source-manifest.json').read_text())
for sid,record in manifest.items():
 p=ROOT/record['public_file'] if 'public_file' in record else None
 if p:
  require(p.exists(),f'Missing artifact {sid}')
  if p.exists():require(hashlib.sha256(p.read_bytes()).hexdigest()==record['public_sha256'],f'Changed evidence: {sid}')
for claim in json.loads((ROOT/'evidence/claims.json').read_text()):
 for p in claim['sources']:require((ROOT/p).is_file(),f"Missing claim source {claim['id']}: {p}")
runs=json.loads((ROOT/'evidence/small/runs.json').read_text())
for r in runs:
 require(abs(math.exp(r['test_nll'])-r['test_ppl'])<1e-9,'PPL/NLL disagreement '+r['run_id'])
 require(r['seed']==42,'Unexpected seed '+r['run_id'])
 run=ROOT/'evidence/small'/r['run_id']
 test=json.loads((run/'test.json').read_text())
 complete=json.loads((run/'COMPLETE.json').read_text())
 for summary_key,raw_key in [('test_nll','nll'),('test_ppl','ppl'),('test_labels','tokens')]:
  require(r[summary_key]==test[raw_key],f"Small run table differs from test JSON: {r['run_id']} {summary_key}")
 require(complete['test']==test,'Completion/test disagreement '+r['run_id'])
 for summary_key,raw_key in [('steps','steps'),('labels','tokens'),('parameters','parameters')]:
  require(r[summary_key]==complete[raw_key],f"Small run table differs from completion: {r['run_id']} {summary_key}")
 for row in csv.DictReader((run/'validation.csv').open()):
  raw_path=run/f"validation-{int(row['step']):06d}.json"
  if raw_path.exists():
   raw=json.loads(raw_path.read_text())
   require(float(row['nll'])==raw['nll'] and float(row['ppl'])==raw['ppl'],f"Validation CSV/JSON disagreement: {r['run_id']} {row['step']}")
  else:
   source=manifest.get(row.get('source_id',''),{})
   require(source.get('original_sha256')==row.get('source_sha256'),f"Validation source hash disagreement: {r['run_id']} {row['step']}")
   require(abs(math.exp(float(row['nll']))-float(row['ppl']))<1e-9,f"Validation PPL/NLL disagreement: {r['run_id']} {row['step']}")
for size in ('57m','157m','512m'):
 configs=[json.loads((ROOT/'evidence/small'/r['run_id']/'config.json').read_text()) for r in runs if r['size']==size]
 for c in configs[1:]:
  for k in ['data_fingerprint','global_batch','steps','effective_train_labels','seq_len','lr','warmup_fraction']:
   require(c[k]==configs[0][k],f'Unmatched {size}: {k}')
matched=json.loads((ROOT/'evidence/4b/matched-comparison.json').read_text())
common=None
for r in matched['rows']:
 curve={int(x['step']):x for x in csv.DictReader((ROOT/'evidence/4b'/r['run_id']/'validation.csv').open())}
 excerpts=json.loads((ROOT/'evidence/4b'/r['run_id']/'validation-excerpts.json').read_text())
 for item in excerpts:
  m=re.search(r'iteration\s+(\d+)\s*\|\s*lm loss value:\s*(\S+)\s*\|\s*lm loss PPL:\s*(\S+)',item['text'])
  require(m is not None,'Unparseable original validation excerpt')
  if m:
   extracted=curve[int(m[1])]
   require(float(extracted['nll'])==float(m[2]) and float(extracted['ppl'])==float(m[3]),'4B CSV differs from original log excerpt')
   require(item['source_id']==extracted['source_id'] and str(item['line'])==extracted['line'],'4B excerpt provenance mismatch')
 common=set(curve) if common is None else common & set(curve)
 row=curve[matched['selected_step']]
 for k in ('step','token_positions','nll','ppl','source_id','line'):
  require(str(r[k])==row[k],f"4B matched table/CSV disagreement: {r['run_id']} {k}")
 require(r['source_id'] in manifest,'Unknown 4B original-log source ID')
 require(abs(math.exp(r['nll'])-r['ppl'])/r['ppl']<1e-6,'4B log-precision NLL/PPL disagreement')
require(matched['selected_step']==max(common),'4B selection is not the largest common validation step')
require(matched['selected_step']==9500,'4B last scheduled validation is not step 9500')
require(matched['stop_token_positions']==9537*256*8192,'4B completed training budget mismatch')
inventory=json.loads((ROOT/'evidence/4b/run-inventory.json').read_text())['runs']
formal={r['run_id']:r for r in inventory if r['run_id'] in {x['run_id'] for x in matched['rows']}}
require(len(formal)==3,'Three formal 4B runs required')
for rid,record in formal.items():
 require(record['status']=='completed_target_stop' and record['last_logged_step']==9537 and 9537 in record['saved_checkpoint_steps'],f'4B incomplete target run: {rid}')
 require(record['last_validation_step']==9500 and record['resumed'] is False,f'4B comparison status mismatch: {rid}')
 require(all(s<=9537 for s in record['saved_checkpoint_steps']),f'Post-target checkpoint included: {rid}')
 events=json.loads((ROOT/'evidence/4b'/rid/'events.json').read_text())
 require(all('9540' not in event['text'] for event in events),f'Post-target event included: {rid}')
for p in ROOT.rglob('*'):
 if not p.is_file() or any(x in p.parts for x in ['__pycache__','.git','.venv','test-output']):continue
 if p.suffix in ['.png','.npz']:continue
 try:s=p.read_text()
 except UnicodeError:continue
 # Generic path/credential checks do not embed any private host names themselves.
 for pattern in [r'/(?:Users|home|mnt|shared)/[A-Za-z0-9_]',r'hf_[A-Za-z0-9]{25,}',r'-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----']:
  require(not re.search(pattern,s),f'Possible private material: {p.relative_to(ROOT)}')
 if p.suffix=='.py':
  try:ast.parse(s)
  except SyntaxError as e:errors.append(f'Syntax: {p}: {e}')
 if p.suffix=='.md':
  require('\\(' not in s and '\\)' not in s,f'Document inline math style: {p.relative_to(ROOT)}')
  for link in re.findall(r'\]\(([^)]+)\)',s):
   if link.startswith(('http:','https:','#','mailto:','{{')):continue
   target=link.split('#')[0]
   require((p.parent/target).exists(),f'Broken local link in {p.relative_to(ROOT)}: {target}')
if errors:
 print('\n'.join(errors));sys.exit(1)
print(f'PASS: {len(manifest)} traceable sources, {len(runs)} completed runs below 4B, syntax/links/privacy checks.')
