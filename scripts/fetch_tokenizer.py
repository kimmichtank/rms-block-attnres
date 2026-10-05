#!/usr/bin/env python3
"""Download pinned tokenizer ONLY; verify bytes. No pretrained weight loading."""
from pathlib import Path
import hashlib
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
REV='f9c4fb962203259992e847217f49cca3412e4f19'
EXPECTED={'tokenizer.json':'1fe93b6152957cf9cfd6d89002467f789ce8b3f3e000b3a2edf27c808ddd0b9e',
          'tokenizer_config.json':'17e312e0087310b49a0884cd2c9c96876dee84a7dd9ccced056abf98331f0f25'}
for name,digest in EXPECTED.items():
 p=ROOT/'vendor/brujula'/name
 if p.exists():
  data=p.read_bytes()
  if hashlib.sha256(data).hexdigest()!=digest:raise ValueError('Existing tokenizer mismatch: '+name)
  continue
 data=urllib.request.urlopen(f'https://huggingface.co/Sakatepon/Brujula-150M-32K-chat/resolve/{REV}/{name}',timeout=60).read()
 if hashlib.sha256(data).hexdigest()!=digest:raise ValueError('Download checksum mismatch: '+name)
 p.write_bytes(data)
print('Pinned tokenizer verified; no model weights downloaded.')
