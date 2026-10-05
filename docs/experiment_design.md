# Reproduction and independent experiment design

## Three different reproduction targets

1. **Rebuild published numbers/figures:** no training, no external data, no checkpoints; use public JSON/CSV.
2. **Rerun the small models and diagnostics:** obtain the pinned public corpus/tokenizer, prepare deterministic splits, train from scratch, run the included evaluator/diagnostics. This is substantial GPU work, not an instant demo.
3. **Independently replicate at 4B+:** port the audited Megatron patch/config to your environment and declare a matched dataset. The historical DCLM preprocessed input is not distributed, and its original download revision is unverified. Do not claim bit-identical 4B reproduction from this package alone.

## Models below 4B: environment and architecture

Historical versions: Python 3.11, PyTorch 2.11.0+cu128, Transformers 5.16.1, NumPy 2.4.6; dependencies in [requirements](../requirements.txt). Install the PyTorch build suited to your GPUs separately.

| Setting |57M|157M|512M (“450M” config)|
|---|---:|---:|---:|
| Trainable parameters |56,836,352|157,433,856|512,416,576|
| Transformer layers |12|18|28|
| Width |512|816|1280|
| Attention heads |8|6|8|
| MLA KV/query compression |64/128|64/192|112/320|
| Sequence length |512|1024|1024|
| Global sequence batch |256|64|64|
| Original per-GPU micro-batch |64|16|16|
| Effective training labels |999,038,753|3,200,000,000|10,000,000,000|
| Updates |9500|74,992|234,011|
| Held-out test labels |4,995,188|5,000,000|5,000,000|

Both use SquaredReLU FFNs (hidden size rounded to a multiple of 64), RoPE, tied GPT-2 vocabulary embeddings, no dropout, seed 42, AdamW peak LR 0.0003, betas 0.9/0.95, epsilon 1e-8, gradient clipping 1, cosine decay to 0.1× peak LR, 2% warmup. Weight decay 0.1 applies to tensors with at least two dimensions; one-dimensional parameters have no decay. BF16 autocast, FP32 master parameters and activation recomputation are used. The objective is globally summed cross entropy divided by the **number of valid labels**, not the average of arbitrarily padded sequence losses.

The 57M model is a custom reduction of the Brújula architecture, not an official Moonshot model. 157M uses the publisher configuration, with random weights and no YaRN extension. The experiment historically called “450M” follows the Brújula 450M configuration but has 512,416,576 trainable parameters in this implementation. The model cards' original training results are unrelated to ours. The source code hashes match the pinned HF revision recorded in [provenance](../provenance/upstream.json).

### Data preparation: historical 57M

Fetch the tokenizer and the first two source files used by the historical bounded preparation:

```bash
python scripts/fetch_tokenizer.py
hf download HuggingFaceFW/fineweb-edu --repo-type dataset \
  --revision 87f09149ef4734204d70ed1d046ddc9ca3f2b8f9 \
  --include 'sample/100BT/000_00000.parquet' \
  --include 'sample/100BT/000_00001.parquet' \
  --local-dir data/fineweb-edu-100bt
python small/prepare_pretrain.py --data data/fineweb-edu-100bt \
  --tokenizer vendor/brujula --out data/prepared-57m --local-only \
  --max-files 4 --train-tokens 1000000000 --heldout-tokens 5000000
```

Use repeated `--include` flags (not an extra positional filename). Upstream dataset terms apply: see [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu). No token corpus is included here.

Preparation preserves document text for tokenization (no chat template/BOS), appends EOS and stores little-endian uint32 shards. NFKC/whitespace-normalized text is used **only** for SHA256 deduplication/splitting. First 16 hash hex digits modulo 1000 allocate buckets 0–9 to validation, 10–19 to test, 20–999 to train. Previously used calibration/evaluation documents are excluded via [hashes](../evidence/data/exclusion-hashes.json), without distributing their text. Exact duplicates are filtered; near duplicates are not independently removed.

Whole documents are selected until each token target is reached. The loader then creates document-isolated causal chunks with one shared context token at adjacent chunk boundaries. A document of length $L$ yields $L-1$ labels. Thus the nominal 1B token preparation is not exactly 1B effective labels. Do not relabel 999,038,753 as an exact billion.

Row-group part names contain the SHA256 prefix of the source filename. Training **sorts part names**, not source filenames, before indexing documents, and uses `numpy.random.default_rng(seed).permutation` over chunk records. Preserve that ordering. Historical part/source checksums are archived in [57M source parts](../evidence/data/57m/source-parts.json). Runtime/timing/path metadata differs after portability edits, so metadata fingerprints need not be identical; verify source bytes, document selections and effective-label counts. A different metadata hash by itself is not proof of different tokens.

### Data preparation: historical 157M

The original 157M selected prefix was drawn after tokenizing the entire 140-file 100BT source. Because hashed part-name order differs from source-name order, simply tokenizing the first few source files does not reproduce the selected dataset.

```bash
hf download HuggingFaceFW/fineweb-edu --repo-type dataset \
  --revision 87f09149ef4734204d70ed1d046ddc9ca3f2b8f9 \
  --include 'sample/100BT/*.parquet' --local-dir data/fineweb-edu-100bt
python small/prepare_full_corpus.py --data data/fineweb-edu-100bt \
  --tokenizer vendor/brujula --out data/full-corpus --expected-files 140
python small/build_training_index.py --source data/full-corpus \
  --out data/prepared-157m --train-tokens 3200000000 \
  --eval-tokens 5000000 --seq-len 1024
```

This reference tokenizer is deterministic and resumable at row-group boundaries, not a throughput-optimized distributed preprocessing service. It can require substantial disk/RAM/time. The generated token/index bytes can be compared with [historical hashes](../evidence/data/157m/indexed.json). A fresh prefix under a cheaper preprocessing protocol is a valid **independent** test, but must be reported as different data rather than an exact rerun.

### Data preparation: historical 512M / “450M” configuration

The 512M experiment uses the same fully tokenized FineWeb-Edu 100BT source and deterministic indexing procedure as the 157M experiment, but selects 10B training labels at sequence length 1024. Build the full corpus as above, then run:

```bash
python small/build_training_index.py --source data/full-corpus \
  --out data/prepared-450m --train-tokens 10000000000 \
  --eval-tokens 5000000 --seq-len 1024
```

The historical train, validation and test fingerprints are retained in each public 450M run config. Match those fingerprints for an exact data rerun; otherwise report the result as an independent replication on a different deterministic prefix.

### Training and evaluation

```bash
for variant in full block rms_block; do
  python small/run.py --size 57m --variant "$variant" \
    --prepared data/prepared-57m --out "runs/57m-$variant" \
    --world-size 4 --micro-batch 64 --global-batch 256
done
```

For 157M, change size/prepared/output and use micro-batch 16/global batch 64. For the 512M-parameter configuration, use `--size 450m`, the 10B-label prepared index, micro-batch 16 and global batch 64.

The trainer emits `config.json`, `runtime.json`, `train.jsonl`, periodic validation JSON, `last.pt`, `test.json` and `COMPLETE.json`. It evaluates validation every 500 updates and at the last update; test once at the end. It retains only the latest checkpoint by default. A validation curve does not imply every model checkpoint has been retained. The same evaluator is used for all variants. `--stop-after-steps` records PAUSED, not COMPLETE. Resume requires `--resume`, preserves the optimizer/RNG/step and rejects a changed config. Do not alter a historical run to force a resume.

Batch-size profiling should be bounded and performed before a comparison suite. Maximize feasible throughput on the target hardware, and then use the **same global batch** for all arms. If micro-batch changes but global batch is preserved, record the changed accumulation/execution. If global batch changes, the number of updates and optimization process change; it is a new setting. Do not compare such a run as though only the residual method changed.

### Frozen-checkpoint diagnostics

Create a JSON mapping with your own run directories, for example:

```json
{"57m":{"full":"runs/57m-full","block4":"runs/57m-block","norm_rms":"runs/57m-rms_block"}}
```

Then:

```bash
python scripts/analyze_checkpoints.py --runs-json runs-map.json \
  --sizes 57m --prepared data/prepared-57m --out runs/diagnostics-57m \
  --sequences 64 --batch-size 64 --seed 260926
```

157M uses 32 sequences/batch 32. Keep one batch for exact quantiles, or reduce the sample count and label it accordingly. The public script rejects averaging batch quantiles. It records model/data fingerprints, selected chunk indices, per-source RMS, sampled attention/effective coefficients, formula parity checks and positive multipliers 0.25/0.5/2/4. The measured representation is the **next aggregator output**, before applying the downstream sublayer, not the final logits. The diagnostic covers Full, Block4 and RMS Block4.

The checkpoint must match the run's completion record, and the data fingerprint must match its test input. A path/timing-only metadata change may require `--allow-data-mismatch` after you inspect the content hashes; that option records both fingerprints and does not pretend they match. Use an empty output directory. Multiplication by one appears as an exact identity anchor in plots, not an additional measured sample.

## Megatron / 4B scaling

The reference GPU configuration uses 64 GPUs across 8 nodes with TP2/DP32/PP1/CP1, sequence parallelism, BF16 and a distributed optimizer. Profile the micro-batch size on the target GPU type before launching the full comparison, then keep the topology and global batch identical across variants.

Fetch the base source and inspect/apply the patch:

```bash
git clone https://github.com/NVIDIA/Megatron-LM.git megatron
git -C megatron checkout 64d156734918e73238de5cad6d97dcfaa6ec1005
git -C megatron apply --check ../scaling/megatron-attnres.patch
git -C megatron apply ../scaling/megatron-attnres.patch
```

The patch changes 13 source files, including the residual state/reader and recomputation/gradient diagnostics. Keep upstream license notices and inspect [the changed-file manifest](../provenance/megatron-modified-files.json).

Use [model config](../scaling/model-config.json), the [audited arguments](../evidence/4b/full/arguments.txt) and the generic generator:

```bash
python scaling/launch.py --variant rms_block --megatron-root megatron \
  --data-prefix data/dclm_text_document --tokenizer-dir data/qwen3-tokenizer \
  --out runs/4b-rms-block --master-addr YOUR_RENDEZVOUS_HOST --node-rank 0
```

It prints only unless `--execute` is added. Launch one process group per node through your own scheduler, supply the correct node rank and rendezvous settings, prepare the compatible environment/dataset/tokenizer, and test a bounded run first. The generator refuses implicit topology changes and existing checkpoint outputs. It explicitly disables experimental tied-gradient workarounds and enables the common gradient-bucket diagnostic policy used by the formal runs.

Backbone: 36 layers, width 2560, 32 query/8 KV heads, head dimension 128, FFN 9728, SwiGLU, QK RMSNorm, tied embeddings, padded vocabulary 151936, no bias/dropout, RoPE base 1e6. The original config reference is Qwen/Qwen3-4B revision recorded in the config; no pretrained weights were loaded.

Training uses sequence 8192, global batch 256, micro-batch 1/DP rank, accumulation 8, seed 42, AdamW peak LR 1e-4/min LR 1e-5, betas 0.9/0.95, epsilon 1e-8, weight decay 0.1 and clip 1. The cosine schedule is 47,684 updates (100,000,595,968 token positions) with 1% warmup; all three original runs reached their target stop at 9537 updates (20,000,538,624 positions). **This is an early segment of a 100B schedule**, not a 20B cosine run. The last scheduled validation at step 9500 follows 19,922,944,000 training positions.

Data is a shared locally indexed DCLM shard, tokenizer preparation appended EOD, with Megatron split 98/2/0. Loss does not mask EOD; attention/position IDs are not reset at document boundaries. Thus this is not the small-model document-isolated protocol. Cached train/valid descriptions match across variants. Original corpus download revision, preprocessing-base commit and a public reconstruction manifest were not established by this audit. Supply your own documented DCLM manifest for independent replication; do not guess those missing details from filenames.

No independent test split exists. Validation uses 10 iterations every 250 steps with an advancing loader, so equal-step variants share evaluation data but different curve points need not. All final 4B test claims remain unavailable. The headline table uses the last common scheduled validation step, 9500, deterministically; the runs then completed and saved at 9537. The values are logged online validation, not a fresh evaluation of the 9537 checkpoints. The earlier 8750 comparison is retained as a separately labeled audit snapshot.

## Independent 4B+ replication

We welcome independent results, including reversals and null effects. A useful submission records:

- Full, Block and RMS Block, with the same model, optimizer, data and training budget.
- Identical initialization protocol, data order, token/label budget, sequence length, optimizer/schedule and global batch; matched topology/execution within the suite.
- Multiple seeds and several block sizes; uncertainty across **runs**, not treated as independent per-token points.
- Full validation curves, final held-out NLL/PPL, throughput/memory and checkpoint/metric hashes; failed/resumed runs reported, not silently omitted.
- Local scaling counterfactuals and optional magnitude statistics, separated from claims about training causality.

Useful follow-ups include different block sizes, more capacity, longer training and additional corpora.

## Mechanism study: raw versus normalized block summaries

### Falsifiable claims

The primary claim to test is: **pre-sum normalization helps because raw activation RMS is not a reliable within-block coefficient for preserving directions useful to later readers.** This claim fails if raw summaries consistently approximate Full block contributions better and small-source interventions have proportionally small loss effects.

A secondary claim is: **the problem is mostly a stable attention/FFN or layer-wise scale mismatch.** This claim passes if dividing each source by a fixed calibration RMS for its layer and source type recovers most of the RMS Block benefit. It fails if per-token normalization is still materially better.

### Stage 1: conditional Full routing census

For every sampled valid token, later reader and completed four-source block, record source layer, attention/FFN type, position in the block, reader distance, $r_i$, $\alpha_i$ and $\alpha_i r_i$. Center $\log r_i$ and $\log\alpha_i$ within the same token/reader/block before fitting effects. This avoids confusing softmax normalization, source count, recency and depth with source type.

Report attention-versus-FFN paired differences in $\log r$, $\log\alpha$ and $\log(\alpha r)$ by source layer and reader distance. A larger attention $\alpha$ supports observational compensation; it is not direct magnitude sensing because the Key is normalized.

### Stage 2: vector approximation of Full

For each token, reader and block, compute

$$
C=\sum_i\alpha_i r_i u_i,\qquad b_{\rm raw}=\sum_i r_i u_i,\qquad b_{\rm rms}=\sum_i u_i.
$$

Fit the best scalar separately for each candidate, $\beta^*(b)=\langle C,b\rangle/\lVert b\rVert_2^2$, and report

$$
E(b)=\frac{\lVert C-\beta^*(b)b\rVert_2}{\lVert C\rVert_2}.
$$

This compares summary direction without penalizing either candidate for an arbitrary overall scale. Report paired $E(b_{\rm raw})-E(b_{\rm rms})$, cosine similarity, and results stratified by attention/FFN RMS ratio, pairwise directional cosine and reader distance. RMS summary is supported as an approximation of Full only when its paired error is reliably lower.

### Stage 3: frozen-model loss interventions

Routing weight is not enough, so evaluate held-out NLL after source-specific Value interventions. Keep the original Full routing weights fixed for the first test and set one Value contribution to zero, rescale it, or replace its radial coefficient by the layer/type calibration RMS. Then repeat with routing recomputed to expose secondary routing effects. Record $\Delta$NLL, representation change and logit change for attention and FFN sources separately.

The key test is whether low-$r_i$ sources have larger loss effects than their raw coefficient predicts, especially after controlling for layer, reader distance and direction overlap. If not, the “small sources are underrepresented” explanation is weakened even if RMS Block trains better.

### Stage 4: minimal training controls

Only after the frozen tests, run matched 57M training with:

- raw Block: $\sum_i r_i u_i$;
- RMS Block: $\sum_i u_i$;
- fixed layer/type calibration: $\sum_i y_i/m_{\ell(i),t(i)}$, where $m_{\ell,t}$ is the median source RMS for layer $\ell$ and module type $t$;
- exponent interpolation: $\sum_i r_i^\gamma u_i$ for a small prespecified set between $\gamma=0$ and $\gamma=1$;
- post-sum normalization of the raw summary, which removes completed-block scale but preserves raw within-block direction.

Use the same initialization protocol, data order, global batch and token budget. For the exponent sweep, also report a version whose completed-summary RMS is matched to raw Block so that within-block direction and block-level Value magnitude are not conflated.

### Decision rules

- **Supports within-block directional balancing:** RMS summary has lower oracle vector error; low-RMS sources have nontrivial loss effects; $\gamma<1$ improves matched training.
- **Supports stable module-scale calibration instead:** fixed layer/type calibration matches per-token RMS normalization.
- **Supports block-level scale removal instead:** post-sum normalization works while pre-sum equalization does not.
- **Insufficient evidence:** observational $\alpha$ differences without vector or loss evidence, or single-run training differences without matched controls.
