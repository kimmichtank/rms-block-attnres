# Generated results

Rebuild with `python scripts/rebuild.py --figures`.

## Models below 4B — final held-out test, seed 42

| Model | Variant | Labels | NLL | PPL |
|---|---|---:|---:|---:|
| 57m | Full | 999,038,753 | 3.632467004 | 37.80597 |
| 57m | Block4 | 999,038,753 | 3.665351609 | 39.06987 |
| 57m | RMS Block4 | 999,038,753 | 3.650837322 | 38.50690 |
| 57m | Full+RMS | 999,038,753 | 3.631821781 | 37.78158 |
| 157m | Full | 3,200,000,000 | 3.082830330 | 21.82007 |
| 157m | Block4 | 3,200,000,000 | 3.083475714 | 21.83416 |
| 157m | RMS Block4 | 3,200,000,000 | 3.077987996 | 21.71467 |
| 512m | Full | 10,000,000,000 | 2.716601201 | 15.12881 |
| 512m | Block4 | 10,000,000,000 | 2.733077455 | 15.38015 |
| 512m | RMS Block4 | 10,000,000,000 | 2.723055271 | 15.22677 |

57M Full+RMS is essentially tied with Full at the endpoint; DDP versus accumulation mismatch.

## 4B — completed 9,537-step runs; last scheduled validation at step 9,500 (no test split)

| Variant | Step | Token positions | NLL | PPL |
|---|---:|---:|---:|---:|
| Block4 | 9500 | 19,922,944,000 | 2.978657 | 19.66140 |
| Full | 9500 | 19,922,944,000 | 2.954836 | 19.19858 |
| RMS Block4 | 9500 | 19,922,944,000 | 2.932211 | 18.76909 |

## 4B — RMS Block versus Block at the same validation NLL

At Block's last scheduled NLL (2.978657), RMS Block reaches the same loss at an interpolated step 7528.9 instead of 9500.

| Data-efficiency speedup | Token reduction | RMS Block throughput / Block | Time-to-quality speedup | Training-time reduction |
|---:|---:|---:|---:|---:|
| 1.262x | 20.75% | 0.9745 | 1.230x | 18.67% |
