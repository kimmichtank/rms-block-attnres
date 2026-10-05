# Evidence directory

This directory contains the machine-readable inputs used to rebuild the tables and figures. It does not contain checkpoints or training corpora.

## Layout

- `small/`: training summaries for the 57M, 157M and 512M-parameter experiments. Each run contains its configuration, completion record, selected training-loss points, the complete validation curve, and final test result. The 512M runs retain the historical `450m_*` identifiers because the source project called this the 450M configuration. `runs.json` is the compact cross-run index.
- `450m/`: a path-redacted matched-run summary for the 512,416,576-parameter experiment. Detailed per-run curves live under `small/450m_*`.
- `4b/`: evidence for the three completed 4B runs. `full/`, `block/`, and `rms_block/` contain resolved arguments, selected training points, validation curves, checkpoint inventories, and redacted log excerpts. `matched-comparison.json` selects the common validation point used in the result table; `step-time-summary.json` records the stable-interval throughput statistics and original log hashes used for the same-loss efficiency calculation; `run-inventory.json` lists the three public runs. `4b/data/` contains the train/validation dataset descriptions recorded by Megatron.
- `diagnostics/`: frozen-checkpoint measurements used for the mechanism analysis. `residual_rms.csv` records layer-output magnitudes; `counterfactual.csv` records the source-rescaling intervention; `summary.json` contains derived statistics; `manifest.json` records the evaluated checkpoints and sampled data.
- `data/`: small-model data provenance, not the dataset itself. It records preparation settings, source-file hashes, indexed-data hashes, selected input plans, and hashes of documents excluded from preparation.
- `claims.json`: machine-readable mapping from each public claim to the evidence files that support it.
- `source-manifest.json`: hashes of the original audited sources and their public redacted exports.

## What helps reproduction?

The files under `data/` help verify that a newly prepared dataset follows the same selection, exclusion, ordering, and indexing rules. They do not replace the FineWeb-Edu source files or contain training text. For 157M, the indexed-data hashes permit byte-level comparison after preprocessing. For 57M, the source-part and input-plan records identify the historical bounded preparation. The 512M run uses the same indexed FineWeb-Edu corpus and fingerprints recorded in its public configs. The 4B dataset descriptions establish that the three compared runs used the same indexed stream, but they are not sufficient to reconstruct the original DCLM shard by themselves.

Some `source_id` fields inside the 4B CSV/JSON files still contain identifiers such as `4b_07_block`. These are immutable provenance labels from the audit and point to exact original log hashes and line numbers; they are not additional public runs or directories.
