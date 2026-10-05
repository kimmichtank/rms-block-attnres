#!/usr/bin/env python3
"""Measure residual magnitudes, true AttnRes coefficients, and scale counterfactuals."""
from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "small"))
from pretrain_core import DocumentChunks, make_model  # noqa: E402


SPECS = {}  # Populated by --runs-json. See docs/experiment_design.md.


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def quantiles(values):
    a = np.asarray(values, dtype=np.float64)
    if not a.size:
        return {"count": 0}
    q = np.quantile(a, [0.1, 0.25, 0.5, 0.75, 0.9, 0.99])
    return {
        "count": int(a.size), "mean": float(a.mean()), "std": float(a.std()),
        "p10": float(q[0]), "p25": float(q[1]), "median": float(q[2]),
        "p75": float(q[3]), "p90": float(q[4]), "p99": float(q[5]),
    }


def rankdata(a):
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=np.float64)
    sorted_a = a[order]
    starts = np.r_[0, np.flatnonzero(sorted_a[1:] != sorted_a[:-1]) + 1]
    ends = np.r_[starts[1:], len(a)]
    for s, e in zip(starts, ends):
        ranks[order[s:e]] = (s + e - 1) / 2.0
    return ranks


def correlation_summary(r, alpha, effective):
    r = np.asarray(r, dtype=np.float64)
    alpha = np.asarray(alpha, dtype=np.float64)
    effective = np.asarray(effective, dtype=np.float64)
    keep = (r > 0) & (alpha > 0) & np.isfinite(r) & np.isfinite(alpha)
    x, y = np.log(r[keep]), np.log(alpha[keep])
    if len(x) < 3:
        return {"count": int(len(x))}
    pearson = float(np.corrcoef(x, y)[0, 1])
    spearman = float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])
    slope = float(np.polyfit(x, y, 1)[0])
    return {
        "count": int(len(x)), "pearson_log_r_log_alpha": pearson,
        "spearman_log_r_log_alpha": spearman,
        "slope_log_alpha_on_log_r": slope,
        "r": quantiles(r[keep]), "alpha": quantiles(alpha[keep]),
        "directional_coefficient": quantiles(effective[keep]),
    }


def masked_numpy(x, mask, max_items=None):
    a = x.detach().float()[mask].cpu().numpy().reshape(-1)
    if max_items and len(a) > max_items:
        take = np.linspace(0, len(a) - 1, max_items, dtype=np.int64)
        a = a[take]
    return a


def rms(x):
    return x.float().square().mean(-1).sqrt()


def normalized(x):
    y = x.float()
    return (y * torch.rsqrt(y.square().mean(-1, keepdim=True) + 1e-6)).to(x.dtype)


def aggregate(agg, sources):
    v = torch.stack(sources, dim=0)
    k = agg.key_norm(v)
    logits = torch.einsum("d,lbtd->lbt", agg.w, k)
    weights = torch.softmax(logits, dim=0)
    h = (weights.unsqueeze(-1) * v).sum(0)
    return h, weights


class Recorder:
    def __init__(self, size, variant, mask, c_values):
        self.size, self.variant, self.mask = size, variant, mask
        self.c_values = c_values
        self.raw_values = []
        self.raw_r = []
        self.output_stats = defaultdict(list)
        self.pair_r, self.pair_alpha, self.pair_effective = [], [], []
        self.within_block_ratio = []
        self.within_block_log_std = []
        self.within_block_raw_ratio = []
        self.within_block_alpha_ratio = []
        self.within_block_compensation_ratio = []
        self.within_block_compensation_slope = []
        self.cf = defaultdict(list)
        self.max_aggregate_error = 0.0
        self.read_count = 0

    def append(self, value, kind):
        idx = len(self.raw_values)
        self.raw_values.append(value)
        ri = rms(value)
        self.raw_r.append(ri)
        self.output_stats[(idx, kind)].append(masked_numpy(ri, self.mask))
        return idx

    def read(self, agg, sources, groups, normalized_groups, read_kind, pending_cf=None):
        h, weights = aggregate(agg, sources)
        exact = agg(sources)
        err = float((h - exact).abs().max())
        self.max_aggregate_error = max(self.max_aggregate_error, err)
        self.read_count += 1

        member_alpha, member_normed = {}, {}
        for source_index, members in enumerate(groups):
            if source_index == 0:
                continue
            a = weights[source_index]
            is_normed = normalized_groups[source_index]
            for member in members:
                member_alpha[member] = a
                member_normed[member] = is_normed
                ri = self.raw_r[member]
                contribution_r = rms(normalized(self.raw_values[member])) if is_normed else ri
                eff = a * contribution_r
                self.pair_r.append(masked_numpy(ri, self.mask, 4096))
                self.pair_alpha.append(masked_numpy(a, self.mask, 4096))
                self.pair_effective.append(masked_numpy(eff, self.mask, 4096))

        complete = (len(self.raw_values) // 4) * 4
        for start in range(0, complete, 4):
            members = list(range(start, start + 4))
            if not all(i in member_alpha for i in members):
                continue
            vals = []
            raw_vals = []
            alpha_vals = []
            for i in members:
                scale = rms(normalized(self.raw_values[i])) if member_normed[i] else self.raw_r[i]
                vals.append(member_alpha[i] * scale)
                raw_vals.append(self.raw_r[i])
                alpha_vals.append(member_alpha[i])
            e = torch.stack(vals, dim=0).float().clamp_min(1e-30)
            rr = torch.stack(raw_vals, dim=0).float().clamp_min(1e-30)
            aa = torch.stack(alpha_vals, dim=0).float().clamp_min(1e-30)
            ratio = e.max(0).values / e.min(0).values
            raw_ratio = rr.max(0).values / rr.min(0).values
            alpha_ratio = aa.max(0).values / aa.min(0).values
            log_std = torch.log(e).std(0, unbiased=False)
            compensation_ratio = ratio / raw_ratio
            log_r = torch.log(rr)
            log_a = torch.log(aa)
            centered_r = log_r - log_r.mean(0, keepdim=True)
            centered_a = log_a - log_a.mean(0, keepdim=True)
            slope = (centered_r * centered_a).mean(0) / centered_r.square().mean(0).clamp_min(1e-20)
            self.within_block_ratio.append(masked_numpy(ratio, self.mask))
            self.within_block_log_std.append(masked_numpy(log_std, self.mask))
            self.within_block_raw_ratio.append(masked_numpy(raw_ratio, self.mask))
            self.within_block_alpha_ratio.append(masked_numpy(alpha_ratio, self.mask))
            self.within_block_compensation_ratio.append(masked_numpy(compensation_ratio, self.mask))
            self.within_block_compensation_slope.append(masked_numpy(slope, self.mask))

        if pending_cf is not None:
            self.counterfactual(agg, sources, groups, h, weights, read_kind, pending_cf)
        return h

    def counterfactual(self, agg, sources, groups, h, weights, read_kind, members):
        member_to_source = {}
        for source_index, group in enumerate(groups):
            for member in group:
                member_to_source[member] = source_index
        base_r = rms(h).clamp_min(1e-12)
        for member in members:
            source_index = member_to_source[member]
            group = groups[source_index]
            for c in self.c_values:
                changed = list(sources)
                if self.variant == "full":
                    changed[source_index] = self.raw_values[member] * c
                elif self.variant == "block4":
                    changed[source_index] = sources[source_index] + (c - 1.0) * self.raw_values[member]
                else:
                    changed[source_index] = sum(
                        normalized(self.raw_values[i] * c) if i == member else normalized(self.raw_values[i])
                        for i in group
                    )
                hc, wc = aggregate(agg, changed)
                rel = rms(hc - h) / base_r
                cosine = 1.0 - torch.nn.functional.cosine_similarity(hc.float(), h.float(), dim=-1)
                alpha_tv = 0.5 * (wc.float() - weights.float()).abs().sum(0)
                key = (read_kind, member % 4, c)
                self.cf[(key, "relative_rms")].append(masked_numpy(rel, self.mask))
                self.cf[(key, "cosine_distance")].append(masked_numpy(cosine, self.mask))
                self.cf[(key, "alpha_total_variation")].append(masked_numpy(alpha_tv, self.mask))

    def result(self):
        output_rows = []
        for (index, kind), chunks in sorted(self.output_stats.items()):
            row = {"output_index": index, "transformer_layer": index // 2,
                   "sublayer": kind, **quantiles(np.concatenate(chunks))}
            output_rows.append(row)
        r = np.concatenate(self.pair_r) if self.pair_r else np.array([])
        a = np.concatenate(self.pair_alpha) if self.pair_alpha else np.array([])
        e = np.concatenate(self.pair_effective) if self.pair_effective else np.array([])
        if len(r) > 500_000:
            rng = np.random.default_rng(260926)
            take = rng.choice(len(r), 500_000, replace=False)
            r, a, e = r[take], a[take], e[take]
        cf_rows = []
        for (key, metric), chunks in sorted(self.cf.items()):
            read_kind, member_in_block, c = key
            cf_rows.append({"read_kind": read_kind, "member_in_block": member_in_block,
                            "c": c, "metric": metric, **quantiles(np.concatenate(chunks))})
        aggregate_cf = defaultdict(list)
        for (key, metric), chunks in self.cf.items():
            read_kind, _member_in_block, c = key
            aggregate_cf[(read_kind, c, metric)].extend(chunks)
        for (read_kind, c, metric), chunks in sorted(aggregate_cf.items()):
            cf_rows.append({"read_kind": read_kind, "member_in_block": "all",
                            "c": c, "metric": metric, **quantiles(np.concatenate(chunks))})
        return {
            "output_rms": output_rows,
            "coefficient_relation": correlation_summary(r, a, e),
            "coefficient_samples": {"r": r, "alpha": a, "effective": e},
            "within_block_directional_coefficient_ratio": quantiles(
                np.concatenate(self.within_block_ratio) if self.within_block_ratio else []),
            "within_block_log_coefficient_std": quantiles(
                np.concatenate(self.within_block_log_std) if self.within_block_log_std else []),
            "within_block_raw_r_ratio": quantiles(
                np.concatenate(self.within_block_raw_ratio) if self.within_block_raw_ratio else []),
            "within_block_alpha_ratio": quantiles(
                np.concatenate(self.within_block_alpha_ratio) if self.within_block_alpha_ratio else []),
            "effective_to_raw_ratio_of_ratios": quantiles(
                np.concatenate(self.within_block_compensation_ratio) if self.within_block_compensation_ratio else []),
            "within_block_slope_log_alpha_on_log_r": quantiles(
                np.concatenate(self.within_block_compensation_slope) if self.within_block_compensation_slope else []),
            "counterfactual": cf_rows,
            "checks": {"read_count": self.read_count,
                       "max_aggregate_formula_error": self.max_aggregate_error},
        }


def build_sources(variant, emb, completed, pending, raw_values):
    if variant == "full":
        return [emb] + raw_values, [[]] + [[i] for i in range(len(raw_values))], [False] * (1 + len(raw_values))
    sources = [emb] + [x[0] for x in completed]
    groups = [[]] + [x[1] for x in completed]
    normed = [False] + [variant == "norm_rms"] * len(completed)
    if pending:
        sources.append(sum(raw_values[i] for i in pending))
        groups.append(list(pending))
        normed.append(False)
    return sources, groups, normed


@torch.no_grad()
def run_batch(native, ids, labels, size, variant, c_values):
    mask = labels != -100
    emb = native.token_embedding_table(ids)
    cos, sin = native.rope(ids.shape[1], ids.device, emb.dtype)
    rec = Recorder(size, variant, mask, c_values)
    completed, pending = [], []
    pending_cf = None
    total_writes = 2 * len(native.blocks)

    def sources():
        return build_sources(variant, emb, completed, pending, rec.raw_values)

    def append(value, kind):
        nonlocal pending_cf
        idx = rec.append(value, kind)
        if variant == "full":
            if (idx + 1) % 4 == 0:
                pending_cf = list(range(idx - 3, idx + 1))
            return
        pending.append(idx)
        if len(pending) == 4:
            members = list(pending)
            if variant == "norm_rms":
                summary = sum(normalized(rec.raw_values[i]) for i in members)
            else:
                summary = sum(rec.raw_values[i] for i in members)
            completed.append((summary, members))
            pending.clear()
            pending_cf = members

    for layer, block in enumerate(native.blocks):
        src, groups, normed = sources()
        read_kind = "next_sublayer"
        h = rec.read(block.attn_res_attn, src, groups, normed, read_kind, pending_cf)
        pending_cf = None
        append(block.sa(block.ln1(h), cos, sin), "attention")

        src, groups, normed = sources()
        h = rec.read(block.attn_res_ffn, src, groups, normed, read_kind, pending_cf)
        pending_cf = None
        append(block.ffwd(block.ln2(h)), "ffn")

    src, groups, normed = sources()
    final = rec.read(native.attn_res_final, src, groups, normed, "final", pending_cf)
    native.summary_mask = mask
    reference = native._run_backbone(emb, cos, sin)
    diff = (final - reference).float()
    reference_scale = rms(reference).mean().clamp_min(1e-12)
    check = {"max_abs": float(diff.abs().max()),
             "relative_rms": float(rms(diff).mean() / reference_scale)}
    result = rec.result()
    result["checks"]["manual_backbone_vs_native"] = check
    result["checks"]["writes"] = len(rec.raw_values)
    assert len(rec.raw_values) == total_writes
    return result


def merge_batch_results(results):
    if len(results) != 1:
        raise ValueError('Quantiles cannot be merged by averaging. Use one batch, or fewer sequences.')
    return results[0]

def load_one(root, size, variant, device, native_source):
    run = root / SPECS[size][variant]
    cfg = json.loads((run / "config.json").read_text())
    if cfg["variant"] != variant:
        raise ValueError(f"Run variant mismatch: expected {variant}, got {cfg['variant']}")
    model = make_model(
        native_source, cfg["width"], cfg["layers"], cfg["heads"], cfg["seq_len"],
        "attnres_full" if variant == "full" else "attnres_block", 4, checkpointing=False,
        summary_mode="norm_rms" if variant == "norm_rms" else None, seed=cfg["seed"],
        publisher_config=cfg.get("publisher_config", False),
    )
    checkpoint = torch.load(run / "last.pt", map_location="cpu", weights_only=False, mmap=True)
    model.load_state_dict(checkpoint["model"], strict=True)
    step, tokens = int(checkpoint["step"]), int(checkpoint["tokens"])
    complete = json.loads((run / "COMPLETE.json").read_text())
    if step != complete["steps"] or tokens != complete["tokens"]:
        raise ValueError("Checkpoint is not the completed run endpoint")
    del checkpoint
    model.eval().to(device)
    return model, cfg, run, step, tokens


def jsonable(value):
    if isinstance(value, np.ndarray):
        return None
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items() if k != "coefficient_samples"}
    if isinstance(value, list):
        return [jsonable(v) for v in value]
    return value


def write_outputs(out, all_results, manifest):
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (out / "summary.json").write_text(json.dumps(jsonable(all_results), indent=2, sort_keys=True) + "\n")
    with (out / "residual_rms.csv").open("w", newline="") as f:
        fields = ["size", "variant", "output_index", "transformer_layer", "sublayer", "count",
                  "mean", "std", "p10", "p25", "median", "p75", "p90", "p99"]
        w = csv.DictWriter(f, fields); w.writeheader()
        for size, variants in all_results.items():
            for variant, result in variants.items():
                for row in result["output_rms"]:
                    w.writerow({"size": size, "variant": variant, **row})
    with (out / "counterfactual.csv").open("w", newline="") as f:
        fields = ["size", "variant", "read_kind", "member_in_block", "c", "metric", "count",
                  "mean", "std", "p10", "p25", "median", "p75", "p90", "p99"]
        w = csv.DictWriter(f, fields); w.writeheader()
        for size, variants in all_results.items():
            for variant, result in variants.items():
                for row in result["counterfactual"]:
                    w.writerow({"size": size, "variant": variant, **row})
    arrays = {}
    for size, variants in all_results.items():
        for variant, result in variants.items():
            for name, value in result["coefficient_samples"].items():
                arrays[f"{size}_{variant}_{name}"] = value
    np.savez_compressed(out / "coefficient_samples.npz", **arrays)


def plot_outputs(out, all_results):
    import matplotlib.pyplot as plt

    colors = {"full": "#3366cc", "block4": "#dc3912", "norm_rms": "#109618"}
    fig, axes = plt.subplots(1, len(all_results), figsize=(13, 4.5), squeeze=False)
    for ax, (size, variants) in zip(axes[0], all_results.items()):
        for variant, result in variants.items():
            rows = result["output_rms"]
            ax.plot([r["output_index"] for r in rows], [r["median"] for r in rows],
                    marker="o", ms=2.5, lw=1.2, label=variant, color=colors[variant])
        ax.set(title=size, xlabel="residual output index", ylabel="median RMS")
        ax.grid(alpha=.25); ax.legend()
    fig.tight_layout(); fig.savefig(out / "residual_rms_by_output.png", dpi=180); plt.close(fig)

    fig, axes = plt.subplots(1, len(all_results), figsize=(12, 4.2), squeeze=False)
    for ax, (size, variants) in zip(axes[0], all_results.items()):
        names, med, p90 = [], [], []
        for variant, result in variants.items():
            q = result["within_block_directional_coefficient_ratio"]
            names.append(variant); med.append(q.get("median", np.nan)); p90.append(q.get("p90", np.nan))
        x = np.arange(len(names)); ax.scatter(x, med, label="median", s=55)
        ax.scatter(x, p90, label="p90", marker="x", s=65)
        ax.set_xticks(x, names, rotation=15); ax.set_yscale("log")
        ax.set(title=size, ylabel="within-block max/min directional coefficient")
        ax.grid(alpha=.25); ax.legend()
    fig.tight_layout(); fig.savefig(out / "within_block_coefficient_ratio.png", dpi=180); plt.close(fig)

    fig, axes = plt.subplots(1, len(all_results), figsize=(12, 4.2), squeeze=False)
    for ax, (size, variants) in zip(axes[0], all_results.items()):
        for variant, result in variants.items():
            rows = [r for r in result["counterfactual"]
                    if r["metric"] == "relative_rms" and r["read_kind"] == "next_sublayer"
                    and r["member_in_block"] == "all"]
            by_c = defaultdict(list)
            for r in rows: by_c[r["c"]].append((r["median"], r["count"]))
            if any(len(v) != 1 for v in by_c.values()):
                raise ValueError("Cannot merge quantiles by averaging")
            by_c[1.] = [(0., 1)]  # Exact identity, not an additional measured sample.
            cs = sorted(by_c)
            ys = [by_c[c][0][0] for c in cs]
            ax.plot(cs, ys, marker="o", label=variant, color=colors[variant])
        ax.set(title=size, xlabel="positive scale c", ylabel="median relative RMS change")
        ax.set_xscale("log", base=2)
        ax.set_xticks([.25, .5, 1, 2, 4], ["0.25", "0.5", "1", "2", "4"])
        ax.grid(alpha=.25); ax.legend()
    fig.tight_layout(); fig.savefig(out / "counterfactual_sensitivity.png", dpi=180); plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument("--runs-json", type=Path, required=True)
    p.add_argument("--prepared", type=Path, help="Relocated data directory")
    p.add_argument("--allow-data-mismatch", action="store_true",
                   help="Explicitly permit a different data fingerprint; records the mismatch")
    p.add_argument("--native-source", type=Path, default=ROOT / "vendor/brujula")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--sizes", default="57m,157m")
    p.add_argument("--variants", default="full,block4,norm_rms")
    p.add_argument("--sequences", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=260926)
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-plots", action="store_true")
    args = p.parse_args()
    global SPECS
    SPECS = json.loads(args.runs_json.read_text())
    if args.batch_size < args.sequences: p.error("Require batch-size >= sequences for exact quantiles")
    root, device = args.root.resolve(), torch.device(args.device)
    sizes, variants = args.sizes.split(","), args.variants.split(",")
    if any(v not in ("full", "block4", "norm_rms") for v in variants):
        p.error("Diagnostic replay supports only full, block4 and norm_rms; Full+RMS is not implemented")
    if args.sequences < 1 or args.batch_size < 1:
        p.error("Sample and batch sizes must be positive")
    if args.out.exists() and any(args.out.iterdir()):
        p.error("Refusing to overwrite existing diagnostics; choose an empty output directory")
    all_results, manifest = {}, {"seed": args.seed, "sequences": args.sequences,
                                "batch_size": args.batch_size, "c_values": [.25,.5,2.,4.],
                                "torch": torch.__version__, "cuda": torch.version.cuda,
                                "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
                                "runs": {}}
    for size in sizes:
        all_results[size] = {}
        first_cfg = json.loads((root / SPECS[size][variants[0]] / "config.json").read_text())
        data = DocumentChunks(args.prepared or Path(first_cfg["prepared"]), "test", first_cfg["seq_len"], first_cfg["eval_tokens"])
        rng = np.random.default_rng(args.seed + (57 if size == "57m" else 157))
        selected = rng.choice(len(data), size=min(args.sequences, len(data)), replace=False).tolist()
        for variant in variants:
            print(json.dumps({"stage": "load", "size": size, "variant": variant}), flush=True)
            model, cfg, run, step, tokens = load_one(root, size, variant, device, args.native_source)
            if cfg["seq_len"] != first_cfg["seq_len"] or cfg["eval_tokens"] != first_cfg["eval_tokens"]:
                raise ValueError("Diagnostic variants must use matched sequence length and evaluation budget")
            data_match = data.fingerprint == cfg["data_fingerprint"]["test"]
            if not data_match and not args.allow_data_mismatch:
                raise ValueError("Test data fingerprint differs from training record; inspect before allowing a mismatch")
            manifest["runs"][f"{size}/{variant}"] = {
                "path": str(run), "checkpoint_sha256": sha256(run / "last.pt"),
                "step": step, "tokens": tokens, "parameters": json.loads((run / "COMPLETE.json").read_text())["parameters"],
                "test_indices": selected, "seq_len": cfg["seq_len"], "data_fingerprint": data.fingerprint,
                "training_record_test_fingerprint": cfg["data_fingerprint"]["test"],
                "data_fingerprint_matches": data_match,
            }
            batch_results = []
            amp = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else contextlib.nullcontext()
            for start in range(0, len(selected), args.batch_size):
                ids, labels = data.batch(selected[start:start + args.batch_size])
                ids, labels = ids.to(device), labels.to(device)
                with amp:
                    batch_results.append(run_batch(model.native, ids, labels, size, variant, [.25,.5,2.,4.]))
            result = merge_batch_results(batch_results)
            all_results[size][variant] = result
            print(json.dumps({"stage": "done", "size": size, "variant": variant,
                              "checks": result["checks"],
                              "relation": result["coefficient_relation"],
                              "counterfactual_rows": len(result["counterfactual"])}), flush=True)
            del model, batch_results
            gc.collect()
            if device.type == "cuda": torch.cuda.empty_cache()
            write_outputs(args.out, all_results, manifest)
    write_outputs(args.out, all_results, manifest)
    if not args.no_plots:
        plot_outputs(args.out, all_results)
    print(json.dumps({"stage": "complete", "out": str(args.out)}), flush=True)


if __name__ == "__main__":
    main()
