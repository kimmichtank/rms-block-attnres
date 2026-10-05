"""Brujula v2 — DeepSeek-style decoder for HuggingFace Transformers, with v3 Block
Attention-Residuals (arXiv 2603.15031) and YaRN context-extension baked into the model.

This is a strict SUPERSET of the v1 module:
  - `config.residual='plain'` reproduces the v1 graph exactly (every v1 checkpoint loads).
  - `config.residual='attnres_block'` (default here) runs v3 Block-AttnRes: each sublayer
    aggregates earlier representations with softmax attention over windowed block-sums,
    instead of the plain residual sum.
  - `config.rope_scaling_method='yarn_aggr'` warps the RoPE frequencies (NTK-by-parts ramp
    + YaRN attention temperature) so the model runs at `rope_scale_len` (32768) even though
    it was pre-trained at `rope_trained_len` (1024). The scaling is recomputed on the fly per
    forward from the config — nothing to re-apply at load time, the 32K behaviour ships in
    the config+weights.

Module + parameter names match the faro training graph exactly, so the trained state_dict
loads strict=True (modulo tied lm_head). Logits are parity-checked <1e-5 against the
canonical faro `GPTLanguageModel` at build time (see build_brujula_v2.py); faro is canonical.

Generation uses NO KV cache — the AttnRes aggregators mix across layers, so a per-step cache
would be intricate; recomputing the (cropped) context each step is simple and correct. Fine
for a 150M model; expect long-context generation to be slow. Run at bf16/fp16 on GPU.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GenerationMixin, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

try:  # works both as a Hub module (relative) and as a local import (absolute)
    from .configuration_brujula_v2 import BrujulaConfig
except ImportError:  # pragma: no cover
    from configuration_brujula_v2 import BrujulaConfig


# ------------------------------------------------------------------ RoPE (+ YaRN)
class RotaryEmbedding(nn.Module):
    """Computes cos/sin on the fly (no persistent buffers — those don't survive
    from_pretrained's meta-device init). The YaRN/PI/NTK warping is folded in here so the
    extended-context positions ship in the config; reproduces faro.nn.rope_scaling.method_tables
    exactly. The per-dim frequency scale and the YaRN attention temperature (mscale) depend ONLY
    on (method, trained_len, scale_len) — not on the table length — so building rows 0..seq_len-1
    on demand gives the same angles as faro's full scale_len table indexed [:seq_len]."""

    def __init__(self, head_dim, scale_len, trained_len, method=None, base=10000):
        super().__init__()
        assert head_dim % 2 == 0, "head_dim must be even for RoPE"
        self.head_dim = head_dim
        self.base = base
        self.scale_len = scale_len
        self.trained_len = trained_len
        self.method = method if method not in ("", "none") else None

    def _scale_and_mscale(self, inv_freq):
        """Return (per-dim position scale, mscale). inv_freq: (head_dim/2,) fp32."""
        m = self.method
        s = self.trained_len / self.scale_len  # interpolation factor (<1 for extension)
        if m is None or m == "naive":
            return torch.ones_like(inv_freq), 1.0
        if m == "pi":
            return torch.full_like(inv_freq, s), 1.0
        if m in ("yarn", "yarn_ms", "yarn_aggr"):
            # NTK-by-parts ramp: high-freq dims (>=beta cycles in trained_len) extrapolate
            # (scale 1), low-freq fully interpolate (scale s), linear between.
            beta = 16.0 if m == "yarn_aggr" else 32.0
            # match faro.nn.rope_scaling EXACTLY (this float32 form, not trained_len*inv/2pi):
            # different operation order = different rounding = ~1e-5 logit drift after 18 layers.
            cycles = self.trained_len / (2 * math.pi / inv_freq)
            ramp = ((cycles - 1.0) / (beta - 1.0)).clamp(0.0, 1.0)
            scale = s * (1.0 - ramp) + 1.0 * ramp
            mscale = 0.1 * math.log(self.scale_len / self.trained_len) + 1.0 if m in ("yarn_ms", "yarn_aggr") else 1.0
            return scale, mscale
        raise ValueError(f"unknown rope_scaling_method {m!r}")

    def forward(self, seq_len, device, dtype):
        half = torch.arange(0, self.head_dim, 2, device=device).float()
        if self.method == "ntk":
            # NTK-static: raise the base so high freqs are ~unchanged, low freqs interpolate.
            ntk_base = self.base * (self.scale_len / self.trained_len) ** (self.head_dim / (self.head_dim - 2))
            inv_freq = 1.0 / (ntk_base ** (half / self.head_dim))
            scale, mscale = torch.ones_like(inv_freq), 1.0
        else:
            inv_freq = 1.0 / (self.base ** (half / self.head_dim))
            scale, mscale = self._scale_and_mscale(inv_freq)
        pos = torch.arange(seq_len, device=device).float()
        freqs = torch.outer(pos, inv_freq) * scale.unsqueeze(0)  # (T, half)
        cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1) * mscale
        sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1) * mscale
        return cos.to(dtype), sin.to(dtype)


def _rotate_half(x):
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q, k, cos, sin):
    cos = cos.unsqueeze(0).unsqueeze(0).to(q.dtype)
    sin = sin.unsqueeze(0).unsqueeze(0).to(q.dtype)
    q_rot = q * cos + _rotate_half(q) * sin
    k_rot = k * cos + _rotate_half(k) * sin
    return q_rot, k_rot


# ------------------------------------------------------------------ norms / FFN
class CastingRMSNorm(nn.RMSNorm):
    """Cast weight to input dtype on the fly so the fused kernel dispatches under autocast."""

    def forward(self, x):
        w = self.weight if self.weight.dtype == x.dtype else self.weight.to(x.dtype)
        return F.rms_norm(x, self.normalized_shape, w, self.eps)


class SquaredReLU(nn.Module):
    def __init__(self, n_embd, dropout=0.0):
        super().__init__()
        hidden_dim = 4 * n_embd
        hidden_dim = (hidden_dim + 63) // 64 * 64
        self.w1 = nn.Linear(n_embd, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.relu(self.w1(x)) ** 2))


# ------------------------------------------------- Attention-Residual aggregator (v2/v3)
class AttnResAggregator(nn.Module):
    """Softmax attention over a list of prior sublayer outputs (arXiv 2603.15031):
        h = sum_i softmax(w . RMSNorm(v_i))_i * v_i
    `w` is a learnable d-dim pseudo-query (zero-init -> uniform start). RMSNorm on the keys
    keeps high-magnitude layers from dominating. Identical math/params to faro's aggregator."""

    def __init__(self, n_embd):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(n_embd))
        self.key_norm = CastingRMSNorm(n_embd)

    def forward(self, prior_values):
        V = torch.stack(prior_values, dim=0)             # [L, B, T, d]
        K = self.key_norm(V)                             # [L, B, T, d]
        logits = torch.einsum("d,lbtd->lbt", self.w, K)  # [L, B, T]
        weights = F.softmax(logits, dim=0)               # over the L source dim
        return (weights.unsqueeze(-1) * V).sum(dim=0)    # [B, T, d]


# ------------------------------------------------------------------ attention / block
class MultiHeadLatentAttention(nn.Module):
    def __init__(self, n_embd, num_heads, head_size, dropout, kv_compression_dim, q_compression_dim, rope):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.W_DKV = nn.Linear(n_embd, kv_compression_dim, bias=False)
        self.W_DQ = nn.Linear(n_embd, q_compression_dim, bias=False)
        self.W_UK = nn.Linear(kv_compression_dim, num_heads * head_size, bias=False)
        self.W_UV = nn.Linear(kv_compression_dim, num_heads * head_size, bias=False)
        self.W_UQ = nn.Linear(q_compression_dim, num_heads * head_size, bias=False)
        self.ln_kv = CastingRMSNorm(kv_compression_dim)
        self.ln_q = CastingRMSNorm(q_compression_dim)
        self.proj = nn.Linear(num_heads * head_size, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.rope = rope

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        c_kv = self.ln_kv(self.W_DKV(x))
        c_q = self.ln_q(self.W_DQ(x))
        k = self.W_UK(c_kv).view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        v = self.W_UV(c_kv).view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        q = self.W_UQ(c_q).view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        q, k = apply_rope(q, k, cos, sin)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, self.num_heads * self.head_size)
        return self.dropout(self.proj(out))


class Block(nn.Module):
    def __init__(self, n_embd, n_head, dropout, kv_compression_dim, q_compression_dim, rope, residual):
        super().__init__()
        head_size = n_embd // n_head
        self.residual = residual
        self.sa = MultiHeadLatentAttention(
            n_embd, n_head, head_size, dropout, kv_compression_dim, q_compression_dim, rope
        )
        self.ffwd = SquaredReLU(n_embd, dropout)
        self.ln1 = CastingRMSNorm(n_embd)
        self.ln2 = CastingRMSNorm(n_embd)
        if residual in ("attnres_full", "attnres_block"):
            self.attn_res_attn = AttnResAggregator(n_embd)
            self.attn_res_ffn = AttnResAggregator(n_embd)

    def forward(self, x, cos, sin):
        # plain residual path only; the attnres paths are driven by the model's forward().
        x = x + self.sa(self.ln1(x), cos, sin)
        x = x + self.ffwd(self.ln2(x))
        return x


# ------------------------------------------------------------------ HF model
class BrujulaPreTrainedModel(PreTrainedModel):
    config_class = BrujulaConfig
    base_model_prefix = "brujula"
    supports_gradient_checkpointing = False
    _no_split_modules = ["Block"]

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, AttnResAggregator):
            nn.init.zeros_(module.w)


class BrujulaForCausalLM(BrujulaPreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "token_embedding_table.weight"}

    def __init__(self, config):
        super().__init__(config)
        self.residual = config.residual
        self.attnres_block_size = max(1, config.attnres_block_size)
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        head_size = config.n_embd // config.n_head
        self.rope = RotaryEmbedding(
            head_size, config.rope_scale_len, config.rope_trained_len,
            method=config.rope_scaling_method, base=config.rope_base,
        )
        self.blocks = nn.ModuleList(
            [
                Block(
                    config.n_embd, config.n_head, config.dropout,
                    config.kv_compression_dim, config.q_compression_dim, self.rope, config.residual,
                )
                for _ in range(config.n_layer)
            ]
        )
        if config.residual in ("attnres_full", "attnres_block"):
            self.attn_res_final = AttnResAggregator(config.n_embd)
        self.ln_f = CastingRMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.post_init()  # init + tie (config.tie_word_embeddings=True)

    def get_input_embeddings(self):
        return self.token_embedding_table

    def set_input_embeddings(self, value):
        self.token_embedding_table = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value):
        self.lm_head = value

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        logits_to_keep=0,
        **kwargs,
    ):
        idx = input_ids
        if idx.size(1) > self.config.block_size:  # extended context window
            idx = idx[:, -self.config.block_size:]
            if labels is not None:
                labels = labels[:, -self.config.block_size:]
        B, T = idx.shape
        x = self.token_embedding_table(idx)
        cos, sin = self.rope(T, x.device, x.dtype)
        x = self._run_backbone(x, cos, sin)
        x = self.ln_f(x)
        # At long context the full [B, T, vocab] logits are the memory hog (32K x 50257 x fp32 ~6.6GB).
        # During generation we only need the last token, so slice the hidden state BEFORE lm_head.
        # logits_to_keep=0 (default / parity / training path) keeps every position.
        if labels is None and logits_to_keep:
            x = x[:, -logits_to_keep:, :]
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
        return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=None)

    def _run_backbone(self, x, cos, sin):
        if self.residual == "attnres_full":
            v_list = [x]
            for block in self.blocks:
                h_attn = block.attn_res_attn(v_list)
                attn_out = block.sa(block.ln1(h_attn), cos, sin)
                v_list.append(attn_out)
                h_ffn = block.attn_res_ffn(v_list)
                ffn_out = block.ffwd(block.ln2(h_ffn))
                v_list.append(ffn_out)
            return self.attn_res_final(v_list)

        if self.residual == "attnres_block":
            # v3 Block-AttnRes: window the source list into blocks of S sublayers so the
            # number of AttnRes sources stays O(N). Verbatim from faro.nn.model.
            emb = x
            completed = []      # finalized block sums
            partial = None      # running sum of the current block's sublayer outputs
            sublayer_in_block = 0
            S = self.attnres_block_size

            def sources():
                return [emb] + completed + ([partial] if partial is not None else [])

            def append_sublayer(v):
                nonlocal partial, sublayer_in_block
                partial = v if partial is None else (partial + v)
                sublayer_in_block += 1
                if sublayer_in_block == S:
                    completed.append(partial)
                    partial = None
                    sublayer_in_block = 0

            for block in self.blocks:
                h_attn = block.attn_res_attn(sources())
                attn_out = block.sa(block.ln1(h_attn), cos, sin)
                append_sublayer(attn_out)
                h_ffn = block.attn_res_ffn(sources())  # sees attn_out via the updated partial
                ffn_out = block.ffwd(block.ln2(h_ffn))
                append_sublayer(ffn_out)

            if partial is not None:  # flush an unfinished trailing block
                completed.append(partial)
            return self.attn_res_final([emb] + completed)

        for block in self.blocks:
            x = block(x, cos, sin)
        return x

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **kwargs):
        # No KV cache: feed the (cropped) running sequence each step, but only ask lm_head for the
        # last token's logits so long-context generation doesn't materialize the full-vocab tensor.
        inputs = {"input_ids": input_ids, "logits_to_keep": 1}
        if attention_mask is not None:
            inputs["attention_mask"] = attention_mask
        return inputs


# Register with the Auto API on import so AutoModelForCausalLM resolves "brujula" cleanly
# (config registration lives in configuration_brujula_v2). Guarded — never crash a Hub import.
try:
    from transformers import AutoConfig, AutoModelForCausalLM

    AutoConfig.register("brujula", BrujulaConfig, exist_ok=True)
    AutoModelForCausalLM.register(BrujulaConfig, BrujulaForCausalLM, exist_ok=True)
except Exception:  # pragma: no cover
    pass


__all__ = ["BrujulaConfig", "BrujulaPreTrainedModel", "BrujulaForCausalLM"]
