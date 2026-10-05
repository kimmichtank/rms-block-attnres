"""Brujula v2 configuration — DeepSeek-style decoder (MLA + RoPE + SquaredReLU, tied
embeddings) with two additions over the v1 config:

  1. `residual` — depth-mixing scheme:
        'plain'         : v1 graph (x = x + sa(ln1(x)); x = x + ffwd(ln2(x)))
        'attnres_full'  : v2 Attention Residuals over ALL prior sublayer outputs
        'attnres_block' : v3 Block-AttnRes — softmax attention over windowed block-sums
                          (`attnres_block_size` sublayers per block).  arXiv 2603.15031.
  2. RoPE context-extension (YaRN family), baked into the model so the 32K behaviour
     ships in the weights+config, with NO manual table surgery at load time:
        rope_scaling_method : None | 'yarn_aggr' | 'yarn' | 'yarn_ms' | 'pi' | 'ntk' | 'naive'
        rope_scale_len      : target context the frequencies are warped for (e.g. 32768)
        rope_trained_len    : the length the base model was pre-trained at (e.g. 1024)

This is a strict SUPERSET of v1: with residual='plain' and rope_scaling_method=None it
reproduces the v1 model byte-for-byte, so the same module can serve the whole family.
"""

from transformers import PretrainedConfig


class BrujulaConfig(PretrainedConfig):
    model_type = "brujula"

    def __init__(
        self,
        vocab_size=50257,
        n_embd=816,
        n_head=6,
        n_layer=18,
        block_size=32768,
        kv_compression_dim=64,
        q_compression_dim=192,
        dropout=0.0,
        rope_base=10000,
        # --- depth-mixing (v2/v3 AttnRes) ---
        residual="attnres_block",
        attnres_block_size=4,
        # --- RoPE context extension (YaRN family) ---
        rope_scaling_method="yarn_aggr",
        rope_scale_len=32768,
        rope_trained_len=1024,
        tie_word_embeddings=True,
        bos_token_id=50256,
        eos_token_id=50256,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer
        self.block_size = block_size
        self.kv_compression_dim = kv_compression_dim
        self.q_compression_dim = q_compression_dim
        self.dropout = dropout
        self.rope_base = rope_base
        self.residual = residual
        self.attnres_block_size = attnres_block_size
        self.rope_scaling_method = rope_scaling_method
        self.rope_scale_len = rope_scale_len
        self.rope_trained_len = rope_trained_len
        # HF-standard aliases so generate()/pipelines/tooling find the usual fields.
        self.hidden_size = n_embd
        self.num_attention_heads = n_head
        self.num_hidden_layers = n_layer
        self.max_position_embeddings = block_size
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


# Register the model_type so stock transformers recognizes "brujula" once this module is
# imported (it is, via trust_remote_code) — without it, a bare AutoConfig/AutoTokenizer load
# falls back to the base config class and prints a spurious "model of type `brujula` to
# instantiate a model of type ``" warning. Guarded so a Hub import never crashes here.
try:
    from transformers import AutoConfig

    AutoConfig.register("brujula", BrujulaConfig, exist_ok=True)
except Exception:  # pragma: no cover
    pass
