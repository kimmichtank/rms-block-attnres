# Design: exactly which magnitudes are removed?

## Claim and scope

The question is whether residual aggregation benefits from retaining each source's radial magnitude after routing has already normalized its Key. We study training outcomes and local scale sensitivity separately. Nothing here establishes that magnitude lacks semantic information.

For a nonzero single-token source define

$$
r_i=\sqrt{\frac1d\sum_{j=1}^d y_{ij}^2},\quad u_i=\frac{y_i}{r_i},\quad
N_\epsilon(y)=\frac{y}{\sqrt{d^{-1}\sum_j y_j^2+\epsilon}}.
$$

$u_i$ has RMS one, not Euclidean norm one; its Euclidean norm is $\sqrt d$. The exact decomposition $y_i=r_i u_i$ uses no affine gain or epsilon. When referring to actual library RMSNorm, write $K_l(y)=\gamma_l\odot N_{\epsilon_l}(y)$ rather than silently treating it as that exact decomposition.

## Four concrete variants

All readers have a learned zero-initialized pseudo-query $w_l$, so initial source-softmax weights are uniform. Each reader's normalized Key has a learned coordinate-wise gain $\gamma_l$. This query is a parameter, not a function of the current hidden state.

**Full** stores the embedding and each attention/FFN output separately. The read is

$$
\alpha_{i\to l}=\frac{\exp(w_l^\top K_l(y_i))}{\sum_j\exp(w_l^\top K_l(y_j))},\qquad
h_l=\sum_i\alpha_{i\to l}y_i.
$$

**Full+RMS, as actually trained at 57M**, changes only the Value expression to the reader's existing normalized Key:

$$
h_l=\sum_i\alpha_{i\to l}K_l(y_i).
$$

It includes the embedding source, every individual sublayer output and the final aggregation. It introduces no new parameters, but the existing gain now participates in the Value path as well as routing. Sources are normalized on read, not stored as a single reader-independent normalized vector. Positive-scale invariance survives a fixed affine gain in the negligible-epsilon limit; exact unit RMS generally does not.

**Block4** groups four successive residual outputs, with attention and FFN counted separately. Completed group $B$ stores $b_B=\sum_{i\in B}y_i$. Each read sees the raw embedding, all completed sums and the current partial sum if one exists, and applies the same normalized-Key/raw-Value attention over those sources.

**RMS Block4** buffers raw outputs. Only when four outputs complete a group does it compute

$$
b_B^{\rm RMS}=\sum_{i\in B}N_{10^{-6}}(y_i).
$$

Each normalization reduces the last, hidden dimension independently for every token and sequence. Computation is FP32, with results cast back to the source dtype before summation. There is no learned gain here. The completed summary is not subsequently forced to have unit RMS: aligned directions can add, and opposed directions can cancel.

The **pending** block still reads as $\sum_{i\in P}y_i$, with raw magnitudes. The embedding is separate and raw. A current source therefore changes representation at the boundary: it previously entered a raw partial sum, then enters a completed sum of individually normalized sources. The small-model implementation requires full groups of four; the audited 12/18-layer runs satisfy this. The Megatron state supports pending outputs too; its 36-layer runs have complete groups.

Relevant code: [portable model wrapper](../small/pretrain_core.py), `full_rms_aggregate`, `HiddenNormalizedSum.transform`, `two_summary_backbone`; [native reader](../vendor/brujula/modeling_brujula_v2.py), `AttnResAggregator`; [Megatron patch](../scaling/megatron-attnres.patch), `AttnResState.append`, `AttnResState.values`, `rms_write`.

## Why pre-sum normalization is different

Take two non-collinear RMS-one vectors. A raw summary $5u_1+u_2$ is generally not parallel to $u_1+u_2$. Normalizing the **sum** cannot recover the latter direction: it rescales an already biased direction. Normalizing each source **before** the sum changes its relative coefficient. There is no averaging over different tokens, sequences or hidden coordinates as source vectors; only the RMS denominator uses the hidden coordinates of the same token.

Full's directional effective coefficient is $\alpha_i r_i$ in the ideal decomposition. For a raw completed Block it is $\alpha_B r_i$. For an RMS completed Block it is approximately $\alpha_B$, up to epsilon and finite precision. This is a coefficient decomposition, not an assertion that the norm of the complete sum equals the sum of coefficient magnitudes; source directions can reinforce or cancel.

## A local invariance argument, not a whole-network expressivity theorem

Freeze parameters and all other source vectors. For $c>0$ and negligible epsilon, $K_l(cy_i)=K_l(y_i)$, so the Full attention weights do not change. Its read nevertheless changes by

$$
h_l'-h_l=(c-1)\alpha_{i\to l}y_i.
$$

Full+RMS approximately removes this effect for each read source. RMS Block removes it when the scaled source is normalized into a completed summary, but not while it contributes to a raw pending sum. Rescaling one member of a **raw Block** can also change the summary's direction and its routing weight; Block and Full need not have the same counterfactual behavior.

These read operators have different structural invariances. RMS Block does not generally reproduce arbitrary Full reads or ordinary raw residual addition. Conversely, normalized-Key/raw-Value Full cannot in general synthesize inverse-$r_i$ compensation as $r_i$ varies independently of direction: its router receives no such radial signal. Neither statement orders the complete trainable networks by strict function-class inclusion. Upstream transformations can change their learned outputs; epsilon, degenerate sources and restricted input domains also require care.

Native Brújula `CastingRMSNorm` inherits PyTorch's default epsilon (`eps=None`, resolved from input dtype). The block write explicitly uses $10^{-6}$. The Megatron reader uses the configured norm epsilon and its own low-precision arithmetic. No exact invariance guarantee is made near zero norm or across arbitrary precision modes.

## Interpretation to test next

The observed early validation advantage of Full+RMS is consistent with normalization reducing an optimization burden associated with radial variation. Geometric preconditioning is a **hypothesis**, not a measured condition-number result. The eventual 57M tie suggests the raw-Value model can largely catch up in this setting. It does not show radial variation is always harmful or always useful.

Block compression adds a separate constraint on routing resolution. Its smoothing/compression effects are also hypotheses about inductive bias. The stronger 57M Full+RMS result relative to RMS Block argues against attributing everything to compression, but gain/embedding/pending differences prevent a perfectly clean factorial interpretation. A matched four-arm extension should pin those policies before any training.

Conceptual precedents include [nGPT](https://arxiv.org/abs/2410.01131), which normalizes representations and weights while retaining learned scale controls, and [Hyperball](https://arxiv.org/abs/2606.16899), which controls weight/update norms at the optimizer level. Neither proves residual-output magnitude is semantically irrelevant. Our aggregation-level intervention is different from both.

## Why can the raw block summary be worse?

The next question is narrower than “are small outputs important?” For a completed block and a later Full reader, define the block's actual Full contribution

$$
C_{B\to l}=\sum_{i\in B}\alpha_{i\to l}r_i u_i.
$$

A raw Block reader restricts the coefficients of the four directions to the form $\beta_{B\to l}r_i$. RMS Block restricts them approximately to $\beta_{B\to l}$. If Full learned $\alpha_i\propto1/r_i$, equal coefficients would follow and RMS Block would resemble Full. If Full learned nearly constant $\alpha_i$, raw Block would resemble Full. Neither pattern should be assumed.

The existing pooled Full diagnostics show partial but insufficient compensation: the fitted slope of $\log\alpha$ on $\log r$ is −0.441 at 57M and −0.355 at 157M, rather than −1. Within a four-source block, the median max/min coefficient ratio changes from raw $r_i$ imbalance 7.05 to Full effective $\alpha_i r_i$ imbalance 5.21 at 57M, but from 5.31 to 8.14 at 157M. This establishes that Full does not consistently cancel magnitude imbalance. It does **not** establish that equal source coefficients are optimal.

Three effects must be separated:

1. **Systematic scale mismatch:** attention and FFN outputs may have different typical RMS because their parameterizations differ. Activation RMS is then a poor cross-module importance scale.
2. **Per-token radial information:** variation in $r_i$ around the source layer's typical scale may contain useful input-dependent information, noise, or both.
3. **Directional redundancy:** a small attention output may point in a direction already supplied by a larger FFN output. Equalizing it is useful only if it contributes information not already represented by the other directions.

Routing weight alone is not source importance. A large $\alpha_i$ can multiply a small $r_i$; source directions can reinforce or cancel; and removing a source can affect all later computation. The required evidence therefore combines conditional routing statistics, vector reconstruction and loss interventions.
