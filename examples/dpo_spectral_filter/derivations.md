# Derivations

Mathematical details supporting the claims in [README.md](README.md).

## 1. The DPO gradient

A language model $\pi_\theta(y \mid x)$ assigns probability to a completion
$y$ given prompt $x$ via the autoregressive factorization:

$$\log \pi_\theta(y \mid x) = \sum_{t} \log \pi_\theta(y_t \mid x, y_1, \ldots, y_{t-1})$$

Each term is the log-softmax at position $t$, evaluated at the token $y_t$.
Computing $\log \pi_\theta(y \mid x)$ is a single forward pass through the
transformer — one scalar out.

The DPO loss (Rafailov et al. 2023, Eq. 7) is:

$$\mathcal{L}_{\mathrm{DPO}} = -\log \sigma\!\Big(\beta \big[\log \pi_\theta(y_w \mid x) - \log \pi_\theta(y_l \mid x) - \log \pi_{\mathrm{ref}}(y_w \mid x) + \log \pi_{\mathrm{ref}}(y_l \mid x)\big]\Big)$$

Define the implicit reward
$\hat{r}_\theta(x, y) = \beta \log \frac{\pi_\theta(y \mid x)}{\pi_{\mathrm{ref}}(y \mid x)}$
and the adaptive weight
$w = \sigma\!\big(\hat{r}_\theta(x, y_l) - \hat{r}_\theta(x, y_w)\big)$.
The gradient is (Rafailov et al. 2023, Section 4):

$$\nabla_\theta \mathcal{L}_{\mathrm{DPO}} = -\beta \, w \, \big[\nabla_\theta \log \pi_\theta(y_w \mid x) - \nabla_\theta \log \pi_\theta(y_l \mid x)\big]$$

Two key structural properties compared to the CE gradient
$\nabla_\theta \mathcal{L}_{\mathrm{CE}} = -\nabla_\theta \log \pi_\theta(y_w \mid x)$:

1. **Contrastive.** The DPO gradient is the *difference* of two score
   functions. CE only pushes up $y_w$; DPO simultaneously pushes up $y_w$ and
   pushes down $y_l$.

2. **Self-regulating.** The sigmoid weight $w$ modulates the gradient based on
   how wrong the model's current implicit reward ranking is. When the model
   already prefers $y_w$ over $y_l$, $w \to 0$ and the gradient vanishes. CE
   has no such mechanism.

## 2. DPO-CE gradient overlap in parameter space

Let $g_w = \nabla_\theta \log \pi_\theta(y_w \mid x)$ and
$g_l = \nabla_\theta \log \pi_\theta(y_l \mid x)$ be the score functions,
and $\alpha$ the angle between them. The dot product of the two loss gradients
is:

$$\langle \nabla_\theta \mathcal{L}_{\mathrm{DPO}},\; \nabla_\theta \mathcal{L}_{\mathrm{CE}} \rangle = \beta w \, \lVert g_w \rVert \big(\lVert g_w \rVert - \lVert g_l \rVert \cos \alpha\big)$$

This makes three quantities visible: the adaptive weight $w$, the score
function norms $\lVert g_w \rVert$, $\lVert g_l \rVert$, and their angular
separation $\alpha$. The sign of the overlap is determined by the geometric
factor $\lVert g_w \rVert - \lVert g_l \rVert \cos \alpha$: it goes negative
when $\frac{\lVert g_l \rVert}{\lVert g_w \rVert} \cos \alpha > 1$, requiring
both directional similarity and $\lVert g_l \rVert > \lVert g_w \rVert$.

**In the diagnostic setting** ($\pi_{\mathrm{ref}} = \pi_\theta$, i.e.
evaluating DPO readiness at an SFT checkpoint), the implicit reward is zero,
$w = 1/2$, and the overlap simplifies to:

$$\frac{\beta}{2} \, \lVert g_w \rVert \big(\lVert g_w \rVert - \lVert g_l \rVert \cos \alpha\big)$$

The dynamics of $\lVert g_w \rVert$ and $\lVert g_l \rVert$ during SFT are not
predictable from first principles. Each is a parameter-space norm
$\lVert (\nabla_\theta f)^\top \nabla_f \log \pi \rVert$, the product of a
loss-space gradient and the parameter Jacobian. The loss-space gradient on
$y_w$ likely shrinks as SFT fits $y_w$-like data, but the Jacobian
$\nabla_\theta f$ can grow as the model develops sharper internal
representations — and the parameter-space norm depends on which factor
dominates. Similarly, $\cos \alpha$ is probably already well below 1 due to
high dimensionality, but its trajectory during training depends on
model-specific details.

The fundamental limitation of the parameter-space picture is that the
directional quantity we care about (is the DPO-CE alignment improving?) is
entangled with scale factors whose dynamics we cannot predict. Any directional
transition can be masked, amplified, or mimicked by norm drift.

## 3. What $\chi_{\mathrm{pos}}$ adds

The parameter-space decomposition (§2) gives DPO-specific insight — the
adaptive weight, the norm asymmetry, the three regimes. But it cannot answer
the question we actually care about: **when is the model ready for DPO?** The
reasons are specific:

**Scale invariance.** The overlap $\delta L$ is the product of $\beta w$,
$\lVert g_w \rVert$, the geometric factor, and implicitly the Jacobian norms.
All of these drift during training. A sign change in the directional alignment
can be masked, amplified, or mimicked by scale drift. To track direction across
checkpoints, we need to divide out all scales.

Applying the chain rule
$\nabla_\theta \mathcal{L} = (\nabla_\theta f)^\top \nabla_f \mathcal{L}$
to $\delta L$ and dividing each factor by its norm:

$$\chi_{\mathrm{pos}} = \frac{(\nabla_f \mathcal{L}_{\mathrm{DPO}})^\top \, \nabla_\theta f_{\mathrm{DPO}} \, (\nabla_\theta f_{\mathrm{CE}})^\top \, \nabla_f \mathcal{L}_{\mathrm{CE}}}{\lVert \nabla_f \mathcal{L}_{\mathrm{DPO}} \rVert_2 \;\lVert \nabla_\theta f_{\mathrm{DPO}} \rVert_F \;\lVert \nabla_\theta f_{\mathrm{CE}} \rVert_F \;\lVert \nabla_f \mathcal{L}_{\mathrm{CE}} \rVert_2}$$

This removes four independent scale factors (two loss sensitivities, two model
sensitivities). What remains is a pure directional quantity that can be
compared across checkpoints.

**$\chi_{\mathrm{pos}}$ as a generalized cosine.** Grouping the normalized
Jacobians into a single matrix clarifies the structure. Define unit vectors
$\hat{u} = u / \lVert u \rVert$ in output space and the normalized
cross-kernel:

$$\hat{\Theta}^{AB} = \frac{\nabla_\theta f_A \, (\nabla_\theta f_B)^\top}{\lVert \nabla_\theta f_A \rVert_F \, \lVert \nabla_\theta f_B \rVert_F}$$

Then $\chi_{\mathrm{pos}}$ takes the form:

$$\chi_{\mathrm{pos}} = \hat{u}_{\mathrm{DPO}}^\top \; \hat{\Theta}^{AB} \; \hat{u}_{\mathrm{CE}}$$

Compare to the parameter-space cosine between the full loss gradients:

$$\cos \phi = \hat{g}_{\mathrm{DPO}}^\top \; I \; \hat{g}_{\mathrm{CE}}$$

Same bilinear structure. The parameter-space cosine uses the identity $I$ as
the metric — all directions in parameter space are weighted equally.
$\chi_{\mathrm{pos}}$ replaces the identity with the normalized eNTK
$\hat{\Theta}^{AB}$, which weights directions by how much the model can
actually move along them.

By Cauchy-Schwarz ($\lVert J^\top u \rVert \leq \lVert J \rVert_F \lVert u \rVert$):

$$|\chi_{\mathrm{pos}}| \leq |\cos \phi|$$

$\chi_{\mathrm{pos}}$ is always tighter than the parameter-space cosine. The
gap between the two is exactly the information the eNTK adds: how well each
loss direction aligns with the principal directions of the Jacobian. When the
loss gradient is aligned with high-eigenvalue Jacobian modes (the bulk),
$\chi_{\mathrm{pos}} \approx \cos \phi$. When the loss gradient projects onto
low-eigenvalue modes (the tail), $\chi_{\mathrm{pos}} \ll \cos \phi$ — the
signal exists in parameter space but the model cannot act on it.

**What $\chi_{\mathrm{pos}}$ loses.** The normalization absorbs the
DPO-specific structure: the adaptive weight $w$, the norm ratio
$\lVert g_l \rVert / \lVert g_w \rVert$, and the three regimes from §2. These
are genuine insights about DPO mechanics that $\chi_{\mathrm{pos}}$ does not
capture. The two decompositions are complementary: parameter space for
DPO-specific structure, $\chi_{\mathrm{pos}}$ for scale-invariant spectral
diagnostics.

## 4. Necessary vs sufficient

The two views give a two-part diagnostic for DPO readiness:

| | condition | what it tells you | cost |
|---|---|---|---|
| **necessary** | $\cos \alpha < 1$ | the contrastive signal $g_w - g_l$ exists | two backward passes |
| **sufficient** | $\chi_{\mathrm{pos}} > 0$ | the signal lives in eigenmodes where the model can make progress | Hutchinson trace estimation |

The switch point is the sufficient condition. The gap between necessary and
sufficient is itself informative: a large gap means the preference-relevant
features are deep in the spectral tail (small eigenvalues, hard to learn); a
small gap means they are closer to the bulk (easier to learn, less DPO-specific
signal).

## TEMP: Expanding $\chi_{\mathrm{pos}}$ with the DPO contrastive structure

The DPO loss involves two forward passes — one on $(x, y_w)$ and one on
$(x, y_l)$ — producing two sets of logits with two different Jacobians:

$$J_w = \nabla_\theta f_w, \qquad J_l = \nabla_\theta f_l$$

The output-space score functions are:

$$u_w = \nabla_{f_w} \log \pi_\theta(y_w \mid x), \qquad u_l = \nabla_{f_l} \log \pi_\theta(y_l \mid x)$$

The DPO parameter gradient (from §1) decomposes via the chain rule as:

$$\nabla_\theta \mathcal{L}_{\mathrm{DPO}} = -\beta w \big(J_w^\top u_w - J_l^\top u_l\big)$$

The CE parameter gradient is:

$$\nabla_\theta \mathcal{L}_{\mathrm{CE}} = -J_w^\top u_w$$

Note that CE is evaluated on the same input $(x, y_w)$ as the preferred
branch of DPO, so the Jacobian is $J_w$ in both cases.

### Numerator of $\chi_{\mathrm{pos}}$

$$\delta L = \langle \nabla_\theta \mathcal{L}_{\mathrm{DPO}},\; \nabla_\theta \mathcal{L}_{\mathrm{CE}} \rangle = \beta w \, \langle J_w^\top u_w - J_l^\top u_l,\; J_w^\top u_w \rangle$$

$$= \beta w \Big(\lVert J_w^\top u_w \rVert^2 - u_l^\top J_l J_w^\top u_w\Big)$$

$$= \beta w \Big(u_w^\top \underbrace{J_w J_w^\top}_{=\,\Theta^{ww}} u_w \;-\; u_l^\top \underbrace{J_l J_w^\top}_{=\,\Theta^{lw}} u_w\Big)$$

Two different kernels appear:

- $\Theta^{ww} = J_w J_w^\top$ — the **self-eNTK** on the $y_w$ input. Same
  input on both sides, so this is a standard (symmetric, positive
  semi-definite) eNTK.
- $\Theta^{lw} = J_l J_w^\top$ — the **cross-eNTK** between the $y_l$ and
  $y_w$ inputs. Different inputs, so this is neither symmetric nor necessarily
  positive semi-definite.

The structure is:

$$\delta L = \beta w \Big(\underbrace{u_w^\top \Theta^{ww} u_w}_{\text{CE helps } y_w \text{ in DPO}} \;-\; \underbrace{u_l^\top \Theta^{lw} u_w}_{\text{CE helps } y_l \text{ in DPO (leakage)}}\Big)$$

The first term is how much a CE update (pushing up $y_w$) also helps the $y_w$
component of the DPO objective, mediated by the self-kernel. This is always
positive ($u_w^\top \Theta^{ww} u_w \geq 0$ since $\Theta^{ww}$ is PSD).

The second term is how much a CE update also helps $y_l$ — the "bulk leakage"
— mediated by the cross-kernel. This can have either sign.

### Concise form of $\chi_{\mathrm{pos}}$

Define unit vectors $\hat{u}_w = u_w / \lVert u_w \rVert$,
$\hat{u}_l = u_l / \lVert u_l \rVert$ and normalized kernels:

$$\hat{\Theta}^{ww} = \frac{J_w J_w^\top}{\lVert J_w \rVert_F^2}, \qquad \hat{\Theta}^{lw} = \frac{J_l J_w^\top}{\lVert J_l \rVert_F \, \lVert J_w \rVert_F}$$

The DPO output is the concatenation $f_{\mathrm{DPO}} = [f_w;\, f_l]$, so
$u_w$ and $u_l$ live in different subspaces of the concatenated output space
(logits from different forward passes). Their norms add in quadrature with no
cross-term:

$$\lVert \nabla_f \mathcal{L}_{\mathrm{DPO}} \rVert = \beta w \sqrt{\lVert u_w \rVert^2 + \lVert u_l \rVert^2}, \qquad \lVert J_{\mathrm{DPO}} \rVert_F = \sqrt{\lVert J_w \rVert_F^2 + \lVert J_l \rVert_F^2}$$

$\beta w$ cancels between numerator and denominator. Define mixing
coefficients:

$$\alpha_w = \frac{\lVert u_w \rVert}{\sqrt{\lVert u_w \rVert^2 + \lVert u_l \rVert^2}}, \quad \alpha_l = \frac{\lVert u_l \rVert}{\sqrt{\lVert u_w \rVert^2 + \lVert u_l \rVert^2}}, \qquad \alpha_w^2 + \alpha_l^2 = 1$$

$$\beta_w = \frac{\lVert J_w \rVert_F}{\sqrt{\lVert J_w \rVert_F^2 + \lVert J_l \rVert_F^2}}, \quad \beta_l = \frac{\lVert J_l \rVert_F}{\sqrt{\lVert J_w \rVert_F^2 + \lVert J_l \rVert_F^2}}, \qquad \beta_w^2 + \beta_l^2 = 1$$

Then:

$$\boxed{\chi_{\mathrm{pos}} = \alpha_w \beta_w \; \hat{u}_w^\top \hat{\Theta}^{ww} \hat{u}_w \;\;-\;\; \alpha_l \beta_l \; \hat{u}_l^\top \hat{\Theta}^{lw} \hat{u}_w}$$

Two terms. Each is a **generalized cosine** ($\hat{u}^\top \hat{\Theta} \hat{u}$)
weighted by mixing coefficients that capture the relative contribution of
$y_w$ vs $y_l$ to the DPO loss norm ($\alpha$) and Jacobian norm ($\beta$).
The self-term uses the self-kernel $\hat{\Theta}^{ww}$; the cross-term uses
the cross-kernel $\hat{\Theta}^{lw}$. The expression does not reduce further
— the two-kernel structure is an irreducible consequence of DPO having two
inputs.

### Is $\chi_{\mathrm{pos}}$ scale-free?

**Yes.** In the unsplit form, $\chi_{\mathrm{pos}}$ is a single generalized
cosine:

$$\chi_{\mathrm{pos}} = \hat{u}_{\mathrm{DPO}}^\top \; \hat{\Theta}^{AB} \; \hat{u}_{\mathrm{CE}}$$

where $\hat{u}_{\mathrm{DPO}}$ is the unit vector of the full concatenated
output-space gradient, $\hat{\Theta}^{AB} = J_{\mathrm{DPO}} J_{\mathrm{CE}}^\top / (\lVert J_{\mathrm{DPO}} \rVert_F \lVert J_{\mathrm{CE}} \rVert_F)$,
and all four norms cancel exactly. $\beta w$ cancels between numerator and
denominator of the DPO output-space gradient. Fully scale-free.

The mixing coefficients $\alpha_w, \alpha_l, \beta_w, \beta_l$ that appear in
the expanded form are *not* a scale dependence of $\chi_{\mathrm{pos}}$ itself
— they are the price of decomposing a single generalized cosine into two terms
with different kernels. The split buys interpretability (self-kernel vs
cross-kernel, the "leakage" structure) but breaks the clean normalization.
If $\alpha_w / \alpha_l$ drifts during training, that changes the *relative
weight of the two terms in the decomposition*, not $\chi_{\mathrm{pos}}$
itself.

**The sufficient condition holds:** $\chi_{\mathrm{pos}} > 0 \iff \delta L > 0$
(since $\chi_{\mathrm{loss}}$ and $\chi_{\mathrm{net}}$ are products of norms,
always positive). $\chi_{\mathrm{pos}} > 0$ means "an SFT step reduces the
DPO loss to first order."

### Observations

1. **$\beta w$ drops out.** The adaptive weight and DPO temperature do not
   affect $\chi_{\mathrm{pos}}$. They are pure scale factors. The directional
   alignment between DPO and CE depends only on the output-space score
   functions $u_w$, $u_l$ and the Jacobian geometry $J_w$, $J_l$.

2. **Self-kernel minus cross-kernel.** The numerator has the structure
   "self-overlap ($\Theta^{ww}$) minus cross-overlap ($\Theta^{lw}$)." The
   cancellation of the shared bulk depends on whether $\Theta^{ww}$ and
   $\Theta^{lw}$ share the same eigenmodes — which is a statement about how
   similar the model's internal representations are for the two completions.
   If $(x, y_w)$ and $(x, y_l)$ activate the same features (shared prompt,
   similar surface structure), the bulk eigenmodes of $\Theta^{ww}$ and
   $\Theta^{lw}$ overlap, and the subtraction kills the shared component.
   What survives is the contribution from eigenmodes where the two inputs
   diverge — can be from the bulk or the tail. 

3. **The cross-kernel is the key object.** The self-term
   $u_w^\top \Theta^{ww} u_w$ is always positive and measures how much a CE
   step helps its own objective through the eNTK. The interesting physics is
   in $u_l^\top \Theta^{lw} u_w$ — how much a CE step on $y_w$ leaks into
   $y_l$ through the cross-kernel. When the two completions share bulk
   representations, $\Theta^{lw}$ is close to $\Theta^{ww}$ in the high-
   eigenvalue subspace, and the leakage is large. When their representations
   diverge (different tail features), the cross-kernel deviates from the
   self-kernel, and the leakage shrinks.

4. **Comparison to §2.** The parameter-space result was
   $\beta w \lVert g_w \rVert (\lVert g_w \rVert - \lVert g_l \rVert \cos \alpha)$.
   The output-space result is
   $\beta w (u_w^\top \Theta^{ww} u_w - u_l^\top \Theta^{lw} u_w)$.
   Same "self minus cross" structure, but the parameter-space version
   compresses everything into a single angle $\alpha$ and two norms, while
   the output-space version resolves it through two distinct kernels. The
   distinction between self-kernel and cross-kernel is what makes the spectral
   tail hypothesis precise — it is not visible from parameter space.

## TEMP 2: The spectral filter in the eigenbasis

The bulk/tail distinction lives in the NTK eigenspectrum — parameter space is
flat and has no such structure. To see *mathematically* that the DPO
contrastive subtraction isolates the tail, we need to project onto eigenmodes.

### Shared-kernel approximation

When $y_w$ and $y_l$ are similar completions to the same prompt — the regime
DPO is designed for — the model's internal representations for the two inputs
are close. The Jacobians $J_w$ and $J_l$ differ primarily in the fine-grained
features that distinguish the two completions, not in the coarse features
driven by the shared prompt and similar surface structure. This means:

$$\Theta^{lw} = J_l J_w^\top \approx J_w J_w^\top = \Theta^{ww}$$

in the high-eigenvalue (bulk) subspace. We make this explicit by approximating
both kernels with a single $\Theta \approx \Theta^{ww} \approx \Theta^{lw}$.

This is the formal statement of "the two completions share bulk
representations." It holds when $y_w$ and $y_l$ are structurally similar
(same length, same topic, same style, differing in subtle quality). It breaks
when the two completions are structurally very different — connecting directly
to Prediction 3 in the README.

### Projection onto eigenmodes

Under the shared-kernel approximation, the numerator of $\delta L$ simplifies:

$$u_w^\top \Theta \, u_w - u_l^\top \Theta \, u_w = (u_w - u_l)^\top \Theta \, u_w$$

Decompose $\Theta$ in its eigenbasis,
$\Theta = \sum_k \lambda_k \, q_k q_k^\top$:

$$(u_w - u_l)^\top \Theta \, u_w = \sum_k \lambda_k \; (u_w \cdot q_k) \; \big((u_w - u_l) \cdot q_k\big)$$

The contribution of eigenmode $k$ to $\delta L$ is:

$$\delta L_k = \lambda_k \; \underbrace{(u_w \cdot q_k)}_{\substack{\text{how much CE} \\ \text{pushes along } q_k}} \; \underbrace{((u_w - u_l) \cdot q_k)}_{\substack{\text{how much } y_w \text{ and } y_l \\ \text{differ along } q_k}}$$

For eigenmode $k$ to contribute, **both** factors must be nonzero:

- $(u_w \cdot q_k)$: the CE loss gradient must project onto this mode. This is
  nonzero for modes the model is currently learning from the $y_w$ data.
- $((u_w - u_l) \cdot q_k)$: the output-space score functions of $y_w$ and
  $y_l$ must *differ* along this mode. This is the contrastive factor.

### Where the cancellation happens

**Bulk eigenmodes** (large $\lambda_k$): these correspond to well-learned,
coarse features — grammar, common token patterns, prompt-level structure. For
these features, $y_w$ and $y_l$ are nearly interchangeable (both are
reasonable completions to the same prompt). Their output-space score functions
agree on how these features should change:

$$u_w \cdot q_k \approx u_l \cdot q_k \quad \Longrightarrow \quad (u_w - u_l) \cdot q_k \approx 0$$

The contrastive factor kills the bulk. Even though $\lambda_k$ is large and
$(u_w \cdot q_k)$ may be large, the product vanishes because the subtraction
removes the shared component.

**Tail eigenmodes** (small $\lambda_k$): these correspond to fine-grained,
late-learned features — the subtle distinctions in helpfulness, factual
accuracy, tone, safety that separate $y_w$ from $y_l$. Here $u_w$ and $u_l$
genuinely differ:

$$(u_w - u_l) \cdot q_k \neq 0$$

These modes survive the subtraction. They are the only modes that contribute
to $\delta L$ — and therefore to $\chi_{\mathrm{pos}}$.

### Summary

The spectral filter mechanism, stated precisely: the DPO contrastive structure
$(u_w - u_l)$ projects the gradient overlap onto eigenmodes where the
preferred and dispreferred completions' loss gradients differ. Under the
shared-kernel approximation ($\Theta^{lw} \approx \Theta^{ww}$), these are
exactly the tail eigenmodes — the spectral band where bulk features have been
subtracted and only preference-relevant features remain.

The two assumptions are explicit:

1. **Shared kernel:** $\Theta^{lw} \approx \Theta^{ww}$ in the bulk. Holds
   when $y_w$ and $y_l$ are structurally similar. Breaks for dissimilar pairs
   — predicting that the spectral filter fails (Prediction 3).

2. **Shared score functions in the bulk:** $u_w \cdot q_k \approx u_l \cdot q_k$
   for bulk eigenmodes. Holds when both completions' loss gradients agree on
   well-learned features. Breaks when the completions diverge at a coarse
   level — again predicting filter failure for low-quality preference pairs.

## TEMP 3: Self-pair $\chi_{\mathrm{pos}}$ as a spectral-position probe

### Self-pair $\chi_{\mathrm{pos}}$

For a self-pair (same input on both sides, $A = B$):

$$\chi_{\mathrm{pos}}(A, A) = \frac{\delta L(A, A)}{\chi_{\mathrm{loss}}^{AA} \cdot \chi_{\mathrm{net}}^{AA}} = \frac{u_A^\top \Theta^{AA} u_A}{\lVert u_A \rVert^2 \cdot \mathrm{Tr}(\Theta^{AA})}$$

In the eigenbasis $\Theta^{AA} = \sum_k \lambda_k q_k q_k^\top$:

$$\chi_{\mathrm{pos}}(A, A) = \sum_k \underbrace{\frac{\lambda_k}{\sum_j \lambda_j}}_{p_k} \cdot \underbrace{\frac{(u_A \cdot q_k)^2}{\lVert u_A \rVert^2}}_{\gamma_k^2}$$

A convex combination: $\sum_k p_k = \sum_k \gamma_k^2 = 1$, so $\chi_{\mathrm{pos}}(A,A) \in [0, 1]$.

**Interpretation:** $\chi_{\mathrm{pos}}(A,A)$ is the **spectral position** of the
loss gradient $u_A$. It tells you where $u_A$ lives in the NTK eigenspectrum,
weighted by eigenvalue mass:

- $\chi_{\mathrm{pos}}(A,A) \to 1$: $u_A$ aligns with the dominant bulk
  eigenmodes — the loss gradient exploits well-resolved, high-eigenvalue
  directions.
- $\chi_{\mathrm{pos}}(A,A) \to 0$: $u_A$ aligns with the tail — the loss
  gradient operates on weak, low-eigenvalue directions.

### Cauchy-Schwarz bound

By Cauchy-Schwarz on $\delta L(A,B) = \langle g_A, g_B \rangle$ in parameter
space, $|\delta L(A,B)|^2 \leq \delta L(A,A) \cdot \delta L(B,B)$. The
$\chi_{\mathrm{loss}}$ and $\chi_{\mathrm{net}}$ factors in the decomposition
satisfy $\chi_{\mathrm{loss}}^{AB} = \sqrt{\chi_{\mathrm{loss}}^{AA} \chi_{\mathrm{loss}}^{BB}}$ and
$\chi_{\mathrm{net}}^{AB} = \sqrt{\chi_{\mathrm{net}}^{AA} \chi_{\mathrm{net}}^{BB}}$,
so they cancel identically. The Cauchy-Schwarz bound translates to:

$$\frac{\chi_{\mathrm{pos}}(A,B)}{\sqrt{\chi_{\mathrm{pos}}(A,A) \cdot \chi_{\mathrm{pos}}(B,B)}} = \frac{\langle g_A, g_B \rangle}{\lVert g_A \rVert \lVert g_B \rVert} = \cos(\angle(g_A, g_B)) \in [-1, 1]$$

The ratio is **exactly the parameter-space cosine**. Bounded in $[-1, 1]$,
scale-free (all norms cancel), with no residual terms.

**This trick only works for $\chi_{\mathrm{pos}}$.** For $\chi_{\mathrm{loss}}$
and $\chi_{\mathrm{net}}$, the cross-pair is the geometric mean of the
self-pairs *by construction* ($\chi_{\mathrm{loss}}^{AB} = \lVert u_A \rVert \lVert u_B \rVert$), so the ratio is always exactly 1. No
information. Only $\chi_{\mathrm{pos}}$ carries directional content that
Cauchy-Schwarz can turn into a bounded cosine.

### Two diagnostics, different roles

Putting this together, we have two scale-free observables derived from
self-pair and cross-pair $\chi_{\mathrm{pos}}$:

**(a) Spectral-position ratio:**

$$R_{\mathrm{spec}} = \frac{\chi_{\mathrm{pos}}(\mathrm{DPO}, \mathrm{DPO})}{\chi_{\mathrm{pos}}(\mathrm{CE}, \mathrm{CE})}$$

Both numerator and denominator live in $[0, 1]$. The ratio measures how much
more tail-concentrated DPO is relative to CE. The hypothesis predicts
$R_{\mathrm{spec}} \ll 1$ — DPO's loss gradient lives in the tail, CE's in
the bulk. This is a **spectral claim**: it tests where each objective sits in
the spectrum, independent of their directional alignment.

**(b) Normalized cross-pair:**

$$R_{\mathrm{dir}} = \frac{\chi_{\mathrm{pos}}(\mathrm{DPO}, \mathrm{CE})}{\sqrt{\chi_{\mathrm{pos}}(\mathrm{DPO}, \mathrm{DPO}) \cdot \chi_{\mathrm{pos}}(\mathrm{CE}, \mathrm{CE})}} = \cos(\angle(g_{\mathrm{DPO}}, g_{\mathrm{CE}}))$$

Bounded in $[-1, 1]$. The parameter-space cosine between the two loss
gradients. This is a **directional claim**: given wherever each objective
lives, how aligned are they? Reaches 1 when the DPO and CE gradients are
colinear.

**Advantages and limitations:**

| observable | measures | advantage | limitation |
|---|---|---|---|
| $R_{\mathrm{spec}}$ | relative spectral position of DPO vs CE | directly tests the "DPO in the tail" hypothesis; cleanly separates spectral from directional | doesn't tell you whether the two gradients help each other |
| $R_{\mathrm{dir}}$ | directional alignment in parameter space | bounded and interpretable (pure cosine); directly tells you if SFT helps DPO | treats all directions equally; doesn't distinguish "aligned in the bulk" from "aligned in the tail" |
| $\chi_{\mathrm{pos}}(\mathrm{DPO}, \mathrm{CE})$ (unnormalized cross) | spectral-weighted alignment | carries both directional and spectral information | its magnitude conflates the two effects; harder to interpret |

**Best combination.** Use $R_{\mathrm{spec}}$ as the main test of the hypothesis
(does DPO live in the tail?) and $R_{\mathrm{dir}}$ as the sanity check
(when the sign flips, is the alignment actually improving?). The unnormalized
$\chi_{\mathrm{pos}}(\mathrm{DPO}, \mathrm{CE})$ is redundant once you have
these two — it's $R_{\mathrm{dir}} \cdot \sqrt{\chi_{\mathrm{pos}}(\mathrm{DPO},\mathrm{DPO}) \cdot \chi_{\mathrm{pos}}(\mathrm{CE},\mathrm{CE})}$,
a product of the two.
