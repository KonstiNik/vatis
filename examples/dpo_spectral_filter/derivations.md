# Derivations

Mathematical details supporting the claims in [README.md](README.md).

## 1. The DPO gradient

A language model $\pi_\theta(y \mid x)$ assigns probability to a completion
$y$ given prompt $x$ via the autoregressive factorization. The DPO loss
(Rafailov et al. 2023, Eq. 7) is:

$$\mathcal{L}_{\mathrm{DPO}} = -\log \sigma\!\Big(\beta \big[\log \pi_\theta(y_w \mid x) - \log \pi_\theta(y_l \mid x) - \log \pi_{\mathrm{ref}}(y_w \mid x) + \log \pi_{\mathrm{ref}}(y_l \mid x)\big]\Big)$$

Define the adaptive weight
$w = \sigma\!\big(\hat{r}_\theta(x, y_l) - \hat{r}_\theta(x, y_w)\big)$ with
implicit reward $\hat{r}_\theta(x, y) = \beta \log \frac{\pi_\theta(y \mid x)}{\pi_{\mathrm{ref}}(y \mid x)}$.
The gradient is:

$$\nabla_\theta \mathcal{L}_{\mathrm{DPO}} = -\beta \, w \, \big[\nabla_\theta \log \pi_\theta(y_w \mid x) - \nabla_\theta \log \pi_\theta(y_l \mid x)\big]$$

Compared to the CE gradient
$\nabla_\theta \mathcal{L}_{\mathrm{CE}} = -\nabla_\theta \log \pi_\theta(y_w \mid x)$,
the DPO gradient has two defining properties: it is **contrastive** (a
difference of score functions, pushing up $y_w$ and pushing down $y_l$) and
**self-regulating** (the weight $w$ vanishes when the model already prefers
$y_w$).

## 2. $\chi_{\mathrm{pos}}$ from the chain rule

The gradient overlap between two loss functions is
$\delta L(A, B) = \langle \nabla_\theta \mathcal{L}_A, \nabla_\theta \mathcal{L}_B \rangle$.
Applying the chain rule $\nabla_\theta \mathcal{L} = (\nabla_\theta f)^\top \nabla_f \mathcal{L}$
to each side and normalizing by the norm of each of the four factors yields:

$$\chi_{\mathrm{pos}}(A, B) = \frac{(\nabla_f \mathcal{L}_A)^\top \, \nabla_\theta f_A \, (\nabla_\theta f_B)^\top \, \nabla_f \mathcal{L}_B}{\lVert \nabla_f \mathcal{L}_A \rVert_2 \;\lVert \nabla_\theta f_A \rVert_F \;\lVert \nabla_\theta f_B \rVert_F \;\lVert \nabla_f \mathcal{L}_B \rVert_2}$$

Each of the four norms is a scale factor that drifts by orders of magnitude
during training (loss gradients shrink as the model fits; Jacobian norms
change as the weight geometry evolves). Dividing them out leaves a pure
directional quantity that can be compared across checkpoints.

Equivalently, with unit vectors $\hat{u} = \nabla_f \mathcal{L} / \lVert \nabla_f \mathcal{L} \rVert$
and the normalized cross-kernel
$\hat{\Theta}^{AB} = \nabla_\theta f_A (\nabla_\theta f_B)^\top / (\lVert \nabla_\theta f_A \rVert_F \lVert \nabla_\theta f_B \rVert_F)$:

$$\chi_{\mathrm{pos}}(A, B) = \hat{u}_A^\top \; \hat{\Theta}^{AB} \; \hat{u}_B$$

A bilinear form with the normalized empirical NTK as the metric — a
generalized cosine.

## 3. Self-pair $\chi_{\mathrm{pos}}$: spectral position

For a self-pair ($A = B$), the cross-kernel becomes the self-eNTK
$\Theta^{AA} = J_A J_A^\top$, which is symmetric and positive semi-definite.
Eigendecomposing $\Theta^{AA} = \sum_k \lambda_k q_k q_k^\top$:

$$\chi_{\mathrm{pos}}(A, A) = \frac{u_A^\top \Theta^{AA} u_A}{\lVert u_A \rVert^2 \cdot \mathrm{Tr}(\Theta^{AA})} = \sum_k \underbrace{\frac{\lambda_k}{\sum_j \lambda_j}}_{p_k} \cdot \underbrace{\frac{(u_A \cdot q_k)^2}{\lVert u_A \rVert^2}}_{\gamma_k^2}$$

A convex combination of $\gamma_k^2$ weighted by $p_k$, with
$\sum_k p_k = \sum_k \gamma_k^2 = 1$. Therefore $\chi_{\mathrm{pos}}(A, A) \in [0, 1]$.

**Interpretation.** $\chi_{\mathrm{pos}}(A, A)$ is the **spectral position** of
the loss gradient $u_A$:

- $\chi_{\mathrm{pos}}(A, A) \to 1$: $u_A$ aligns with the bulk (high-$\lambda_k$
  eigenmodes). The loss gradient exploits well-resolved, dominant directions.
- $\chi_{\mathrm{pos}}(A, A) \to 0$: $u_A$ aligns with the tail (low-$\lambda_k$
  eigenmodes). The loss gradient operates on weak, fine-grained directions.

## 4. The two diagnostics

### Cauchy-Schwarz identity

$\delta L(A, B) = \langle g_A, g_B \rangle$ satisfies Cauchy-Schwarz:
$|\delta L(A, B)|^2 \leq \delta L(A, A) \cdot \delta L(B, B)$. The
$\chi_{\mathrm{loss}}$ and $\chi_{\mathrm{net}}$ factors in the decomposition
satisfy
$\chi_{\mathrm{loss}}^{AB} = \sqrt{\chi_{\mathrm{loss}}^{AA} \chi_{\mathrm{loss}}^{BB}}$
and $\chi_{\mathrm{net}}^{AB} = \sqrt{\chi_{\mathrm{net}}^{AA} \chi_{\mathrm{net}}^{BB}}$
by construction (geometric means of norms), so they cancel identically:

$$\frac{\chi_{\mathrm{pos}}(A, B)}{\sqrt{\chi_{\mathrm{pos}}(A, A) \cdot \chi_{\mathrm{pos}}(B, B)}} = \frac{\delta L(A, B)}{\sqrt{\delta L(A, A) \cdot \delta L(B, B)}} = \frac{\langle g_A, g_B \rangle}{\lVert g_A \rVert \lVert g_B \rVert}$$

The normalized cross-pair $\chi_{\mathrm{pos}}$ equals the parameter-space
cosine — exactly, no residual norms. Bounded in $[-1, 1]$.

### Definitions

**Spectral-position ratio (tests the hypothesis):**

$$R_{\mathrm{spec}} = \frac{\chi_{\mathrm{pos}}(\mathcal{L}_{\mathrm{DPO}}, \mathcal{L}_{\mathrm{DPO}})}{\chi_{\mathrm{pos}}(\mathcal{L}_{\mathrm{CE}}, \mathcal{L}_{\mathrm{CE}})}$$

Both numerator and denominator live in $[0, 1]$ (§3). Each term measures
where a loss gradient sits in *its own* kernel's spectrum — DPO's
self-eNTK for the numerator, CE's for the denominator. The hypothesis
predicts $R_{\mathrm{spec}} > 1$, rising during SFT: late-SFT CE has
exhausted its bulk and lives in its own tail
($\chi_{\mathrm{pos}}(\mathcal{L}_{\mathrm{CE}}, \mathcal{L}_{\mathrm{CE}})$
small — standard pretraining dynamics continuing), while the fresh DPO
objective has a loss gradient that exploits the bulk of its own kernel
($\chi_{\mathrm{pos}}(\mathcal{L}_{\mathrm{DPO}}, \mathcal{L}_{\mathrm{DPO}})$
large).

**Directional alignment (readiness diagnostic):**

$$R_{\mathrm{dir}} = \frac{\chi_{\mathrm{pos}}(\mathcal{L}_{\mathrm{DPO}}, \mathcal{L}_{\mathrm{CE}})}{\sqrt{\chi_{\mathrm{pos}}(\mathcal{L}_{\mathrm{DPO}}, \mathcal{L}_{\mathrm{DPO}}) \cdot \chi_{\mathrm{pos}}(\mathcal{L}_{\mathrm{CE}}, \mathcal{L}_{\mathrm{CE}})}} = \cos(\angle(g_{\mathrm{DPO}}, g_{\mathrm{CE}}))$$

Bounded in $[-1, 1]$. The parameter-space cosine between the two loss
gradients. The sign tells you whether an SFT step reduces the DPO loss to
first order.

### Cost

- $R_{\mathrm{spec}}$: requires $\chi_{\mathrm{pos}}$ self-pair values →
  Hutchinson trace estimation through vatis.
- $R_{\mathrm{dir}}$: reduces to $\langle g_A, g_B \rangle / (\lVert g_A \rVert \lVert g_B \rVert)$
  — two backward passes and a dot product. **No Hutchinson, no Jacobian norms,
  no per-sample machinery.** Can be evaluated at every SFT checkpoint at
  negligible cost.

### Complementary roles

| observable | tests | advantage | limitation |
|---|---|---|---|
| $R_{\mathrm{spec}}$ | hypothesis: CE has exhausted its own bulk while DPO has untapped bulk in its own kernel | directly tests the mechanism; cleanly separates spectral structure from directional alignment | doesn't tell you whether SFT actually helps DPO |
| $R_{\mathrm{dir}}$ | readiness: does SFT help DPO? | cheap to compute; direct practical signal | treats all directions equally; can't distinguish "aligned in the bulk" from "aligned in the tail" |

The two are logically independent. $R_{\mathrm{dir}}$ can indicate readiness
even if the spectral-tail hypothesis fails; $R_{\mathrm{spec}}$ can support
the hypothesis independently of whether $R_{\mathrm{dir}}$ reaches any useful
threshold. See the 2×2 outcome table in the README.

## 5. The spectral filter mechanism

The hypothesis — that the DPO contrastive structure cancels the bulk and
isolates the tail — can be made precise under one approximation. When $y_w$
and $y_l$ are structurally similar completions to the same prompt, the
model's Jacobians at the two inputs produce similar kernels in the
high-eigenvalue subspace:

$$\Theta^{lw} = J_l J_w^\top \approx J_w J_w^\top = \Theta^{ww} \equiv \Theta$$

This is the formal statement of "the two completions share bulk
representations." It is the regime DPO is designed for — structurally
similar pairs differing in subtle quality.

Under this approximation, the numerator of $\chi_{\mathrm{pos}}(\mathcal{L}_{\mathrm{DPO}}, \mathcal{L}_{\mathrm{CE}})$
decomposes in the eigenbasis of $\Theta$ as:

$$u_w^\top \Theta \, u_w - u_l^\top \Theta \, u_w = (u_w - u_l)^\top \Theta \, u_w = \sum_k \lambda_k \; (u_w \cdot q_k) \; \big((u_w - u_l) \cdot q_k\big)$$

The contribution of eigenmode $k$ has two factors. Both must be nonzero for
the mode to contribute:

- $(u_w \cdot q_k)$: the CE loss gradient projects onto this mode.
- $((u_w - u_l) \cdot q_k)$: the output-space score functions of $y_w$ and
  $y_l$ **differ** along this mode. This is the contrastive factor.

**Bulk eigenmodes** ($\lambda_k$ large): $y_w$ and $y_l$ share well-learned
coarse features, so $u_w \approx u_l$ in these directions and the contrastive
factor vanishes. The bulk is killed by the subtraction — even though
$\lambda_k$ is large and the CE projection is nonzero.

**Tail eigenmodes** ($\lambda_k$ small): $y_w$ and $y_l$ differ along these
modes — the fine-grained features where preference matters. The contrastive
factor is nonzero. These modes survive.

The filter mechanism depends on two assumptions:

1. **Shared kernel:** $\Theta^{lw} \approx \Theta^{ww}$ in the bulk. Holds
   when $y_w$ and $y_l$ are structurally similar. Breaks for dissimilar
   pairs — predicting filter failure there.
2. **Shared bulk score functions:** $u_w \cdot q_k \approx u_l \cdot q_k$ for
   bulk eigenmodes. Holds when both completions' loss gradients agree on
   well-learned features. Breaks when they diverge at a coarse level.

Both assumptions are physically motivated and both predict the same failure
mode: the filter degrades for low-quality preference pairs where $y_w$ and
$y_l$ are structurally very different.
