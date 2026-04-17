# DPO Gradient Analysis: π(y|x), Log-Ratios, and Contrastive Diagnostics

A conversation exploring the mechanics of Direct Preference Optimization (DPO), focusing on the gradient geometry of the DPO loss relative to standard cross-entropy, and implications for training diagnostics.

---

## What does π(y|x) mean as a neural network?

π(y|x) is just the autoregressive language model itself, viewed through the lens of RL/decision theory where we call it a "policy." Concretely, for a prompt x and a complete response y = (y₁, y₂, ..., y_T), the model gives you:

$$\pi_\theta(y|x) = \prod_t \pi_\theta(y_t \mid x, y_1, \ldots, y_{t-1})$$

Each factor is a softmax over the vocabulary at timestep t — exactly what a transformer already computes at every token position. So π(y|x) is just the joint probability of the full completion y, computed as the product of next-token probabilities. There's no new architecture here; the "policy" *is* the language model. The paper explicitly states this: the policy network represents both the language model and the implicit reward.

For a specific, fixed completion y given a specific prompt x, π_θ(y|x) is a single scalar number ∈ [0, 1] — the probability that this model would generate exactly that sequence y.

In practice you never work with it directly but with its log, which you get by summing the per-token log-probs:

$$\log \pi_\theta(y|x) = \sum_t \log \pi_\theta(y_t \mid x, y_1, \ldots, y_{t-1})$$

Each term in that sum is just the log-softmax output at position t, evaluated at the token y_t that actually appears in the completion. So computing log π_θ(y|x) is a single forward pass through the transformer with the concatenated sequence [x; y], reading off the log-prob of each response token, and summing them up. One scalar out.

---

## What does π_θ / π_ref mean in the DPO loss?

The DPO loss (Eq. 7 in the paper) contains terms like:

$$\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)}$$

This ratio is computed in log-space, so in practice you're just computing the difference of two log-probabilities:

$$\log \pi_\theta(y|x) - \log \pi_{\text{ref}}(y|x)$$

Each of these is a scalar you get by summing the per-token log-probabilities over the full sequence y. You run the same sequence y through two models — your trainable model (parameterized by θ) and a frozen copy of the reference/SFT model — and take the difference of their total log-probs.

**Mechanistically, this log-ratio acts as an implicit reward.** The paper derives (Eq. 5) that for the optimal policy, the reward can be written as:

$$r(x, y) = \beta \log \frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \log Z(x)$$

So the quantity β log [π_θ(y|x) / π_ref(y|x)] is literally the model's current estimate of the reward (up to the partition function, which cancels in the Bradley-Terry preference model since it only depends on x). The ratio measures how much the trainable policy has diverged from the reference on this specific completion.

**Why this works without a separate reward model:** The classic RLHF pipeline trains an explicit reward r_ϕ(x,y) and then runs PPO against it. The DPO insight is that you can reparameterize r in terms of the optimal policy itself (the change of variables in Eq. 5), substitute into the Bradley-Terry preference model, and the partition function Z(x) cancels. What you're left with is a loss that only involves log-probability ratios from two forward passes — one through π_θ, one through π_ref — no reward model, no RL loop.

---

## Comparing DPO and CE gradients

The CE loss on y_w is just:

$$\mathcal{L}_{\text{CE}} = -\log \pi_\theta(y_w | x)$$

with gradient:

$$\nabla_\theta \mathcal{L}_{\text{CE}} = -\nabla_\theta \log \pi_\theta(y_w | x)$$

The DPO gradient (from the paper's Section 4) is:

$$\nabla_\theta \mathcal{L}_{\text{DPO}} = -\beta \cdot \sigma(\hat{r}_\theta(x, y_l) - \hat{r}_\theta(x, y_w)) \cdot \left[\nabla_\theta \log \pi_\theta(y_w | x) - \nabla_\theta \log \pi_\theta(y_l | x)\right]$$

where $\hat{r}_\theta(x, y) = \beta \log \pi_\theta(y|x)/\pi_{\text{ref}}(y|x)$.

Two key differences:

**1. The contrastive term.** DPO doesn't just push up y_w — it simultaneously pushes down y_l via the $-\nabla_\theta \log \pi_\theta(y_l | x)$ term. CE is purely "imitative," DPO is contrastive.

**2. The adaptive weight.** The sigmoid prefactor $\sigma(\hat{r}_\theta(x, y_l) - \hat{r}_\theta(x, y_w))$ modulates the entire gradient based on how wrong the model's current implicit reward ranking is. When the model already correctly assigns higher implicit reward to y_w than y_l, this weight goes toward 0. When the model gets it wrong, the weight is close to 1. CE has no such self-regulation.

---

## Dot product of DPO and CE gradients

Let $g_w = \nabla_\theta \log \pi_\theta(y_w|x)$ and $g_l = \nabla_\theta \log \pi_\theta(y_l|x)$, and $w = \sigma(\hat{r}_\theta(x, y_l) - \hat{r}_\theta(x, y_w))$.

Then:

$$\nabla \mathcal{L}_{\text{CE}} = -g_w$$
$$\nabla \mathcal{L}_{\text{DPO}} = -\beta w (g_w - g_l)$$

Their dot product:

$$\langle \nabla \mathcal{L}_{\text{DPO}}, \nabla \mathcal{L}_{\text{CE}} \rangle = \beta w \cdot \langle g_w - g_l, g_w \rangle = \beta w \cdot (\|g_w\|^2 - \langle g_l, g_w \rangle)$$

Which can be rewritten as:

$$\beta w \cdot \|g_w\| \cdot \left(\|g_w\| - \|g_l\| \cos \alpha\right)$$

where α is the angle between g_w and g_l.

### Three regimes

- **y_w and y_l very different** (α ≈ 90°): ⟨g_l, g_w⟩ vanishes, dot product ≈ βw‖g_w‖², always positive. DPO and CE push in roughly the same direction.

- **y_w and y_l similar** (α small, g_l ≈ g_w): dot product ≈ βw(‖g_w‖² − ‖g_w‖‖g_l‖), can be very small or even negative if ‖g_l‖ > ‖g_w‖. Pushing down y_l actively fights pushing up y_w.

- **Anticorrelated score functions** (α > 90°): dot product is even larger than the orthogonal case. Pushing down y_l actively *helps* push up y_w.

---

## Does normalization yield 1 − cos(α)?

Normalizing $\|g_w\|^2 - \langle g_l, g_w \rangle$ by $\|g_w\| \cdot \|g_l\|$ gives:

$$\frac{\|g_w\|}{\|g_l\|} - \cos(\alpha)$$

This equals 1 − cos(α) **only if** ‖g_w‖ = ‖g_l‖. Normalizing by ‖g_w‖² instead gives:

$$1 - \frac{\|g_l\|}{\|g_w\|} \cos(\alpha)$$

Again only 1 − cos(α) when norms match. The clean form requires an equal-norm assumption. The deviation from 1 − cos(α) is entirely controlled by the norm ratio — a per-sample quantity that evolves over training.

### Is the result typically > 0?

Yes. For the expression to go negative requires:

$$\frac{\|g_l\|}{\|g_w\|} \cdot \cos(\alpha) > 1$$

This needs *both* cos(α) being large (similar gradient directions) *and* ‖g_l‖ > ‖g_w‖ — a fairly restrictive joint condition.

Over DPO training this might become *more* likely though: as the model fits y_w better, ‖g_w‖ shrinks while ‖g_l‖ can grow. The norm ratio ‖g_l‖/‖g_w‖ drifts upward. Whether it crosses the threshold depends on how fast cos(α) decreases as the model learns to separate the two completions.

---

## Readiness diagnostic: When is the model ready for DPO?

At initialization of DPO, π_θ = π_ref, so the implicit reward r̂_θ = 0 for everything, the sigmoid weight is σ(0) = 0.5 uniformly, and the entire signal comes from g_w − g_l. If the SFT model hasn't yet learned representations that distinguish y_w from y_l, then g_w ≈ g_l, and the effective DPO gradient g_w − g_l is a tiny, noisy vector.

**Proposed diagnostic:** Measure cos(α) between g_w and g_l across the preference dataset *before* starting DPO, as a function of SFT training. When cos(α) drops meaningfully below 1, the contrastive signal has something to grab onto.

---

## Is g_w ≈ g_l actually valid after SFT?

The assumption is actually pretty weak for several reasons:

**High dimensionality alone kills it.** Gradients live in a space with millions/billions of dimensions. Two vectors in such a space need to share a *lot* of structure to have cos(α) anywhere near 1 — random vectors in high-d are nearly orthogonal by concentration of measure. Even if y_w and y_l are both reasonable completions to the same prompt, the token-level differences propagate through different attention patterns, different MLP activations, different embedding/unembedding rows.

**SFT creates explicit asymmetry.** SFT is trained on y_w (or y_w-like completions). After SFT, the model has *fit* y_w, meaning ‖g_w‖ is relatively small and the gradient points along residual fine-tuning directions. y_l was never trained on, so g_l is larger and points wherever the model's current deficit on that sequence happens to be.

**The shared part is the prompt.** The main source of correlation is that both gradients flow through the same prompt prefix representations. But the prompt contribution is shared and approximately cancels in g_w − g_l anyway, so the DPO contrastive signal is dominated by the divergent part of the completions.

**Conclusion:** cos(α) is probably already meaningfully below 1 right after SFT, meaning DPO has a nontrivial contrastive signal from the start. The more interesting diagnostic might not be "is cos(α) small enough" but rather tracking *how* the decomposition into norm ratio and angle evolves during DPO itself — that trajectory tells you about the training dynamics rather than just a readiness threshold.
