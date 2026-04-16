# DPO as a spectral filter

## Hypothesis

Preference optimization (DPO) acts as a spectral filter on the empirical
Neural Tangent Kernel (eNTK). By contrasting a preferred response $y_w$ against
a dispreferred response $y_l$ to the same prompt, the DPO gradient subtracts out
the bulk eigenmodes shared by both responses and isolates the spectral tail —
the late-learned, fine-grained directions in parameter space that encode
preference-relevant distinctions.

### Setup in LNA language

The LNA decomposition factors the linearized loss change between two
distributions $A$ and $B$ as:

$$\delta L(A, B) = \chi_{\mathrm{loss}} \cdot \chi_{\mathrm{net}} \cdot \chi_{\mathrm{pos}}$$

where $\chi_{\mathrm{pos}}$ captures the spectral overlap: which eigenmodes of
the eNTK are shared between the gradient directions of $A$ and $B$, weighted by
eigenvalue.

Now consider two eval batches at a given checkpoint:

- **$A$ = preferred completions** ($y_w$ given prompt $x$)
- **$B$ = dispreferred completions** ($y_l$ given same prompt $x$)

Because $y_w$ and $y_l$ respond to the same prompt and share most surface-level
structure (syntax, vocabulary, topic), their per-sample gradients project onto
nearly the same bulk eigenmodes of the eNTK. The spectral bulk — large
eigenvalues, well-learned features — is shared. The spectral tail — small
eigenvalues, late-learned or unresolved features — is where the two responses
diverge.

### The SFT gradient lives in the bulk

SFT minimizes cross-entropy on $y_w$:

$$\mathbf{g}_{\mathrm{SFT}} = J^\top \nabla_f \mathcal{L}(y_w)$$

This gradient is dominated by the large eigenvalues of $J J^\top$ (the eNTK).
The model is pushed to "predict these tokens better", but the bulk eigenmodes
are already resolved, so the gradient mostly reinforces what the model already
knows. Diminishing returns.

### The DPO gradient isolates the tail

The DPO loss involves the difference of log-probabilities:

$$\mathcal{L}_{\mathrm{DPO}} = -\log \sigma\!\Big(\beta \big[\log \pi_\theta(y_w \mid x) - \log \pi_\theta(y_l \mid x) - \log \pi_{\mathrm{ref}}(y_w \mid x) + \log \pi_{\mathrm{ref}}(y_l \mid x)\big]\Big)$$

The gradient is proportional to:

$$\mathbf{g}_{\mathrm{DPO}} \propto J^\top \Big(\nabla_f \log \pi_\theta(y_w \mid x) - \nabla_f \log \pi_\theta(y_l \mid x)\Big)$$

The subtraction cancels the shared bulk projection. What survives is the
component along eigenmodes where $y_w$ and $y_l$ differ — the spectral tail.

### Consequences

1. **Why DPO outperforms more SFT on good data.** SFT keeps pushing on
   already-resolved bulk modes. DPO targets the unresolved tail where the
   model still has room to improve.

2. **Why DPO is unstable.** Operating in the tail means small eigenvalues,
   small gradients, high relative variance. Stochastic estimation (minibatch
   noise, any Hutchinson-style approximation) has proportionally larger
   errors. This predicts sensitivity to learning rate, $\beta$, and batch
   composition.

3. **Why the reference model matters.** The KL penalty to $\pi_{\mathrm{ref}}$
   anchors the bulk eigenmodes. Without it, the tail-only gradient can drag
   bulk modes through nonlinear coupling, degrading general capabilities. The
   reference model pins the bulk so only the tail moves. Consistent with the
   empirical observation that DPO without KL regularization degrades fluency.

4. **When DPO should fail.** The bulk cancellation relies on $y_w$ and $y_l$
   being spectrally close in the bulk — i.e., structurally similar completions
   that differ in subtle quality. When pairs are structurally very different (a
   500-token helpful response vs a 20-token refusal), the bulk doesn't cancel
   cleanly and DPO behaves more like SFT-on-the-difference. This may explain
   why DPO on low-quality preference data (where pairs are structurally
   dissimilar) tends to underperform.

## Testable predictions

Using vatis, we can measure $\chi_{\mathrm{pos}}(A, B)$ where $A$ = preferred
and $B$ = dispreferred completions at the same prompt, evaluated at different
checkpoints during training.

### Prediction 1: $\chi_{\mathrm{pos}}(y_w, y_l)$ is small at the SFT checkpoint

At the end of SFT, the bulk is resolved but the tail is not. The preferred and
dispreferred responses share the bulk (by construction — same prompt, similar
surface structure), so their gradient overlap lives mostly in well-resolved
eigenmodes. $\chi_{\mathrm{pos}}$, which weights overlap by the spectral
position, should be small because the tail modes where $y_w$ and $y_l$ diverge
have small eigenvalues.

### Prediction 2: $\chi_{\mathrm{pos}}(y_w, y_l)$ shifts during DPO

As DPO training progresses, the optimizer resolves the tail modes that
distinguish $y_w$ from $y_l$. At intermediate DPO checkpoints,
$\chi_{\mathrm{pos}}(y_w, y_l)$ should show a transient bump — the same
"spectral arrival" signature we looked for in the spectral\_tail experiment.
After those modes are resolved, $\chi_{\mathrm{pos}}$ should decrease again.

### Prediction 3: structurally dissimilar pairs weaken the filter

For preference pairs where $y_w$ and $y_l$ differ substantially in length,
topic, or format, the bulk cancellation is incomplete.
$\chi_{\mathrm{pos}}(y_w, y_l)$ should be larger (more bulk leakage) and less
predictive of DPO training dynamics.

## Using $\chi_{\mathrm{pos}}$ with the DPO loss directly

The predictions above use the CE loss inside vatis: $y_w$ and $y_l$ are
separate eval batches, and we measure spectral overlap of their CE gradients.
But we can also plug $\mathcal{L}_{\mathrm{DPO}}$ itself into vatis as the
loss function. This gives access to a different set of observables that probe
the hypothesis from the inside — asking not "how do the two completions relate
in CE-space?" but "where does the DPO gradient actually live in the spectrum?"

To set up notation: write the eNTK eigendecomposition as
$\Theta = \sum_k \lambda_k \mathbf{v}_k \mathbf{v}_k^\top$ and the
logit-space gradient of a loss $\mathcal{L}$ as
$\mathbf{u} = \nabla_f \mathcal{L}$. Then:

$$\chi_{\mathrm{pos}}(A, B) = \frac{\sum_k \lambda_k \, (\hat{\mathbf{u}}_A \cdot \mathbf{v}_k)(\hat{\mathbf{u}}_B \cdot \mathbf{v}_k)}{\sum_k \lambda_k}$$

where $\hat{\mathbf{u}} = \mathbf{u} / \lVert \mathbf{u} \rVert$. This is
the eigenvalue-weighted cosine alignment between the two logit-space gradient
directions, normalized by the trace. It tells you: in which spectral band do
$A$ and $B$ overlap, and how large are the eigenvalues there?

### Measurement A: DPO self-pair — $\chi_{\mathrm{pos}}(\mathcal{L}_{\mathrm{DPO}}, \mathcal{L}_{\mathrm{DPO}})$

The self-pair $\chi_{\mathrm{pos}}$ is positive by definition — it is
$\sum_k \lambda_k (\hat{\mathbf{u}}_{\mathrm{DPO}} \cdot \mathbf{v}_k)^2 / \sum_k \lambda_k$,
a sum of non-negative terms. Its magnitude tells us where the DPO gradient
sits in the NTK spectrum: large means the gradient projects onto bulk
eigenvectors (large $\lambda_k$), small means it projects onto tail
eigenvectors (small $\lambda_k$).

The interesting structure is the temporal dynamics during SFT, which split into
two regimes:

1. **Early SFT — resolving the bulk.** The optimizer is reshuffling the
   dominant eigenmodes: the eigenbasis is rotating rapidly, eigenvalues are
   shifting. The DPO gradient $\mathbf{u}_{\mathrm{DPO}}$ projects onto modes
   that are currently unresolved or being restructured. Its projection is
   unstable — the eigenvectors it aligns with keep moving under it. This
   predicts **fluctuations** in
   $\chi_{\mathrm{pos}}(\mathcal{L}_{\mathrm{DPO}}, \mathcal{L}_{\mathrm{DPO}})$
   during the early SFT phase. The signal size is unclear a priori — the DPO
   gradient may be small in norm (the model hasn't developed the features
   preference optimization cares about), so the fluctuations could be
   dominated by Hutchinson noise.

2. **Late SFT — approaching the tail.** The bulk is stabilized, the large
   eigenvalues and their eigenvectors have settled. The DPO gradient now gets a
   clean projection onto well-defined tail modes. If these tail modes have
   growing eigenvalues (the model is starting to develop the fine-grained
   features that distinguish $y_w$ from $y_l$), then
   $\chi_{\mathrm{pos}}(\mathcal{L}_{\mathrm{DPO}}, \mathcal{L}_{\mathrm{DPO}})$
   should increase — the DPO loss is picking up a clean spectral signal. This
   is the regime where the model is ready for preference optimization.

### Measurement B: DPO vs CE — $\chi_{\mathrm{pos}}(\mathcal{L}_{\mathrm{DPO}}, \mathcal{L}_{\mathrm{CE}}(y_w))$

Unlike the self-pair, the cross-pair
$\chi_{\mathrm{pos}}(\mathcal{L}_{\mathrm{DPO}}, \mathcal{L}_{\mathrm{CE}})$
can be negative. The sign carries physical meaning: it measures whether a CE
update (an SFT step on $y_w$) improves or hurts the DPO loss.

- $\chi_{\mathrm{pos}} > 0$: the SFT gradient and DPO gradient project onto
  the same eigenmodes with the same sign — an SFT step also helps the
  preference objective. The two losses share a common spectral basis.
- $\chi_{\mathrm{pos}} \approx 0$: the two gradients live in decoupled
  spectral bands — SFT neither helps nor hurts DPO.
- $\chi_{\mathrm{pos}} < 0$: the SFT gradient opposes the DPO gradient in the
  dominant spectral band — continued SFT actively hurts preference alignment.

**Predicted dynamics during SFT:**

Early in SFT, $\mathbf{u}_{\mathrm{CE}}$ lives in the bulk and
$\mathbf{u}_{\mathrm{DPO}}$ targets features the model hasn't developed yet.
The two gradients project onto different spectral bands, so their
eigenvalue-weighted overlap averages to $\approx 0$. As SFT progresses and
the model reaches the tail where preference-relevant features live, common
structure emerges: both gradients start to project onto the same modes, and
$\chi_{\mathrm{pos}}(\mathcal{L}_{\mathrm{DPO}}, \mathcal{L}_{\mathrm{CE}})$
becomes positive.

The zero-to-positive transition marks the point where SFT and DPO become
spectrally coupled — where the model has developed enough structure that
next-token prediction and preference optimization share a common basis. This
is a candidate diagnostic for **when to switch from SFT to DPO**: before the
transition, DPO has no spectral foothold; after it, the model is ready.

### Implementation note

Measurements A and B require vatis to accept a non-CE loss function for
$\chi_{\mathrm{loss}}$ (currently only the closed-form CE path is wired up).
The math is one extra `torch.autograd.grad(L, logits)` call plus a masked
squared sum — trivial, but it needs the custom-loss plumbing from the v1.2
work order.

## Practical perspective

### When to switch from SFT to DPO

**The claim:** we can tell you when a model is ready to switch from SFT to
DPO — not by waiting for the loss to plateau, but by measuring whether the
two objectives are directionally aligned.

**The problem.** A standard alignment pipeline runs SFT first, then DPO. The
transition point is chosen by heuristic — run SFT until the loss plateaus,
then switch. There is no principled diagnostic for whether the model has
developed the internal representations that preference optimization needs.

**The idea.** Take a held-out preference pair $(y_w, y_l)$ for a prompt $x$.
At each SFT checkpoint, ask: does an SFT gradient step also reduce the DPO
loss? The natural quantity is the gradient overlap:

$$\delta L(\mathcal{L}_{\mathrm{DPO}}, \mathcal{L}_{\mathrm{CE}}) = \langle \nabla_\theta \mathcal{L}_{\mathrm{DPO}},\; \nabla_\theta \mathcal{L}_{\mathrm{CE}} \rangle$$

Positive means SFT helps DPO. Zero means they're orthogonal. Negative means
SFT hurts DPO. But raw $\delta L$ is unusable as a training diagnostic —
applying the chain rule to each gradient shows why. The product decomposes
into:

- the **loss sensitivity** of each objective — how steep each loss landscape is
  in output space ($\nabla_f \mathcal{L}$, one per objective),
- the **model sensitivity** to weight perturbations — how much the outputs
  change when you nudge the parameters ($\nabla_\theta f$, one per input),
- and the directional alignment between the two.

The first four factors (two loss sensitivities, two model sensitivities) each
change by orders of magnitude during training, and since DPO and CE are
evaluated on different inputs, all four are independent. Any directional signal
is buried under their drift.

Identifying and dividing out all four scales yields:

$$\chi_{\mathrm{pos}}\!\big(\mathcal{L}_{\mathrm{DPO}},\; \mathcal{L}_{\mathrm{CE}}(y_w)\big) = \frac{\overbrace{(\nabla_f \mathcal{L}_{\mathrm{DPO}})^\top}^{\text{DPO loss direction}} \; \overbrace{\nabla_\theta f_{\mathrm{DPO}} \, (\nabla_\theta f_{\mathrm{CE}})^\top}^{\text{model coupling}} \; \overbrace{\nabla_f \mathcal{L}_{\mathrm{CE}}}^{\text{CE loss direction}}}{\underbrace{\lVert \nabla_f \mathcal{L}_{\mathrm{DPO}} \rVert_2 \;\lVert \nabla_\theta f_{\mathrm{DPO}} \rVert_F}_{\text{DPO scales}} \;\underbrace{\lVert \nabla_\theta f_{\mathrm{CE}} \rVert_F \;\lVert \nabla_f \mathcal{L}_{\mathrm{CE}} \rVert_2}_{\text{CE scales}}}$$

The numerator is $\delta L$ expanded via the chain rule. The denominator
normalizes each object by its magnitude. What's left — $\chi_{\mathrm{pos}}$
— is the pure directional alignment between the two objectives, independent
of how large any gradient or sensitivity happens to be.

**What the sign tells you:**

- **$\chi_{\mathrm{pos}} > 0$**: an SFT step also helps the DPO objective —
  the two losses share a common feature basis. The model is ready for DPO.
- **$\chi_{\mathrm{pos}} \approx 0$**: the two objectives are orthogonal —
  SFT is resolving features (e.g. grammar, common syntax) that have nothing to
  do with what distinguishes the preferred from the dispreferred response.
  Starting DPO here means the preference gradient has no foothold.
- **$\chi_{\mathrm{pos}} < 0$**: the two objectives are in conflict —
  continued SFT actively hurts preference alignment.

**Expected trajectory during SFT:**

```
chi_pos
  ^
  |          ┌─────────── model is ready for DPO
  |         /
  + - - - -/- - - - - - - - - - - -  0
  |       /
  |      /
  +─────·
  └──────────────────────────────────> SFT step
    early SFT:              late SFT:
    learning grammar,       resolving features
    basic token patterns    DPO cares about
```

Early in SFT, the model learns to produce grammatical text, predict common
token patterns, maintain coherence — features shared equally by the preferred
and dispreferred response. The DPO gradient targets the subtle features that
distinguish them (helpfulness, factual accuracy, safety). These are unrelated,
so $\chi_{\mathrm{pos}} \approx 0$.

As SFT progresses and the model develops richer internal representations, the
features being resolved begin to overlap with what preference optimization
cares about. $\chi_{\mathrm{pos}}$ becomes positive.

The **zero-to-positive transition** marks the earliest point at which DPO can
get a clean signal. Before it, the model lacks the representational
prerequisites. After it, further SFT has diminishing returns — the remaining
gains require the contrastive structure that only DPO provides.

**Cost:** a couple of backward passes on a held-out preference batch per
checkpoint — negligible compared to a training step. No changes to the
training loop; the measurement is purely diagnostic.

## Relation to the spectral\_tail experiment

The spectral\_tail experiment asked: can $\chi_{\mathrm{pos}}$ detect when a
model resolves shared semantic structure between Python and C++ implementations
of the same algorithms? The signal was ambiguous — a possible bump at 70m but
not reproducibly separable from Hutchinson noise.

This experiment is the same question in a setting where the "shared structure"
is operationally defined (the preference dataset tells us which features matter)
and the "spectral arrival" should be more pronounced (DPO explicitly targets
the distinguishing features, whereas pretraining resolves them incidentally).

If $\chi_{\mathrm{pos}}$ shows clear spectral arrival dynamics during DPO, it
retroactively validates the spectral\_tail approach and suggests the earlier
experiment was limited by model scale and probe design, not by the method
itself.

## Open questions

- Is the linearization (LNA) accurate enough in the DPO regime? DPO operates
  with small learning rates on a pretrained model, which is favorable for
  linearization. But the spectral tail has small eigenvalues, so higher-order
  terms may matter precisely where we're looking.
- Can we disentangle "DPO resolves the tail" from "DPO changes the bulk
  geometry enough that $\chi_{\mathrm{pos}}$ moves as a side effect"? The
  self-pair controls and the unrelated-pair controls should help here.
- What's the right granularity — per-prompt $\chi_{\mathrm{pos}}$, or
  aggregated across a batch of preference pairs? Per-prompt is noisier but
  more informative.
