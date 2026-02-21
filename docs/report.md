# Domain Generalisation on PACS

An English translation of the project report, with the numbers taken from the
notebooks' stored output rather than the original write-up where the two
disagree. Points where the original report's claims do not survive a check
against the code are marked as corrections, inline, after the paragraph they
affect.

## 1. The problem

A model trained on photographs and asked to classify sketches usually fails,
not because the objects changed but because the rendering did. *Domain
generalisation* is the setting where this is measured deliberately: the model
sees several **source** domains during training and is evaluated on a
**target** domain it has never seen, with no target images and no target
labels available at training time. This is stricter than domain adaptation,
which allows unlabelled target data.

**PACS** is the standard benchmark for it. Four domains — `photo`,
`art_painting`, `cartoon`, `sketch` — share seven classes: dog, elephant,
giraffe, guitar, horse, house, person. 9,991 images in total.

| domain | images | dog | elephant | giraffe | guitar | horse | house | person |
|---|---|---|---|---|---|---|---|---|
| `art_painting` | 2,048 | 379 | 255 | 285 | 184 | 201 | 295 | 449 |
| `cartoon` | 2,344 | 389 | 457 | 346 | 135 | 324 | 288 | 405 |
| `photo` | 1,670 | 189 | 202 | 182 | 186 | 199 | 280 | 432 |
| `sketch` | 3,929 | 772 | 740 | 753 | 608 | 816 | 80 | 160 |

Two features of this table matter later. The domains differ in size by more
than a factor of two, so pooling them without reweighting means sketch
contributes nearly twice as many gradients per epoch as art_painting. And the
class balance differs sharply between domains: sketch is 21% horses and 2%
houses, photo is 17% houses and 12% horses. A classifier that predicts each
domain's majority class scores 20.8% on sketch and 25.9% on photo, so 1/7 =
14.3% is not the right floor to compare against.

The protocol used throughout is *leave-one-domain-out*: train on three, test
on the fourth, repeat four times.

## 2. Baselines

Everything below uses `ResNet18` pretrained on ImageNet. The four experiments
in this section differ in how much of it is allowed to move.

### 2.1 No training at all

The pretrained backbone is kept and `fc` is replaced by a freshly initialised
7-way layer which is never trained. Accuracy per domain:

| domain | accuracy | correct / total |
|---|---|---|
| `art_painting` | 9.33% | 191 / 2048 |
| `cartoon` | 17.11% | 401 / 2344 |
| `photo` | 5.39% | 90 / 1670 |
| `sketch` | 17.92% | 704 / 3929 |

> **Correction.** The original report presents this as a transfer result. It
> is not: with an untrained head the logits are a random linear function of
> the features, so this is the chance level of the protocol. The spread
> between 5.39% and 17.92% is the interaction between each domain's class
> imbalance and whichever logit the random head happens to prefer — cartoon
> and sketch score higher because their majority classes are larger. Nothing
> about the backbone's quality is being measured.

### 2.2 Frozen backbone, linear probe

The backbone is frozen and only `fc` is trained, on one domain at a time,
SGD, four epochs.

| domain | accuracy |
|---|---|
| `art_painting` | 89.70% |
| `cartoon` | 89.89% |
| `photo` | 98.32% |
| `sketch` | 84.17% |

> **Correction.** These are **training** accuracies. The evaluation loader in
> that cell is built from `train_dataset` — the same images the head was just
> fitted on. The numbers are still informative in one narrow sense: a linear
> map on frozen ImageNet features can separate seven PACS classes within a
> single domain to 84–98%, so the features are linearly sufficient for the
> task. But they are not generalisation measurements and cannot be compared
> with anything else in this report.
>
> The original report gives 89.78 / 89.89 / 93.98 / 84.17. Cartoon and sketch
> match the notebook exactly; art differs by 0.08 and photo by 4.34. Since a
> re-run would perturb all four, this looks like transcription.

### 2.3 Fine-tune on one domain, evaluate on the other three

The whole network is trained on a single domain — SGD, lr 1e-3, batch 32,
four epochs, cross-entropy — and then evaluated on each of the other three.
Rows are the training domain:

| train ↓ / test → | `art_painting` | `cartoon` | `photo` | `sketch` |
|---|---|---|---|---|
| **`art_painting`** | — | 52.77 | 95.69 | 44.03 |
| **`cartoon`** | 60.74 | — | 83.23 | 60.88 |
| **`photo`** | 59.57 | 22.70 | — | 26.65 |
| **`sketch`** | 22.27 | 34.00 | 21.98 | — |

Read as a matrix of style gaps, this is asymmetric in a specific and
interpretable way. Training on `art_painting` gives 95.69% on `photo`;
training on `photo` gives 59.57% back on `art_painting`. Training on
`cartoon` gives 83.23% on `photo`; the reverse gives 22.70%. The stylised
domains transfer to photographs far better than photographs transfer to
stylised domains.

The backbone explains it. It was pretrained on ImageNet — photographs — so
photo-like features are already present before any fine-tuning. Fine-tuning
on a stylised domain teaches shape and part structure while leaving the
photographic features intact, and photo is then classified using both.
Fine-tuning on photo teaches nothing new, and the model has no reason to
develop the abstraction a sketch requires. `sketch` is the hardest target in
three of four rows and the worst source in all three of its columns: line
drawings share the least surface statistics with everything else.

> **Correction.** The original report filled this table's diagonal with the
> §2.2 linear-probe numbers. That mixes two training regimes — frozen
> backbone versus full fine-tune — and inserts a training-set accuracy into
> a table of held-out accuracies, which makes the diagonal look like a
> ceiling the off-diagonal is failing to reach. The diagonal is left blank
> here; a model is never evaluated on the domain it was trained on.
>
> Six of the twelve off-diagonal entries also differ from the notebook's
> output, by large margins: 80.98 vs 52.77, 79.04 vs 60.74, 82.70 vs 22.70,
> 64.04 vs 34.00, 78.12 vs 59.57, 62.78 vs 22.27, and 79.24 vs 95.69. The
> other five match to two decimals, which again rules out a re-run. The
> notebook's values are used above.

### 2.4 Leave-one-domain-out

Three source domains, the fourth held out entirely.

| held-out domain | accuracy |
|---|---|
| `art_painting` | 32.08% |
| `cartoon` | 52.47% |
| `photo` | 58.80% |
| `sketch` | 57.16% |

> **Correction.** These are lower than the single-source results in §2.3 —
> `art_painting` gets 32.08% from three source domains but 60.74% from
> `cartoon` alone — which should not happen if the extra data were being
> used. The cause is the training loop: for each epoch it runs one complete
> pass over the first source domain, then one over the second, then one over
> the third. Gradients are never mixed across domains within a batch, so the
> model is pulled toward each domain in turn and every epoch ends biased
> toward whichever loader came last. That is sequential fine-tuning with
> catastrophic forgetting, not empirical risk minimisation over the pooled
> source distribution.
>
> Section 5 runs the standard protocol — one batch from every source domain
> per step, concatenated into a single gradient — and reaches 58.64% on
> `art_painting` with plain SGD, 26 points above this table.
>
> A second, smaller bug: the per-epoch loss is the sum over all three
> domains divided by `len(dataloader)`, the length of whichever loader the
> loop variable was left pointing at. The printed losses are off by a factor
> of roughly three, varying with domain size.

### 2.5 With augmentation

Colour jitter (brightness, contrast, saturation 0.4, hue 0.1), Gaussian blur
(kernel 5, σ ∈ [0.1, 2.0]) and additive Gaussian noise (σ = 0.01), applied to
the source domains only. Same loop as §2.4.

| held-out domain | no augmentation | with augmentation | Δ |
|---|---|---|---|
| `art_painting` | 32.08% | 45.02% | +12.94 |
| `cartoon` | 52.47% | 66.13% | +13.66 |
| `photo` | 58.80% | 75.09% | +16.29 |
| `sketch` | 57.16% | 57.67% | +0.51 |

Augmentation is worth 13–16 points on three of four targets and nothing on
the fourth. The asymmetry is informative rather than noise: these
transformations perturb colour and texture statistics, which is exactly what
separates paintings, cartoons and photographs from each other. Sketches are
line drawings — near-binary, no colour to jitter, and blurring a line drawing
does not produce anything closer to a different line drawing. The one target
that needs the most help is the one this augmentation cannot help. Closing
the sketch gap needs augmentations that attack shape and stroke — edge
extraction, posterisation, RandAugment-style geometric distortion — not
photometric ones.

Repeating §2.3's pairwise matrix with the augmented source transform confirms
the mechanism:

| train ↓ / test → | `art_painting` | `cartoon` | `photo` | `sketch` |
|---|---|---|---|---|
| **`art_painting`** | — | 54.10 | 95.75 | 45.20 |
| **`cartoon`** | 58.35 | — | 81.86 | 54.44 |
| **`photo`** | 62.40 | 23.81 | — | 29.93 |
| **`sketch`** | 31.49 | 38.01 | 47.07 | — |

The largest gains are in the `sketch` row — 22.27 → 31.49, 21.98 → 47.07 —
where the model had almost nothing to work with. Where a pair already
transferred well (`art_painting` → `photo`, 95.69 → 95.75) augmentation
changes nothing, and in two cells it costs a couple of points.

## 3. Domain-adversarial training (DANN)

Augmentation attacks the problem from outside the model: perturb the input
distribution and hope the features become invariant as a side effect. DANN
attacks it directly — make domain identity unrecoverable from the features by
construction.

A shared `ResNet18` feature extractor **Gf** produces 512-d features that
feed two heads: a 7-way label predictor **Gy**, trained on source labels
only, and a binary domain classifier **Gd** trained to tell source from
target. Between Gf and Gd sits a **gradient reversal layer**: the identity on
the forward pass, multiplication by `-λ` on the backward pass.

```
                        +--> Gy --> class (7)      CE, source only
Gf (ResNet18, 512-d) ---+
                        +--> GRL(-λ) --> Gd --> domain (2)    CE, source + target
```

`L = L_y(source) + L_d(source + target)`. Because of the sign flip, the same
descent step that improves Gd's ability to discriminate domains *degrades*
Gf's ability to supply the information Gd needs. At the fixed point the
features carry no domain signal but still carry class signal — because Gy's
gradient, which is not reversed, keeps pulling in that direction. Target
labels are used only for the final evaluation.

Target `cartoon`, sources `photo`/`sketch`/`art_painting`. Adam, lr 1e-4, 32
images per domain stream, four epochs. The source and target streams are
drawn in parallel and concatenated, so each step sees 32 source and 32 target
images; Gd trains on all 64 and Gy on the source half.

### 3.1 Result

Target accuracy rises from 17.11% at initialisation to **71.67%** after four
epochs. In the t-SNE embedding of the 512-d features, the distance between
the source and target centroids falls from 6.2817 to 4.5311, and class
structure that is not visible before training is clearly present after it.

> **Correction — the comparison baseline.** The original report compares
> 71.67% against 32.08% (no augmentation) and 45.02% (with augmentation),
> concluding a gain of roughly 40 points. Those two numbers are
> `art_painting`'s leave-one-out results, not `cartoon`'s. Cartoon's are
> 52.47% and 66.13%. Against the augmented baseline the gain is **5.5
> points**, and against the unaugmented one 19 — still a gain, an order of
> magnitude smaller than claimed.

> **Correction — λ = 6e-5.** At that value the reversed gradient arriving at
> the feature extractor is about four orders of magnitude smaller than the
> classification gradient. Gd still learns, since its own gradients are
> unscaled, but it exerts almost no pressure on the features: the mechanism
> the section is about is effectively switched off. Most of the improvement
> over the §2.5 baseline is ordinary fine-tuning on three domains plus
> augmentation, and the centroid movement is what fine-tuning does to any
> embedding, not evidence of adversarial alignment. The published DANN
> schedule ramps λ from 0 toward 1 over training — starting near zero so the
> domain head is worth listening to before it is fought — and that is the
> first thing to change.

> **Correction — what the 71.67% is measured on.** The augmented transform
> was applied to the target loader as well as the source loaders, so the
> headline number is accuracy on colour-jittered, blurred, noised cartoons,
> not on cartoons. The notebook now reports both.

> **Correction — training accuracy.** The per-step training accuracy was
> averaged over the full 64-image batch, including the 32 target images,
> using their labels. Under the domain-generalisation protocol those labels
> do not exist. It is a monitoring statistic rather than a reported result,
> but it is computed from data the method is not allowed to see. Now
> source-only.

> **On the centroid distance.** 6.2817 → 4.5311 is computed in the t-SNE
> embedding, where distances are not metric and depend on perplexity, random
> initialisation and the number of points. The number is meaningful only as a
> before/after comparison within one embedding configuration; it should not
> be read as "the domains are 28% closer".

## 4. Optimisers and flatness

The §2.4 protocol bug means none of the leave-one-out numbers so far measure
what they were meant to. This section restarts from the standard protocol and
varies the optimiser.

Each step draws one batch of 64 from **every** source domain and concatenates
them: a single gradient over 192 images spanning all three styles. Each source
domain is split 80/20 into train and validation; the held-out domain is
touched once, at the end. 5,000 iterations, `ResNet18`, lr 1e-4. The
evaluation pass over all six source loaders runs every 100 iterations.

Against that fixed backbone and schedule, five things vary:

- **SGD** — `optim.SGD`, no momentum.
- **SGD + frozen BatchNorm** — every `BatchNorm2d` held in eval mode with its
  affine parameters frozen.
- **Adam** — adaptive per-parameter step sizes.
- **FAD** — flatness-aware descent.
- **MIRO** — Adam plus a KL penalty toward the frozen pretrained model's
  features.

### 4.1 Freezing BatchNorm

A BatchNorm layer normalises by statistics estimated from the data it sees.
When that data is a mixture of art, photo and cartoon, the statistics
describe the mixture — and the mixture is exactly what will not be present at
test time. Freezing the layers keeps ImageNet's statistics, which are at
least a fixed, known reference rather than one fitted to the wrong
distribution.

The one implementation detail that matters: `model.train()` walks the whole
module tree and puts every BatchNorm layer back into training mode, so the
freeze has to be re-applied *after* every such call — including the one
inside the periodic evaluation block, which switches to `eval()` and back.

### 4.2 FAD

The premise behind flatness methods is that generalisation under
distribution shift tracks the flatness of the minimum. A sharp minimum is
fitted to the source distribution's particular curvature and falls apart when
the distribution moves; a flat one degrades gently.

Two things get called flatness:

- **Zeroth-order flatness** `R⁰(θ)` — the largest *loss increase* anywhere in
  a ρ-ball around θ. SAM minimises it, by taking the gradient at an
  adversarially perturbed point instead of at θ.
- **First-order flatness** `R¹(θ)` — the largest *gradient norm* in that
  ball. GAM minimises it. This catches sharp directions that a zeroth-order
  measurement misses because the loss happens to be low at the one point it
  sampled.

FAD minimises both: `R^{ρ,α}(θ) = α·R⁰(θ) + (1−α)·R¹(θ)`. Neither term has a
closed-form gradient, but both can be approximated by finite differences of
gradients — no Hessian-vector products:

```
g0 = ∇L(θ)                                        plain gradient
g1 = ∇L(θ + ρ·g0/‖g0‖)          h0 = g1 - g0        ≈ ∇R⁰
g2 = ∇L(θ + ρ·h0/‖h0‖)
g3 = ∇L(θ + ρ·g2/‖g2‖)          h1 = g3 - g2        ≈ ∇R¹

θ ← θ - lr·( g0 + β(α·h0 + (1-α)·h1) )
```

Four backward passes per step. ρ = 0.05, α = 0.5, β = 1.0, ξ = 1e-6 (added to
every norm before dividing). The update is applied by hand rather than
through an optimiser, so there is no momentum or adaptive state — this is
plain SGD on a modified gradient, which is what makes the SGD row the right
comparison for it.

### 4.3 MIRO

Fine-tuning on three domains moves the model away from the general-purpose
features it started with, and some of what it loses is exactly the
domain-general structure the target domain would need. MIRO adds a term that
resists the drift: the pretrained model is treated as an oracle for what
domain-general features look like, and the fine-tuned model is penalised for
departing from it.

Concretely, forward hooks on `layer3` and `layer4` collect the intermediate
feature maps of both the fine-tuned model and a frozen copy of the pretrained
one on the same batch. Each map is reduced to a per-channel mean and variance
over the spatial dimensions, treated as a diagonal Gaussian, and the KL
divergence between the two is added to the loss with weight
`miro_lambda` = 0.1, averaged over the two layers.

This is a simplified MIRO. The published method inserts a learnable
projection between the two feature spaces before comparing them, and weights
each layer by an estimate of its reliability. Here the raw features are
compared directly.

### 4.4 Results

Target accuracy:

| optimiser | → `sketch` | → `art_painting` |
|---|---|---|
| SGD | 54.01% | 58.64% |
| SGD + frozen BN | 50.98% | 62.35% |
| Adam | 60.96% | 78.76% (frozen BN) |
| FAD + frozen BN | 53.75% | 67.24% |
| MIRO (Adam) + frozen BN | — | 74.32% |

Final test loss, same runs: SGD 1.4347 / 1.2411; SGD+BN 1.3386 / 1.2588;
Adam 1.5434 / 2.0100; FAD 1.2137 / 0.9635; MIRO — / 0.8053.

**Adam wins on both targets**, by 16.4 points over SGD with the same
BatchNorm treatment on `art_painting` and 7.0 on `sketch`. Its per-source
validation curves show textbook overfitting: training accuracy above 95%,
validation loss climbing while validation accuracy stays flat.

> **Correction.** The original report reads those curves as a reason to
> prefer FAD over Adam. They are not. The validation splits are held out from
> the **source** domains, so a rising validation loss there says the model is
> becoming overconfident on source-like data — it says nothing directly about
> transfer to the unseen domain. Under this protocol, accuracy on the
> held-out domain *is* the generalisation measurement, and Adam's is the best
> of the five. A method that overfits the source domains and still transfers
> best has not been refuted by the overfitting.

**FAD beats SGD consistently in sign** — +4.9 on `art_painting`, +2.8 on
`sketch` — at four gradient evaluations per step instead of one. It is the
right comparison, since both are plain descent without optimiser state, and
the improvement is in the direction the theory predicts. But it is one seed
per cell, and FAD does not reach Adam on either target: 67.24% against
78.76%, 53.75% against 60.96%.

**Freezing BatchNorm helps one target and hurts the other** — +3.7 on
`art_painting`, −3.0 on `sketch`. With a single seed per cell, this is
consistent with the freeze doing nothing and the difference being noise.

**MIRO lands between FAD and Adam** at 74.32%, ahead of SGD and FAD, 4.4
points behind Adam. The interesting detail is the loss: MIRO has the lowest
test loss of all nine runs (0.8053) while Adam has the highest (2.0100)
despite being more accurate. The KL penalty is buying calibration — MIRO's
confidence tracks its correctness — while Adam produces a model that is
confidently wrong when it is wrong and right more often anyway. Which of
those is preferable depends on whether the downstream use needs a label or a
probability.

Four plausible reasons MIRO does not beat Adam here:

1. **Backbone capacity.** The published results use ResNet50. A
   regularisation term that constrains where a large model can go has less to
   work with on ResNet18, whose fine-tuned features are already close to the
   pretrained ones.
2. **`miro_lambda` = 0.1 may be too small**, in which case the penalty is a
   rounding error on the classification loss and MIRO is Adam with overhead.
   The 4.4-point deficit against plain Adam argues against this being the
   whole story, though — if the term were inert the two would match.
3. **Unstable feature statistics.** Channel means and variances of `layer3`
   and `layer4` move substantially early in training, so the KL target is
   noisy exactly when the penalty has the most influence.
4. **No learnable projection.** This is the most likely one. Comparing raw
   feature statistics gives the penalty no freedom to align two spaces that
   differ by a fixed transform: if the fine-tuned features are a rotation of
   the pretrained ones, the KL is large even though no information has been
   lost, and minimising it fights a change that was harmless. The published
   projection exists precisely to absorb that.

### 4.5 What would make this table mean something

Nine runs, two target domains, one seed each. The `sketch` and `art_painting`
columns disagree about the sign of the BatchNorm effect, which is the clearest
possible signal that single-seed differences of a few points are not
separable from seed noise. What is missing:

- **Multiple seeds per cell**, with the spread reported rather than the point
  estimate. Three seeds would already distinguish FAD's +4.9 from nothing.
- **All four target domains.** `photo` and `cartoon` are the two easiest
  targets and neither appears here, so the comparison is drawn only on the
  hard half of the benchmark.
- **Model selection on the source validation splits.** The reported numbers
  are whatever iteration 5,000 produced. The splits exist and are already
  being evaluated every 100 steps; using them to pick a checkpoint is free
  and is what the DomainBed protocol requires.
- **A matched compute budget.** FAD takes four backward passes per step, so
  at equal wall-clock it gets a quarter of the iterations. The comparison
  above gives every method 5,000 steps, which flatters FAD on time and
  penalises it on nothing.

## 5. Summary

- The pairwise transfer matrix is strongly asymmetric, and the asymmetry
  follows from the ImageNet backbone: stylised domains transfer to
  photographs much better than the reverse.
- Photometric augmentation is worth 13–16 points on three targets and 0.5 on
  `sketch`, because it perturbs colour and texture, which is not what makes a
  line drawing hard.
- DANN reaches 71.67% on `cartoon` against a 66.13% augmented baseline — a
  real but modest gain, and with λ = 6e-5 the adversarial mechanism is not
  what produced it.
- Under the correct leave-one-domain-out protocol, plain Adam is the strongest
  of the five optimisers on both targets tested, including against both
  flatness-aware and feature-regularised alternatives.
- With one seed per cell, differences of a few points between the five
  optimisers should not be interpreted.

## References

- Li et al., *Deeper, Broader and Artier Domain Generalization*, ICCV 2017 —
  PACS.
- Ganin & Lempitsky, *Unsupervised Domain Adaptation by Backpropagation*,
  ICML 2015 — DANN and the gradient reversal layer.
- Foret et al., *Sharpness-Aware Minimization for Efficiently Improving
  Generalization*, ICLR 2021 — SAM, zeroth-order flatness.
- Zhang et al., *Gradient Norm Aware Minimization*, CVPR 2023 — GAM,
  first-order flatness.
- Cha et al., *Domain Generalization by Mutual-Information Regularization
  with Pre-trained Models*, ECCV 2022 — MIRO.
- Gulrajani & Lopez-Paz, *In Search of Lost Domain Generalization*, ICLR 2021
  — DomainBed, and the case for multiple seeds and explicit model selection.
