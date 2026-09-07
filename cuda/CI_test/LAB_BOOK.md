# Lab book

Findings as they land. The "decisions a reviewer will challenge" section is the one to
keep current — every entry there is a question that will be asked.

## Environment of record

- date: 2026-09-07
- python 3.12.13, numpy 2.4.6, scipy 1.15.3, numba 0.67.0, ml_dtypes 0.6.0, pyarrow 25.0.1
- BLAS: scipy-openblas 0.3.31.188.0, all pools pinned to 1 thread (verified via threadpoolctl)
- OS: Windows 10, `np.float128` unavailable — mpmath is the high-precision reference
- local GPU: GeForce MX230 (Pascal, cc 6.1). No TF32, no BF16 tensor cores, FP64 at 1/32.
  Usable for CUDA correctness work and for E0.2's transcendental comparison; useless for Tier 5.

## E0.1 — the noise floor  ✅ PASSES

- [x] bitwise-identical `stat` vectors, FP64 vs FP64, **across two processes**
- [x] flip rate exactly 0.0 at α ∈ {0.001, 0.01, 0.05}, 3 seeds, 2340 tests
- [x] **E0.1b anti-control detects an injected 1-ULP perturbation** — the gate has power

Gate G0 is green and mechanised: `runner` refuses any experiment without a passing record
keyed on (code_digest, env_digest, hostname).

## E0.3 — implementation variance at FP64

**Numba's `np.linalg.inv` and NumPy's disagree bitwise on 58% of 4×4 correlation
matrices**, single-threaded, same machine. 30% of p-values differ. But:

| κ | max abs Δp | decision flips at α=0.01 |
|---|---|---|
| 1e2 | 4.8e-15 | 0 / 4000 |
| 1e4 | 3.3e-14 | 0 / 4000 |
| 1e6 | 6.0e-14 | 0 / 4000 |
| 1e8 | 2.7e-13 | 0 / 4000 |

Ubiquitous at ULP level, invisible at the decision level across the whole κ range tested.
**Consequence for the design:** the implementation is part of the cell key and must never
be mixed within a comparison — but it does not by itself threaten E0.1.

Still to do: the three parcorr routes and log1p-vs-naive z as a full factorial, run
through the *same* `metrics.flip_counts` as the format comparison so the two spreads are
directly comparable. That comparability is the whole point of E0.3.

## E1.1 / E1.3 — flip rate by format and level

Synthetic controlled-κ matrices, 8 variables, n = 200, 5 seeds, levels 0–2.
Requested κ = 1e8 (realised ≈ 5.5e7). Flip rate vs FP64, α = 0.01:

| format | level 0 | level 1 | level 2 |
|---|---|---|---|
| float32 | 0.00% | 0.00% | 0.00% |
| tf32 | 0.00% | 0.12% | 1.51% |
| float16 | 0.00% | 0.12% | 1.51% |
| bfloat16 | 0.00% | — | 7.44% |
| float8_e4m3 | 0.00% | 6.61% | 21.57% |
| float8_e5m2 | 0.00% | 7.80% | 17.62% |

Three things to note.

1. **TF32 and FP16 agree to every decimal place, at every level and α.** They share a
   10-bit mantissa and the exponent range never binds on [−1, 1]. This is the built-in
   correctness check, and it passes.
2. **Flip rate rises steeply with level** — the submatrix grows, conditioning worsens.
   This is E1.3's stratification and it is the mechanism behind C3.
3. **FP32 is clean everywhere tested**, including at requested κ = 1e12. See below.

## E1.2 — the κ that matters is the submatrix's, not the design's

This is the most useful finding so far and it was not anticipated.

With the full 8×8 correlation matrix at κ ≈ 5.5e11, the log₁₀ κ of the submatrices
**actually inverted**:

| level | distribution of log₁₀ κ_proxy |
|---|---|
| 0 | 0:59, 1:54, 2:21, 3:6 |
| 1 | 1:141, 2:339, 3:273, 4:63, 5:2, 6:4 |
| 2 | 2:109, 3:816, 4:892, 5:208, 6:59, 7:15 |

**The parent matrix's conditioning does not transfer to the submatrices.** They top out
around 1e7 even when the design is at 1e12. Since FP32's flip onset sits near κ ≈ 1e8
(measured separately at m=4: 0.20% at κ=1e8, 0.57% at m=6, 1.27% at m=8/κ=1e10), FP32
never flips in this setup — not because FP32 is exact, but because the instrument never
reaches the regime where it fails.

**Consequences.** (a) The synthetic controlled-κ generator is a weak instrument for FP32
and must not be used to conclude "FP32 is safe". (b) Reaching high submatrix κ needs
either larger `|S|` or genuinely near-collinear variable subsets — which is exactly the
lagged-variable case, so this *strengthens* the C3 motivation and moves `datasets/var.py`
earlier. (c) Recording per-test κ rather than per-dataset κ was the right call.


## E3.1 — the time-series hazard (C3)  ✅ MECHANISM CONFIRMED

AR(1), 3 series, τmax = 4 (12 variables), n_obs = 4000, 5 seeds, levels 0–2 pooled.
Flip rate (%) vs FP64 at α = 0.01:

| format | φ=0 | φ=0.6 | φ=0.9 | φ=0.95 | φ=0.99 |
|---|---|---|---|---|---|
| float32 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| tf32 | 0.000 | 0.000 | 0.011 | 0.005 | 0.276 |
| float16 | 0.000 | 0.000 | 0.011 | 0.005 | 0.276 |
| bfloat16 | 0.000 | 0.027 | 0.076 | 0.606 | 2.159 |
| float8_e4m3 | 0.000 | 0.530 | 4.237 | 3.983 | 45.323 |
| float8_e5m2 | 0.000 | 1.905 | 5.249 | 18.164 | 58.261 |

**C3 holds for every format below FP32.** Flip rate rises monotonically with φ, and the
rise is steep: E5M2 goes 0 → 58%. TF32 and FP16 remain identical at every φ (the built-in
check still passing). The φ=0.95 column dips slightly below φ=0.9 for tf32 and e4m3 —
small counts, within seed noise; needs more seeds before being read as anything.

**FP32 is exactly 0.00% at every φ**, exactly as the calibration predicted. Submatrix
κ tops out at 10² even at φ=0.99, three orders below FP32's ~1e8 onset. This is a
property of the instrument, not of FP32: an AR(1) cannot reach that regime.

### The FP8 saturation prediction — confirmed, with a sharp threshold

Pre-registered: E4M3 rounds any |r| ≥ 0.96875 to exactly 1.0, and at φ=0.99 the lag-1/2/3
correlations are 0.99 / 0.98 / 0.97 — all above it. Status breakdown for E4M3, 18480 tests:

| φ | OK | ρ=1 | \|ρ\|>1 | pinv | undefined |
|---|---|---|---|---|---|
| 0.90 | 18474 | 0 | 0 | 6 | 0 |
| 0.95 | 18480 | 0 | 0 | 0 | 0 |
| 0.99 | **3551** | 770 | 3579 | 6605 | **7554** |

Clean at φ=0.95 (lag-1 = 0.95, below threshold), catastrophic at φ=0.99 (lag-1 = 0.99,
above it). The transition is sharp and mechanistically explained.

Two things follow. **(a)** In the flagship time-series regime FP8's behaviour is
*degeneracy*, not rounding — it must be reported through the status breakdown, and the
45% figure above is computed on the comparable subset with 7554 undefined excluded.
Without the status column those would have been counted as "not rejected = no flip",
understating the effect enormously. This is failure mode A caught in the wild.
**(b)** 6605 of 18480 tests (36%) take the pseudo-inverse branch — a *code-path* flip
distinct from numeric perturbation. That is E2.4's measurement, already recorded.

### What this does not yet answer

E3.2 (does autocorrelation push work into levels ≥ 2?) needs **cascade mode**, which is
not built. Under the current α = 1.0 / complete-graph device the per-level test counts are
fixed by construction, so the level distribution cannot shift. Gate G3 stays open.


## E2.1 — amplification (contribution C1)  ⚠️ NEGATIVE RESULT, gate G2

Cascade mode. Random sparse DAG (N=25, edge_prob=0.15), **population** correlations so
FP64 carries no sampling error, n=1000, α=0.01, max_level=3, PC-stable, **30 seeds**.

Numerator from the free-running cascade; denominator from replaying the **union** of every
format's trajectory (the only test set that privileges no format's path).

| format | flip rate | union flips | edges changed | **amplification** | sepset differ | traj Jaccard |
|---|---|---|---|---|---|---|
| float32 | 0.0000 | 0 | 0 | undefined | 0.00% | 1.000 |
| tf32 | 0.0000 | 4 | 0 | 0.00 | 0.04% | 0.999 |
| float16 | 0.0000 | 4 | 0 | 0.00 | 0.04% | 0.999 |
| bfloat16 | 0.0004 | 42 | 4 | 0.095 | 0.34% | 0.992 |
| float8_e4m3 | 0.0076 | 714 | 43 | 0.060 | 4.10% | 0.903 |
| float8_e5m2 | 0.0202 | 1811 | 131 | 0.072 | 7.65% | 0.783 |

**Amplification is ≈ 0.06–0.10, not > 1.** Roughly fourteen flipped CI decisions are
needed to change one edge. The path dependence **damps** per-test flips rather than
amplifying them.

The mechanism is straightforward in hindsight: PC removes an edge if *any* separating set
accepts. A flip that changes one candidate's decision usually leaves the edge outcome
unchanged, because another candidate still accepts (or none did). The algorithm is robust
by redundancy, and that redundancy is exactly what absorbs the flips.

### Denominator sensitivity — a reviewer will ask

Amplification depends on which test set supplies the denominator (10 seeds, N=25):

| format | edges | union | intersection | FP64-replay | amp(union) | amp(inter) | amp(replay) |
|---|---|---|---|---|---|---|---|
| float8_e4m3 | 9 | 156 | 96 | 116 | 0.058 | 0.094 | 0.078 |
| float8_e5m2 | 46 | 554 | 237 | 299 | 0.083 | 0.194 | 0.154 |

A 2.3× spread — worth reporting explicitly — but **every choice lands well below 1**, so
the qualitative conclusion does not depend on the definition.

### What this does to the paper

Gate **G2** as written says amplification ≈ 1 collapses C1 and "the paper becomes a
routine precision table". The measurement is *below* 1, which is a stronger statement than
G2 anticipated and, importantly, is a **result rather than a failure**:

> Constraint-based causal discovery absorbs per-test precision flips: a 2% per-test flip
> rate at FP8-E5M2 yields a skeleton differing in ~0.07 edges per flip. The mechanism is
> the redundancy of the separating-set search.

That licenses reduced precision far more strongly than a precision table would, and it is
the pre-registered negative outcome §6 asks for rather than something to quietly abandon.

**But the sepset channel tells a different story, and it is the better headline.** At
E5M2, 7.65% of separating sets differ while the skeleton barely moves. Separating sets
determine v-structures and therefore orientation, so the CPDAG can change even when the
skeleton is identical. **The skeleton is robust; the orientation may not be.** E2.3 is
currently unmeasured by anyone and now looks like the strongest result available.

Caveat before generalising: this is population correlations on a sparse, well-conditioned
DAG, where FP64 is essentially exact and few tests sit near threshold. Whether
amplification stays below 1 in the ill-conditioned time-series regime is being measured
separately.


## E2.1 in the ill-conditioned regime — the ratio is the wrong lens

Cascade on VAR AR(1), φ=0.99, 4 series, τmax=4 (16 variables), n=4000, 20 seeds:

| format | flip rate | union flips | edges changed | amplification | sepset differ | traj Jaccard | mean edges (ref → var) |
|---|---|---|---|---|---|---|---|
| float32 | 0.0000 | 0 | 0 | undefined | 0.00% | 1.000 | 12.15 → 12.15 |
| tf32 | 0.0031 | 1087 | 7 | 0.006 | 0.42% | 0.994 | 12.15 → 12.40 |
| bfloat16 | 0.0372 | 13076 | 81 | 0.006 | 1.73% | 0.944 | 12.15 → 16.00 |
| float8_e4m3 | 0.5602 | 133743 | 501 | 0.004 | 66.8% | 0.124 | 12.15 → 30.90 |
| float8_e5m2 | 0.6565 | 188680 | 852 | 0.005 | 71.6% | 0.049 | 12.15 → **50.85** |

Amplification is *lower* here (0.004–0.006), so the sub-1 finding is robust across regimes.
But the ratio is now actively misleading, and this is the key interpretive point:

**At a 66% flip rate the algorithm is not amplifying a small perturbation through path
dependence — it is operating on essentially random input.** The ratio is small because the
denominator exploded, not because the damage is small. The damage is enormous: the
recovered skeleton goes from 12 edges to 51, a 4× inflation, and the trajectories are
almost disjoint (Jaccard 0.05).

So C1's framing — "a small per-test flip rate is amplified into a much larger graph
divergence" — is not supported in either regime:

- where the flip rate is small (FP32, TF32), the graph barely moves, so there is nothing
  to amplify;
- where the graph moves a lot (FP8 on autocorrelated data), the flip rate is enormous too,
  so it is breakdown rather than amplification.

## E4.2 — directional asymmetry: false POSITIVES dominate

| regime | format | edges lost | edges gained | mean edges ref → var |
|---|---|---|---|---|
| DAG, well-cond. | float8_e5m2 | 41 | 90 | 37.30 → 38.93 |
| VAR φ=0.99 | bfloat16 | 2 | 79 | 12.15 → 16.00 |
| VAR φ=0.99 | float8_e4m3 | 63 | 438 | 12.15 → 30.90 |
| VAR φ=0.99 | float8_e5m2 | 39 | **813** | 12.15 → **50.85** |

**21:1 in favour of gained edges at E5M2.** The proposal notes that the fabricated
secondary source asserts differences are dominated by *false negatives* (missed edges);
the measurement says the opposite, decisively. Useful for §1.2's positioning, since it is
a second concrete point on which that source is checkably wrong.

Mechanism: a degenerate test cannot remove an edge, so degeneracy retains edges and the
graph inflates. **This direction is conditional on a design decision and must be reported
as such** — see reviewer-challenge item 9.


## E2.3 — the CPDAG-level number  (no amplification found; see the correction below)

Orientation implemented: v-structures from separating sets, then Meek R1–R3 to a fixpoint.
**Cross-validated against causal-learn's `dag2cpdag` on 60 random DAGs: 60/60 identical.**

`orientation_differ` is defined only on edges BOTH graphs recovered, so it isolates the
arrow channel from skeleton differences entirely. Note the percentages below have DIFFERENT
denominators (sepset% over removed pairs, orientation% over retained edges), so they must
not be divided by one another -- see the correction below.

### Well-conditioned DAG (population correlations, N=25, 30 seeds)

| format | skeleton% | sepset% | orientation% | CPDAG SHD | v-str ref → var | conflicts ref → var |
|---|---|---|---|---|---|---|
| float32 | 0.00 | 0.00 | 0.00 | 0 | 1435 → 1435 | 570 → 570 |
| tf32 | 0.00 | 0.04 | 0.00 | 3 | 1435 → 1437 | 570 → 573 |
| float16 | 0.00 | 0.04 | 0.00 | 3 | 1435 → 1437 | 570 → 573 |
| bfloat16 | 0.36 | 0.34 | 0.73 | 13 | 1435 → 1425 | 570 → 561 |
| float8_e4m3 | 3.84 | 4.10 | 6.77 | 122 | 1435 → 1471 | 570 → 585 |
| float8_e5m2 | 11.71 | 7.65 | 10.23 | 197 | 1435 → 1610 | 570 → 630 |

### Ill-conditioned VAR, φ=0.99 (16 variables, 20 seeds)

| format | skeleton% | sepset% | orientation% | CPDAG SHD | v-str ref → var | conflicts ref → var |
|---|---|---|---|---|---|---|
| float32 | 0.00 | 0.00 | 0.00 | 0 | 4 → 4 | 2 → 2 |
| tf32 | 2.88 | 0.42 | 3.32 | 14 | 4 → 5 | 2 → 1 |
| bfloat16 | 33.3 | 1.73 | 37.76 | 155 | 4 → 57 | 2 → 15 |
| float8_e4m3 | 206 | 66.84 | 53.69 | 455 | 4 → 706 | 2 → 186 |
| float8_e5m2 | 351 | 71.58 | 24.16 | 663 | 4 → 1148 | 2 → 282 |

### What this does to C1 — corrected

An earlier draft of this section claimed orientation amplifies sepset differences by up to
21.8x. **That was wrong and is retracted.** It divided orientation% by sepset%, but those
two percentages have different denominators over different populations: sepset% is over
pairs the search REMOVED, orientation% is over edges it RETAINED. The retained set is much
smaller, so a percentage on that base is mechanically larger, and the ratio was largely
measuring denominator sizes.

Measured on a count basis -- same kind of object on top and bottom -- the picture is:

| regime | format | edges/flip | orient/flip | orient / sepset-diff |
|---|---|---|---|---|
| DAG | bfloat16 | 0.095 | 0.095 | 0.15 |
| DAG | float8_e4m3 | 0.060 | 0.048 | 0.11 |
| DAG | float8_e5m2 | 0.072 | 0.027 | 0.08 |
| VAR φ=0.99 | tf32 | 0.006 | 0.007 | 0.89 |
| VAR φ=0.99 | bfloat16 | 0.006 | 0.007 | **2.53** |
| VAR φ=0.99 | float8_e4m3 | 0.004 | 0.001 | 0.07 |

**There is no large amplification in any channel.** Every count-based ratio is O(1) or
below; the largest is 2.53 orientation changes per differing separating set (BF16 on VAR),
which is what one collider orienting two edges plus a little Meek propagation would
produce anyway. **C1's mechanism claim is not supported.**

Nor is there a dissociation between the channels. Comparing like with like:

| regime / format | skeleton% | orientation% |
|---|---|---|
| VAR, tf32 | 2.9 | 3.3 |
| VAR, bfloat16 | 33.3 | 37.8 |
| DAG, float8_e5m2 | 11.7 | 10.2 |

Skeleton and orientation degrade at **roughly the same proportional rate**. An earlier
"skeleton robust, orientation fragile" reading conflated the VAR regime's 72% sepset
divergence with the DAG regime's small skeleton change; it does not hold.

What survives, and is worth reporting:

- FP32 is exact in every regime tested (0 flips, 0 edge changes, 0 orientation changes).
- TF32 is mildly affected and tracks FP16 exactly, as the shared 10-bit mantissa requires.
- BF16 and below degrade proportionally and across all channels at once.
- FP8 on autocorrelated data collapses by **saturation**, not rounding (|r| >= 0.96875
  becomes exactly 1.0), so it must be reported through the status breakdown.

### Conflicting colliders

`n_conflicts` counts distinct edges that two unshielded triples demand in opposite
directions — a graph with any is **not a valid CPDAG**. FP64 itself has 570 (DAG) and 2
(VAR), so conflicts are a property of PC on an imperfect skeleton, not of precision. But
they grow with precision loss (VAR: 2 → 282 at E5M2), which is worth reporting as its own
degradation channel.

## Gate status

| gate | test | status |
|---|---|---|
| G0 | E0.1 noise floor | ✅ zero, cross-process, with a working anti-control |
| G1 | flips at FP32 and TF32 | ✅ does not fire — TF32 reaches 0.28% at φ=0.99, 1.5% at level 2. FP32 clean in every regime reachable so far. |
| G2 | amplification ≈ 1 | ⚠️ **fires.** Every count-based ratio is O(1) or below (0.004–2.5). C1 is not supported in any channel. Pre-registered negative result. |
| G3 | time-series stays in levels 0–1 | still open — needs cascade mode, not just VAR data |

G1 deserves care in the write-up. The headline is **not** "low precision is dangerous" —
it is that *true FP32 is clean in every regime tested, while TF32 is not*, and TF32 is what
PyTorch silently substitutes for FP32 on Ampere and later. That is the practitioner-facing
result.

## Decisions a reviewer will challenge

1. **Rounded inputs, FP64 accumulation.** Models low-precision *storage* with a wide
   accumulator, which is how FP8 tensor-core matmul behaves. It does **not** model
   low-precision accumulation. — *measured: storage only; `round_stage` records it.*
2. **Unit diagonal under rounding.** Forced to exactly 1.0. It is representable in every
   format in the ladder, so this is currently a no-op; it stops being one under any
   rescaling. — *decision: forced, recorded in `keep_unit_diag`.*
3. **Symmetry under rounding.** The correlation matrix is made *bitwise* symmetric once,
   at construction, by mirroring the upper triangle. Elementwise rounding of a bitwise
   symmetric matrix is bitwise symmetric, so every format inherits it and no triangle
   mirroring is needed per format. — *decision: mirror once, assert in `checks`.*
4. **Realised vs requested κ.** Normalising to unit diagonal changes the spectrum;
   realised κ runs ≈ 0.55× requested. — *plots must use realised. `with_condition_number`
   returns both.*
5. **Sample size n.** Enters through `sqrt(n − |S| − 3)`, so it shifts every statistic.
   — *n = 200 for the κ sweeps; must be swept, since near-threshold mass differs by an
   order of magnitude between the SSDBM regime (n = 47–160) and telemetry (n ≈ 10,000).*
6. **P-value underflow.** Resolved structurally: the harness stores `stat`, never `p`.
   `scipy.special.erfc` flushes subnormals at stat ≈ 37.5 (earlier than `math.erfc` at
   ≈ 40); `log10p` uses the asymptotic expansion beyond that.
7. **TF32 rounding mode.** PTX exposes both `cvt.rn` (nearest-even) and `cvt.rna` (ties
   away), and which one is emitted has varied. Default is `rne`, with `rna` and `trunc`
   selectable. **The local GPU has no TF32, so this is uncalibrated: every TF32 number is
   provisional until checked against real hardware.**
9. **NaN never removes an edge.** `skeleton.discover` refuses to treat an undefined
   statistic as "accept the null". The alternative would delete edges on the strength of a
   computation that failed, which is indefensible — but the choice *determines the sign* of
   E4.2's answer: degeneracy retains edges, so errors present as false positives. State the
   policy alongside the result, and report the counterfactual if a reviewer asks.
10. **Amplification denominator.** union / intersection / FP64-replay give a 2.3× spread
   (0.058 / 0.094 / 0.078 at E4M3). Union is the default because it privileges no
   trajectory. All three land below 1, so the conclusion is definition-independent.
8. **ml_dtypes double-rounds.** float64 → float8 goes via float32, so the E4M3 saturation
   boundary sits one float32 ULP below the exact midpoint (0.9687499702 vs 0.96875).
   Immaterial numerically, but it is why the test asserts to 1e-6 and not 1e-9.

## Open questions

- E0.2 as specified expects divergence in the α = 0.01 tail, but `crit(0.01) = 2.576`,
  where every credible normal-CDF implementation agrees to ~1 ULP. Implementations diverge
  above x ≈ 8, where p < 1e-15 — disjoint from where decisions are marginal. The effect
  that *would* show up at α = 0.01 is linear-algebra ordering and FMA contraction
  perturbing ρ, amplified by `1/(1−r²)` as |r| → 1 — strongest in the φ → 1 regime.
  Restructure E0.2 to separate the two before spending cluster time on it.
- E5.1 contrasts datacenter against consumer GPUs. Nibi provides H100s; the only consumer
  part to hand is the MX230. Confirm whether an L40S or similar is reachable.
