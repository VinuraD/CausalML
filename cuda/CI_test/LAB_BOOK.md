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

E3.2 (does autocorrelation push work into levels ≥ 2?) needs cascade mode, which is now
built — the `levels` table records per-level test counts for every cascade cell. The
experiment itself has not been run, so gate G3 stays open.


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


## External implementations — what the reference GPU code actually does

Read, not built. The decision is to reproduce cuPC's two parallelisation strategies in an
engine where precision is a parameter, rather than depend on code whose precision is fixed
at compile time. Sources checked 2026-09-07.

**cuPC** — `https://github.com/LIS-Laboratory/cupc`, GPL-3.0, last push 2021-03-19.
GitHub ships only **cuPC-S**; **cuPC-E exists only in the original artifact zip** at
`http://lis.ee.sharif.edu/pub/2020_tpds_zjhs_cupc/cupc.zip` (the site's HTTPS certificate
is expired — plain HTTP works).

**1. It is `double` throughout, with no way to change it.** No typedef, no `#define`, no
template parameter, no build flag:
```c
__global__ void cal_Indepl0(double *C, int *G, double th, double *pMax, int n);
extern "C" void Skeleton(double* C, int *P, int *G, double *Th, int *l,
                         int *maxlevel, double *pMax, int* SepSet);
```
A float variant would be a mass `double`→`float` edit across ~200 KB of kernel source.
**This is the citable fact for §1**, and it is stronger than the proposal assumes: the
reference implementation did not weigh the precision choice, it hard-coded FP64.

**2. It computes Fisher-z with the NAIVE form, and inconsistently between levels.** No
`log1p`, no `atanh`, no `normcdf`, no `erfc` appear anywhere in the source:
```c
// level 0 and levels >= 2 -- the ratio form
res = abs( 0.5 * log( abs ( (1 + res) / (1 - res) ) ) );
// level 1 -- the difference-of-logs form
Z   = fabs( 0.5 * (log(fabs((1 + rho))) - log(fabs(1 - rho))) );
```
The two are **not bitwise identical**, so cuPC's level-1 and level-≥2 statistics have
different rounding behaviour *by construction*. Both forms are already factors in
`stats.py` (`z_naive`, `z_log1p`), so E0.3 can put a number on what that costs.

**3. No p-value is ever computed on device.** cuPC compares |Z| against a per-level
threshold precomputed in R — independent convergence on the same "store the statistic,
threshold post hoc" design this harness uses. Its `pMax` output is raw |Z|, *not* on a
p-value scale, despite pcalg documenting that slot as one.

**4. Two apparent bugs in the R driver, both decision-changing and unrelated to precision.**
(a) `threshold[i] <- ...` starting at `i = 0` is a silent no-op in R, so `Th[level]` ends
up `|qnorm(α/2)|/sqrt(n − level − 4)` rather than `sqrt(n − level − 3)` — **every level is
tested against the next level's threshold**. (b) The sepset index reconstruction
(`y <- floor(i/p)+1; x <- i-((y-1)*p)+1`) inverts the mapping incorrectly. Neither is
execution-verified here; both are read from source. This is direct support for E0.3's
thesis that implementation variance exceeds dtype variance.

**5. GPUCSL** (`hpi-epic/gpucsl`, MIT, on PyPI, last push 2022-11-23) is a cleaned-up
cuPC-S with template-based levels — and it carries a live unexamined precision choice. All
CI math is `double`, but the z accumulator is `float`: `float *zmin`, and
`write_atomic_minimum(&zmin[...], (float)Z)`. Its reported p-values therefore derive from
a **float32** statistic. It also reproduces cuPC's threshold off-by-one faithfully, so
**cross-checking cuPC against GPUCSL would not expose it — only pcalg would.**

**6. TF32 is irrelevant to both.** Neither links cuBLAS or cuSOLVER; all linear algebra is
hand-written in-kernel. TF32 matters only for our own torch-based engine, which does go
through cuBLAS. Per NVIDIA, `CUBLAS_DEFAULT_MATH` does *not* use TF32 for FP32 (it
promises at least the requested mantissa bits), and `NVIDIA_TF32_OVERRIDE=0` is a
library-wide kill switch worth setting explicitly in the GPU job.

**7. CUDA Math API ULP bounds** (Programming Guide §5.5 — the Math API pages themselves
just point at it): `normcdf` 5, `log1p` 1, `log` 1, `sqrt` 0, `atanh` 2, `erfc` 5 (double).
Relevant to E0.2 arm (a): a device-side p-value via `normcdf` would cost 5 ULP, whereas the
`log1p` route to atanh costs 1.

**Licensing.** cuPC is GPL-3.0, GPUCSL is MIT; combining them in one distributed artifact
would make the whole thing GPL-3. Read-and-reproduce takes neither as a dependency, so the
question does not arise — but it must not be forgotten if that ever changes.


## Cluster environment of record — Nibi, staged 2026-09-07

`nibi.alliancecan.ca`, `/home/vinurad/scratch/precision-pc`, `module load python/3.12.4 arrow`.

| package | laptop | Nibi |
|---|---|---|
| python | 3.12.13 | 3.12.4 |
| numpy | 2.4.6 | 2.4.2 |
| scipy | 1.15.3 | **1.18.1** |
| numba | 0.67.0 | 0.65.1 |
| pyarrow | 25.0.1 | 25.0.0 |
| pandas | 2.3.3 | **3.0.5** |
| ml_dtypes | 0.6.0 | 0.5.4 |
| threadpoolctl | 3.6.0 | 3.6.0 |

**Every one of these is in `env_digest`, so the two machines cannot share an E0.1 gate
record and `array.sbatch` re-establishes the noise floor on the compute node before
computing anything.** That is the correct behaviour, not a workaround: a compute node is
genuinely a different environment, and a floor measured on the laptop says nothing there.

Consequence to state in the paper: cluster and laptop results are **not** guaranteed
bitwise identical. `test_results_are_reproducible_between_cluster_and_laptop` proves
agreement only for a *matched* environment. Whether the two agree in practice is now an
empirical question with a recorded answer, which is better than an assumption.

pandas crosses a major version (2.3 → 3.0), but the compute path never imports it — the
worker writes shards through pyarrow, and pandas appears only in `cli.cmd_report`, which
runs locally after `fetch`. So the major bump cannot affect any computed number.

### Two Alliance-specific facts worth recording

**pyarrow is deliberately blocked from pip.** `avail_wheels pyarrow` reports version
`9999`, Alliance's sentinel for "do not install this from pip"; the PyPI fallback then
fails with `from versions: none` because their patched pip (`26.1.2+computecanada`)
applies a constraints file on top of `PIP_CONFIG_FILE`. The supported route is
`module load arrow`, which supplies pyarrow 25.0.0. Two consequences for the harness: the
staging loop must test importability *before* trying pip, or the sentinel reads as a real
failure; and the venv must be built `--system-site-packages`, or a module-provided package
is invisible inside it no matter which modules are loaded.

**`~/scratch` is a symlink.** `pwd -P` resolves it to `/scratch/<user>/...`, which is a
different path string from the `/home/<user>/scratch/...` that the job script bakes in.
Same inode, so it happens to work — but the physical target is not guaranteed to be
mounted identically on the compute nodes, so both `preflight` and `stage.sh` now use
logical `pwd`: absolute, because SLURM expands neither `~` nor `$VARS` in `#SBATCH`
directives, but not symlink-resolved.


## Cross-machine agreement at matched precision — first real measurement

`configs/e0_noise_floor.yaml`, FP64, 3 seeds, 2340 level-0 tests, run twice: once on Nibi
(job 21298780, x86_64 Linux, numpy 2.4.2) and once on the laptop (AMD64 Windows, numpy
2.4.6), then joined per test on `(seed, level, test_idx)`.

| quantity | result |
|---|---|
| bitwise identical | **no** |
| max abs difference in `stat` | 4.44e-15 |
| max relative difference | 7.18e-13 |
| max abs difference in `rho` | 1.94e-16 |
| **decision flips at α ∈ {0.001, 0.01, 0.05}** | **0 / 2340** |
| status codes agreeing | all 2340 |

**The same pattern as the E0.3 implementation finding, on a new axis.** Two machines
differing in CPU architecture, operating system, BLAS build and numpy version disagree at
the ULP level on essentially every test, and agree on every single decision. Ubiquitous
numerical disagreement, invisible at the decision level.

Two things follow.

**This is the CPU-vs-CPU baseline that E0.2 needs.** E0.2 asks whether GPU and CPU agree
at matched precision. Before attributing any disagreement to the device, it has to be
compared against the disagreement two *CPUs* already show — which is now measured, not
assumed. Anything at or below ~1e-13 relative is within the cross-machine floor.

**It settles whether cluster and laptop results may be pooled: they may not.** Cell
digests cover the code but not the environment, so the same config on both machines
produces the same shard filename with different contents. `fetch` now refuses to merge
into a directory holding shards from a different `env_digest`; that guard was written from
principle and is now justified by measurement.

Caveat, stated plainly: this is level 0 on i.i.d. Gaussian data, so κ ≈ 1 and every
statistic sits far from the threshold. It is the easy regime. Whether cross-machine
agreement survives at high κ and higher levels is exactly the question E0.2 must ask, and
this run does not answer it.


## Gate G0 (paper plan) — GPU throughput, with an honest CPU baseline

`torch` batched engine (`precision_pc/gpu.py`) against two CPU baselines, level-1 tests,
NVIDIA MX230 (Pascal, **2 SMs**, 2 GB):

| N | tests | per-test Python loop | batched NumPy | GPU fp64 | **GPU vs batched** | batched vs loop |
|---|---|---|---|---|---|---|
| 20 | 3 420 | 0.289 s | 0.0040 s | 0.0073 s | **0.56×** | 71× |
| 30 | 12 180 | 0.987 s | 0.0138 s | 0.0213 s | **0.65×** | 71× |
| 45 | 42 570 | 3.350 s | 0.0530 s | 0.0946 s | **0.56×** | 63× |
| 60 | 102 660 | 8.607 s | 0.1242 s | 0.1500 s | **0.83×** | 69× |
| 80 | 246 480 | 22.703 s | 0.3564 s | 0.3616 s | **0.99×** | 64× |

**Against the per-test Python loop the GPU looks 75× faster. Against a batched NumPy
implementation of the same algorithm it LOSES, up to parity at ~250k tests.** The 75×
was entirely an artifact of a strawman baseline — exactly what the paper plan warns
against ("report the CPU baseline as the author-maintained implementation, not a
strawman"). Only the second column is reportable.

So on a 2-SM consumer part, G0's kill condition is close to met: the GPU does not beat
batched NumPy anywhere up to 246k tests. The trend is monotone in batch size
(0.56 → 0.99), so the crossover sits just beyond, and an H100 with 132 SMs should cross
far earlier — but that is a prediction, not a measurement, until the GPU job runs there.

**Separately, this exposes a 60–70× optimisation of the existing CPU path.**
`engine.evaluate` loops over tests in Python; the same algorithm batched in NumPy is
63–71× faster at every size tested. Every laptop-side sweep in this study has been paying
that factor.

## Is the simulated low precision faithful? — yes

Every Tier 1–3 number came from *simulated* reduced precision: round the inputs,
accumulate in float64. The GPU can do the other thing — exact inputs, arithmetic actually
in float32 — so the assumption is now testable rather than asserted. 2520 level-2 tests
per row, 6 seeds, max |Δρ| against an FP64 reference:

| κ | A: round inputs, compute fp64 | B: exact inputs, compute fp32 | C: both | flips A / B / C |
|---|---|---|---|---|
| 1e2 | 2.14e-07 | 3.19e-07 | 3.19e-07 | 0 / 0 / 0 |
| 1e4 | 5.99e-06 | 5.26e-06 | 5.26e-06 | 0 / 0 / 0 |
| 1e6 | 4.68e-05 | 9.28e-05 | 9.28e-05 | 0 / 0 / 0 |
| 1e8 | 4.43e-04 | 5.04e-04 | 5.04e-04 | 0 / 0 / 0 |
| 1e10 | 2.89e-03 | 2.82e-03 | 2.82e-03 | 0 / 0 / 0 |

**Simulated and real single precision agree within a factor of two at every condition
number, share the same κ scaling, and give the same (zero) flip count.** The laptop-based
methodology is validated on real hardware. Note also B ≈ C: once the arithmetic runs in
fp32, additionally rounding the inputs changes nothing, because fp32 compute already
rounds every intermediate.

**And zero FP32 flips even at κ = 1e10** — where the *explicit-inverse* route flipped
0.2% at κ = 1e8. Same dtype, same conditioning, different route. Another direct
confirmation of E0.3's thesis that the algorithm choice outweighs the dtype.


## The engine is now batched — 31-41x, bitwise safe

`engine.evaluate` looped over tests in Python. The same algorithm batched in NumPy:

| N | level | tests | loop | batched | speedup |
|---|---|---|---|---|---|
| 20 | 1 | 3 420 | 0.424 s | 0.0114 s | 37× |
| 45 | 1 | 42 570 | 5.776 s | 0.1395 s | 41× |
| 60 | 1 | 102 660 | 13.898 s | 0.3492 s | 40× |
| 16 | 2 | 10 920 | 1.549 s | 0.0490 s | 32× |

End to end, `configs/e1_kappa.yaml` (35 cells) went from roughly two minutes to **8.8 s**.

**The constraint was bitwise identity, not speed.** `code_digest` covers the engine, so a
change moving any decision-bearing value by one ulp would invalidate every computed cell
and force the E0.1 gate to be re-established — in exchange for throughput. So the fast
path calls the SAME LAPACK entry points on stacked arrays (`np.linalg.cholesky`, `solve`,
`inv` are gufuncs over the identical per-matrix kernel). A hand-rolled batched Cholesky
would be faster still and would change the summation order, which is precisely the E0.3
"route matters" axis — not a change to make silently for throughput.

Verified across 42 combinations (iid and controlled-κ up to 1e12, levels 1–2, `inv` and
`chol` routes, float64/float32/bfloat16):

| field | agreement |
|---|---|
| `rho`, `stat`, `status`, `min_diag` | **bitwise identical everywhere** |
| `kappa_proxy` | at most **1 ulp**, on ~1 element in 660 |

The one field that moves is a diagnostic: `kappa_proxy` is binned by log10 for E1.2 and
never enters a decision, so a 1-ulp move cannot change a result. The cause is that NumPy's
stacked and 2-D Cholesky differ in the last ulp for some matrices — the same phenomenon as
the numba-vs-numpy E0.3 finding, on yet another axis. It is asserted in the tests rather
than assumed, so a future change that widens it fails loudly.

Degenerate inputs fall back to the scalar routes for the whole bucket, because NumPy's
stacked `cholesky`/`inv` raise for the entire batch without saying which element failed —
unlike torch's `cholesky_ex`. Exactness is kept; speed is given up only where the data is
degenerate.

**Still unbatched: `skeleton.discover`**, which drives cascade mode and has its own
per-test loop. Replay-mode configs get the full speedup; cascade gets it only in the
union-replay arm. That is the next obvious target.


## Cascade mode is batched too — 5-7x, and exactly equal

`skeleton.discover` also looped per test, so cascade mode did not benefit from the engine
batching. It now processes each level in **waves**: wave k evaluates the k-th candidate of
every still-live edge in one batched call, and edges that accept drop out before wave k+1.

| N | tests | loop | wave-batched | speedup |
|---|---|---|---|---|
| 15 | 997 | 0.113 s | 0.0215 s | 5.2× |
| 25 | 2 353 | 0.273 s | 0.0467 s | 5.8× |
| 40 | 5 114 | 0.591 s | 0.1013 s | 5.8× |
| 60 | 13 124 | 1.790 s | 0.2425 s | 7.4× |

End to end, `configs/e2_1_amplification.yaml` (210 cells) went from about six minutes to
**43.7 s**. Less than the engine's 31–41× because `stop_at_first` resolves most edges in
the first wave or two, so the batches shrink quickly — the win comes from batching across
*edges*, and there are only so many waves.

**Why waves rather than batching every candidate.** The search short-circuits, and the
executed trajectory is a recorded OUTPUT: it feeds union-replay and E3.2's level counts.
Evaluating every candidate at once would run tests the loop never reaches and inflate that
record — changing results, not just timing. Waves execute exactly the same tests in the
same quantity; only the evaluation order differs, and the trajectory is re-sorted into the
loop's edge-major order so even its digest matches.

**Order-dependent PC is deliberately not batched.** With `stable=False` a removal takes
effect immediately, so an edge removed early changes the candidate sets of edges
considered after it — and a wave across edges reorders exactly that dependency. E2.2
compares the two variants, so both must stay exact; that path keeps the loop.

Verified equal on 16 combinations (4 datasets × 2 formats × both `stop_at_first`) for
edges, separating sets, trajectory digest, `stat` bytes, statuses and level counts. And
re-running `e2_1_amplification` after the change reproduced **every per-format total
exactly** — same flips, edge changes, sepset divergence and trajectory sizes as the
pre-batching run.


## E0.2 restructured — and the answer is a well-attributed null

The proposal expects cross-device disagreement at α=0.01 "where implementations diverge
most", and hoped it might be "possibly the strongest single result". It is not, and there
is now a quantitative reason why rather than an impression.

### Arm A — the transcendental, with ρ held identical

Where does implementation disagreement actually live?

| stat | p (erfc) | p (ndtr) | relative disagreement | |
|---|---|---|---|---|
| 2.5758 | 1.0001e-02 | 1.0001e-02 | **7.8e-15** | ← α = 0.01 |
| 3.2900 | 1.0019e-03 | 1.0019e-03 | **2.2e-15** | ← α = 0.001 |
| 8.0 | 1.2442e-15 | 1.3323e-15 | **7.1e-02** | |
| 10.0 | 1.5240e-23 | 0 | 1.0 | |

**Thirteen orders of magnitude apart.** The region where normal-tail implementations
disagree (stat ≳ 8, p < 1e-15) is disjoint from the region where decisions are marginal.
Run as specified, E0.2 measures a difference that cannot reach a threshold.

And CUDA's transcendentals are not the problem the proposal supposes. Against numpy's
`log1p`, on 20 200 values including |r| → 1, statistic scaled to n = 4000:

| implementation | max \|Δstat\| | max relative | flips at α=0.01 |
|---|---|---|---|
| `log1p` CUDA | 1.14e-13 | 3.47e-16 | **0** |
| `atanh` CUDA | 1.14e-13 | 4.03e-16 | **0** |
| `log1p` torch CPU | 5.68e-14 | 3.80e-16 | **0** |
| naive `log((1+r)/(1-r))` | 1.14e-13 | **2.31e-12** | 0 |

CUDA agrees with numpy to 4e-16. The largest disagreement in the table is not the device
at all — it is the *naive z-transform formula*, at 2.3e-12, six orders worse than the
device difference. E0.3's thesis again.

### Arm B — the linear algebra, with the transform held identical

Every backend forms the statistic the same way (host-side float64 `log1p`), so any
difference is summation order, FMA contraction, or the device. `gpu.evaluate` was written
for this: it computes ρ on the device but the statistic on the host.

| regime | backend | max\|Δρ\| | max\|Δstat\| | flips@.01 | amp observed | amp predicted |
|---|---|---|---|---|---|---|
| iid N=16 | inv | 5.6e-17 | 1.3e-15 | 0 | 6.4e1 | 2.3e1 |
| iid N=16 | **gpu** | 1.4e-17 | 8.9e-16 | 0 | 6.4e1 | 2.3e1 |
| VAR φ=0.99 | inv | 4.4e-16 | 1.1e-12 | 0 | 4.9e3 | 4.7e3 |
| VAR φ=0.99 | **gpu** | 3.0e-15 | 1.6e-12 | 0 | 4.7e3 | 4.7e3 |
| factor load=.999 | inv | 1.6e-12 | 1.0e-10 | 0 | 4.9e4 | 4.9e4 |
| factor load=.999 | **gpu** | 2.2e-12 | 1.4e-10 | 0 | 4.9e4 | 4.9e4 |

Three things.

**The amplification mechanism is confirmed.** `amp observed` tracks the analytic
prediction √(n−k−3)/(1−r²) across four orders of magnitude. A ρ perturbation is amplified
into the statistic exactly as `atanh`'s derivative says it should be, which is why |r| → 1
is the danger zone — and why the α=0.01 *tail* was the wrong place to look.

**The GPU is not special.** Its disagreement with the CPU reference is the same order as
`inv` versus `chol` on the same CPU. "GPU and CPU disagree at identical precision" is
true, and unremarkable: it is the magnitude two CPU routes already show.

**Zero flips everywhere**, even at max\|Δstat\| = 1.4e-10. A statistic would have to land
within 1e-10 of the threshold, which across ~10⁴ tests essentially never happens.

### The mechanism, isolated

How far one ulp of ρ travels, n = 4000:

| r | sensitivity | Δstat per ulp |
|---|---|---|
| 0.0 | 6.3e1 | 1.4e-14 |
| 0.99 | 3.2e3 | 3.4e-13 |
| 0.9999 | 3.2e5 | 3.5e-11 |
| 1 − 1e-8 | 3.2e9 | 3.5e-07 |

Seven orders of amplification between r = 0 and near-duplicate variables — the same
`1/(1−r)` structure that governs submatrix conditioning, reappearing in the transform.

### What to report

**Cross-device reproducibility is not a problem for constraint-based causal discovery at
FP64.** Disagreement exists and is measurable, it grows exactly as the analytic
sensitivity predicts, and it never reaches a decision. The restructuring is what makes
that a finding rather than a null: the naive framing would have reported "no effect at
α=0.01" without being able to say whether that was because the effect is absent or
because the experiment looked in the wrong place.

## Gate status

| gate | test | status |
|---|---|---|
| G0 | E0.1 noise floor | ✅ zero, cross-process, with a working anti-control — and independently re-established on Nibi (job 21298780), where `array.sbatch` runs the gate on the compute node before any worker starts |
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
