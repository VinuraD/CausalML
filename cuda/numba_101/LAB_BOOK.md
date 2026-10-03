# Lab book — Lesson 1

Record findings here as you get them, not afterwards. A reviewer will ask about
every line in the "Decisions a reviewer will challenge" section, so fill those in
as you make the decisions.

## Environment of record

Fill in from `python check_env.py`:

- date:
- python / numpy / numba / scipy / ml_dtypes:
- BLAS threads pinned (all five vars = 1): yes / no
- CPU:

## E0.1 — the noise floor

- [ ] `flip_rate(p, p, alpha) == 0.0` exactly, FP64 vs FP64, identical input
- N, n_samples used:
- result:

Nothing below this line means anything until that box is ticked.

## E0.3 — formula vs precision

### 1c — naive `log((1+r)/(1-r))` vs `log1p`

- r at which the naive form's relative error exceeds 1e-12:
- does the z-transform error survive into the p-value, or does erfc hide it?
- figure: `figures/e03_log1p.png`

### 2b — conditioning behaviour of the three routes

- observed degradation order (expected: recursive, then inverse, then Cholesky):
- κ at which each route exceeds 1e-8 relative error:
- figure: `figures/e03_conditioning.png`

**The question this control exists to answer:** is the spread between the three
*methods* larger or smaller than the spread between *dtypes* in E1.1? Answer:

## E1.1 — precision → decisions

- figures: `figures/e11_rho_error.png`, `figures/e11_flip_rate.png`
- flip rate at α = 0.01, κ = 1e2, per format:
- flip rate at α = 0.01, κ = 1e6, per format:
- lowest-precision format with a flip rate below (your threshold):

## E1.2 — cheap κ

- proxy vs exact, log-log correlation:
- κ range over which the proxy is monotone in the truth:
- spread at fixed proxy value:
- speedup vs `np.linalg.cond`:
- figure: `figures/e12_cond_calibration.png`

## Decisions a reviewer will challenge

1. **Rounded inputs, float64 accumulation.** This models low-precision *storage*
   with a wide accumulator, which is how FP8 tensor-core matmul behaves. It does
   **not** model low-precision accumulation — a different experiment. State
   plainly which one you measured.
   - what I measured:
2. **Unit diagonal under rounding.** Kept exact, or rounded with everything else?
   - decision and why:
3. **Symmetry under rounding.** One triangle rounded and mirrored (exact symmetry
   preserved), or both triangles rounded independently?
   - decision and why:
4. **Realised vs requested κ.** Normalising to unit diagonal changes the
   spectrum, so the realised condition number is not the one you asked for.
   Plots use realised: yes / no
5. **Sample size n.** n enters through `sqrt(n - k - 3)`, so it shifts every
   p-value. Which n, and why that one?
6. **P-value underflow.** `2*(1 - norm.cdf(x))` is exactly 0 for x above about
   8.3; the erfc form is not. A flushed-to-zero p-value cannot flip, which biases
   the flip rate downward. Which form produced the numbers reported above?

## Open questions for Lesson 2

-
