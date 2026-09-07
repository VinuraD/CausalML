# precision_pc — experiment harness

Harness for *Precision Requirements for GPU-Accelerated Constraint-Based Causal
Discovery* (`precision_causal_discovery_proposal.pdf`).

**Status: Tiers 0-2 (incl. orientation) plus E3.1, E4.1, E4.2, and the cluster fan-out.
171 tests green, none of which needs a cluster or a GPU.** Tier 5 and the GPU engine are
not built yet.

```powershell
conda activate causal
cd D:\Github\CausalML\cuda\CI_test
python -m precision_pc doctor                       # environment check, no computation
python -m precision_pc gate                         # E0.1 + E0.1b, records the result
python -m precision_pc run configs/e1_kappa.yaml --run-id e1_k1e8
python -m precision_pc report e1_k1e8
python -m pytest
```

## Why it is shaped this way

Five design decisions carry most of the weight. Each exists to stop a *wrong* result
rather than a broken one.

**The harness stores the test statistic, never the p-value.** `2*(1 - Phi(x))` is exactly
`0.0` above x ≈ 8.3, and a p-value that has flushed to zero can never flip — which biases
every flip rate downward, in exactly the low-precision formats the paper is about. Because
the Fisher-z statistic is standard normal, `p < α ⟺ stat > crit(α)` holds exactly for a
single constant, independent of `n` and `|S|`. So thresholds are applied post hoc at any α
without recomputation, and the underflow is eliminated structurally.

**Every test carries a status.** `NaN > threshold` is `False` — "not rejected" — which is
the same decision as a well-behaved test that was also not rejected. Folding those
together would count degenerate low-precision output as *agreement*. `metrics.flip_counts`
refuses to run unless the caller states a policy for undefined decisions, and reports
`n_undefined` separately. This is not hypothetical: at κ≈5e7, E4M3 produces 2103 undefined
decisions out of 3080 tests.

**Test sets are content-addressed.** Replay joins per-format results positionally. If two
test sets are ever mixed, test #7134 is compared against a *different* test #7134, and
flips are manufactured out of nothing with no trace in the output. Every result records
the `testset_digest` it was computed against, and comparison across differing digests
raises.

**The gate crosses a process boundary and has an anti-control.** Running deterministic
code twice in one process is bitwise identical whether or not the pinning worked, so an
in-process double-run is a gate with almost no power. `gate.run` re-runs the reference
cell in a fresh interpreter, and `run_anticontrol` injects a 1-ULP perturbation to prove
the apparatus can still detect a difference. A gate that can only pass is not a gate.

**A shard's existence means the cell is done.** Cells are content-addressed by a key that
includes the experiment, mode, testset digest *and* code digest, so a code edit cannot
silently reuse stale results. Resume is free, and the eventual cluster fan-out is a
work-stealing loop over `precision-pc cells` / `run-cell` rather than a new architecture.

## Layout

| module | responsibility |
|---|---|
| `env.py` | thread pinning, and `verify()` which checks it via threadpoolctl rather than trusting env vars |
| `checks.py` | stage-boundary assertions: dtype (NEP 50), exact symmetry, unit diagonal, finiteness |
| `provenance.py` | `code_digest` / `config_digest` / `env_digest`, git state, BLAS vendor, TF32 flags |
| `formats.py` | the ladder as a registry; hand-rolled TF32; per-format saturation points |
| `stats.py` | Fisher-z; `crit(α)`; `log10p` for the deep tail; z-transform and CDF as E0.3 factors |
| `corr.py` | exact symmetrisation, submatrix extraction, controlled-κ generator |
| `parcorr.py` | three routes (inv / chol / regression), each returning `(rho, status, κ_proxy, min_diag)` |
| `testset.py` | the SoA descriptor batch, its digest, and union/Jaccard for the bias measurement |
| `engine.py` | one cell = fixed test set × format × variants |
| `metrics.py` | flip accounting, bitwise identity, κ coverage, corrected amplification |
| `schema.py` / `store.py` | declared Parquet schemas, atomic content-addressed shards, subsampling |
| `gate.py` | E0.1 and E0.1b as cached, provenance-keyed evidence |
| `skeleton.py` | cascade mode: PC-stable and order-dependent search, recording trajectory + sepsets |
| `orient.py` | v-structures, Meek R1-R4, CPDAG comparison (cross-validated vs causal-learn) |
| `cascade.py` | the three-arm driver — free search for the numerator, union replay for the denominator |
| `datasets/gaussian.py` | random DAGs and linear-Gaussian SCMs, population or sampled |
| `datasets/var.py` | VAR / AR(1) / common-factor processes and lag embedding (Tier 3) |
| `runner.py` / `cli.py` | config → cells → shards; `doctor`, `gate`, `run`, `cells`, `run-cell`, `report` |

## Two corrections to the proposal

**The α = 1.0 pairing device.** It fixes the test set exactly at level 0. At level ℓ ≥ 1
nothing is removed either, so the level-ℓ test count is `C(N,2)·C(N−2,ℓ)` — affordable at
level 1 for N ≲ 1000, dead at level 2 and on the replication arm. The sharper objection is
not cost: at α = 1.0 the conditioning sets are drawn from the *complete* graph, whereas a
real run draws them from the estimated adjacency, which is enriched for genuine neighbours
and therefore for ill-conditioned submatrices. **α = 1.0 samples better-conditioned
matrices than the algorithm actually meets, biasing the flip rate toward the G1 kill
condition.** Hence three modes — `cascade`, `replay`, and `union` (the union of every
format's trajectory, which removes the bias rather than reporting it). `mode` is recorded
on every row.

**Amplification.** `SHD / flip_rate` has units of *edges* and moves by orders of magnitude
with N for no scientific reason. `metrics.Amplification` reports
`edges_changed / n_flipped_tests` — dimensionless, and what C1 actually claims — and
returns `None`, not infinity, when nothing flipped. The proposal's literal ratio is kept
under the name `shd_per_flip_rate`.

## A third correction, from the measurements

**Amplification (C1) is not supported.** Every count-based ratio is O(1) or below:
0.004–0.10 skeleton edges changed per flipped CI decision, and 0.05–2.5 orientation changes
per differing separating set. The skeleton absorbs flips because an edge is removed if
**any** separating set accepts. Skeleton and orientation degrade at roughly the same
proportional rate — there is no fragile channel. This is the pre-registered negative
outcome §6 asks for. (A ratio-of-percentages figure quoted earlier in development was
invalid — different denominators over different populations; see `LAB_BOOK.md`.)

## Running it on a cluster

```
python -m precision_pc hpc init          # write ~/.precision-pc/hpc.toml
python -m precision_pc hpc doctor        # no network: config, ssh, key, TTY
python -m precision_pc hpc preflight     # ONE connection, one second-factor prompt
python -m precision_pc hpc stage         # build the remote venv on a login node
python -m precision_pc hpc submit configs/e1_kappa.yaml
python -m precision_pc hpc status  e1_kappa      # exit 3 while queued
python -m precision_pc hpc fetch   e1_kappa
python -m precision_pc report      e1_kappa
```

The fan-out is a **work-stealing pull**, not a static partition: every array task runs the
same worker, claims the next unclaimed cell via `os.link`, computes it, writes its shard.
Cell costs vary by orders of magnitude — an FP8 cell hits the pseudo-inverse fallback on
most tests — so a fixed `k::N` split would leave most tasks idle. It also decouples task
count from cell count, so `--array=0-15%8` is a pure throughput knob and `MaxArraySize` is
never a limit.

Every design choice in the transport exists to spend fewer **connections**, because on an
MFA cluster a connection costs an interactive second factor: submit is two connections,
everything else is one, and `status` returns progress, queue state and a log excerpt from
a single probe.

`mock` is a second backend that runs the real worker as local subprocesses against a fake
cluster directory. The whole submit → poll → fetch loop, the claim protocol and the
provenance check are therefore tested on a laptop; only SSH transport and SLURM itself
are not.

## Not built yet

Tier 5 and the GPU engine, E3.2's level-distribution shift, and E4.3/E4.4. The TF32
rounding mode also remains uncalibrated — see `LAB_BOOK.md` §7 — because the local Pascal
card has no TF32 at all, so every TF32 number is still marked provisional.
