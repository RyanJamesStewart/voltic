# FlashIV log-price chain — equivalence finding for voltic's `b_normalized`

**Source under examination**: Le Floc'h & Healy, *"Implying Volatility: How
Fast Can We Go?"*, arXiv:2605.29102v1 (27 May 2026), Proposition 2 (Eq. 7-9),
the FlashIV log-price residual chain.

**Cross-reference**: Le Floc'h, *"Faster Monotone Implied Volatility Solver"*
(ThiopheneIV), arXiv:2605.22427v3 (27 May 2026), Proposition 3 (Eq. 9-11) —
independent derivation of the same identities.

**Bottom line**: We ported the chain into voltic's lower-region Householder
objective, validated it term-by-term against finite differences and against
voltic's existing `b_normalized` form, and confirmed they are **algebraically
equivalent at f64 precision**. The headline accuracy goal — closing the
deep-OTM |h|≥4 max-σ-error gap (2.19e-11 on the 100k mpmath oracle) — **did
not move** after the swap. We then removed the port and document the finding
here so future readers do not redo the experiment.

## What FlashIV proposes (Section 3)

In FlashIV's normalised OTM coordinates (Eq. 1):

- `x = ln(F_*/K_*) ≤ 0`,  `e^x = F_*/K_* ≤ 1`
- `c = C_OTM / F_*` (the normalised OTM call price)
- `v = σ·√T` (total volatility)

Substitutions: `h = x/v`,  `t = v/2`.

FlashIV Proposition 1, Eq. 4-5 (the cancellation-free Black price identity for
canonical OTM `x ≤ 0`,  `v > 0`):

```
  ln(c) = -½(h² + t²) - ln(2) - x/2 + ln(N⁺ - N⁻)
  N⁺   = erfcx(-(h+t)/√2)
  N⁻   = erfcx(-(h-t)/√2)
```

with `erfcx(z) = exp(z²)·erfc(z)`. Because `erfcx` is bounded positive for
all real `z`, the difference `N⁺ - N⁻` remains computable even when `c`
itself underflows to 0 in f64 (FlashIV Remark 1).

FlashIV Proposition 2, Eq. 7-9 then derives the derivative chain for
`ℓ(v) ≡ ln(c(x, v))`:

```
  ℓ'(v)            = (2/√(2π)) / (N⁺ - N⁻)
  ℓ''(v)/ℓ'(v)     = (h+t)(h-t)/v - ℓ'(v)
  ℓ'''(v)/ℓ'(v)    = (-3h² - t² + (h²-t²)²)/v²
                     - 3·ℓ'(v)·(ℓ''/ℓ')
                     - [ℓ'(v)]²
```

FlashIV's pitch (Remark 2): after `N⁺, N⁻` are produced for the objective
evaluation, all three derivatives need only elementary arithmetic
(~25 mul-adds), no additional erfcx or exp evaluations. This is FlashIV's
claimed cost win for the inner Householder loop.

## Voltic's existing form (`src/black.rs:60-79`)

`b_normalized(x, σ)` already computes the cancellation-free Black price via
an erfcx-difference factoring:

```
  α = -x / (σ·√2),    β = σ / (2·√2)
  factor = ½·e^(-α² - β²)
  s = α + β,           t = |α - β|
  b(x, σ) = factor·(erfcx(t) - erfcx(s))             if α ≥ β
          = e^(-2αβ) - factor·(erfcx(t) + erfcx(s))  otherwise
```

The lower-region objective uses
`L = ln b`,  `L' = b'/b`,  `L'' = b''/b - L'²`,  `L''' = b'''/b - 3·L'·L'' - L'³`,
with `b'`, `b''`, `b'''` from the analytic-derivative module.

The relation between voltic's `b` and FlashIV's `c` is **purely algebraic**:

```
  b(x, σ) = e^(x/2) · c_FlashIV(x, v)        (with v = σ in this comparison)
  ln b    = x/2 + ln c_FlashIV
  d^k/dσ^k [ln b] = d^k/dσ^k [ln c_FlashIV]   (the x/2 term has σ-derivative zero)
```

So the FlashIV chain and the voltic chain produce the **identical**
`(L, L', L'', L''')` quadruple, modulo the constant `x/2` offset on `L` itself
(which cancels out of the objective `g(σ) = 1/L - 1/ln β` to the precision of
the offset, ε-level, in the regions where both forms are accurate).

## The empirical port (held in a private branch, then removed)

We implemented FlashIV §3 verbatim in `src/flashiv.rs` (since deleted):

- `erfcx_fast`: the A&S 4-region rational approximation FlashIV uses for the
  first H3 pre-step (~2.3 ns).
- `ln_c_and_derivs_simd` / `ln_c_and_derivs_scalar`: Proposition 2 chain.
- `asym_otm_seed`: Section 3.3.2 closed-form seed for deep OTM.

We wired `objective_lower` to take its `(L, L', L'', L''')` from the FlashIV
chain instead of from `b_normalized` and the analytic derivatives, then ran
the standard validation gauntlet:

### Finite-difference cross-check

`tests/flashiv.rs::derivative_chain_matches_finite_difference_at_dense_grid`:
central differences of `ℓ', ℓ'', ℓ'''` against the FlashIV analytic forms on
8 grid points covering `(x, σ) ∈ {-0.5..-4.0} × {0.3..2.0}`. All three
derivatives passed to FD-truncation tolerance (ℓ' ~1e-5, ℓ'' ~1e-4,
ℓ''' ~1e-3). The chain is correctly implemented.

### Algebraic identity with `b_normalized`

`tests/flashiv.rs::ln_c_residual_at_deep_otm_matches_composition_in_well_conditioned`:
on a sweep of `(x, σ)` with both forms in their well-conditioned range,
`ln b - (x/2 + ln c_FlashIV)` was below 1e-12 relative on every point.
The two forms agree to f64 epsilon where both are accurate.

### Accuracy on the headline regression band (the load-bearing test)

100k SplitMix64 dataset with mpmath-200-bit reference σ, deep_otm band
`|h| = |x/(σ·√T)| ≥ 4`:

| voltic build               | max |σ_voltic − σ_mpmath| |
| -------------------------- | ------------------------- |
| pre-FlashIV (origin/main)  | 2.188e-11                 |
| post-FlashIV-port          | 2.188e-11                 |

**The headline number does not move.** That is the diagnostic finding.

### Throughput (the FlashIV pitch)

| voltic build               | rational kernel ns/option |
| -------------------------- | ------------------------- |
| pre-FlashIV                | 214                       |
| post-FlashIV-port          | 124                       |

The port wins ~90 ns per call on the rational kernel itself (FlashIV's claim
holds) but in the public bench the rational kernel is reached only on the
NaN-fallback path (the fast Cheb+Halley kernel is the hot path), so the
amortized improvement is -0.1 to -2.3 ns / option.

## Why the algebra doesn't move the accuracy

The diagnostic answer fell out of staring at `b_normalized` and the FlashIV
chain side by side:

> Both forms compute `N⁺ − N⁻` (modulo the `α ≥ β` reflection in voltic's
> form, which is exactly the substitution that makes the subtraction stable).

The cancellation is in the **inner difference of two close erfcx values**.
Voltic's form does it once, in `b_normalized`. FlashIV's form does it once,
in `ℓ'(v) = (2/√(2π)) / (N⁺ - N⁻)`. The two are the same subtraction in
different clothing. The erfcx values themselves are computed by the same
A&S-style rational at the same input precision, so the f64 cancellation floor
is **identical**.

The 2.19e-11 ceiling on |h|≥4 is therefore bounded by the **f64 inversion
floor**: at the input-price magnitude, the lowest f64 representable above the
true price differs from the true price by an amount that, when inverted
through `dσ/dC ≈ 1/vega`, lands a vol displacement of ~2e-11. No algebraic
rearrangement at the solver level can move this; closing it requires either
(a) an inversion in a different objective whose conditioning floor is lower
at this regime, or (b) accepting the floor as a stated limit.

## Why FlashIV is still useful to voltic

The FlashIV paper itself (§3.3.3 case i) carries a separate idea: a
**Bachelier-microscopic branch** that bypasses the BS path entirely when the
price is so small that even `ln(c)` underflows. This is a *different*
algorithm, not the chain in Proposition 2 — it solves NaN, not the
conditioning floor. The Bachelier branch was implemented in voltic v1.1.0
(see `src/jackel.rs` / `bench/bachelier_micro.rs`); see the v1.1.0 release
notes for the closure data.

The takeaway for FlashIV's algebraic Proposition 2: **proven equivalent at
f64; algorithmic structure preserved; not adopted because the win is
unmeasurable in voltic's public-bench amortization model**. The exercise
clarified that voltic's existing form was already at the FlashIV optimum.

---

*Authored 2026-05-31, voltic v1.1.0 release window.*
