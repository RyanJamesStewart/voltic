# Implementation spec: Jäckel "Let's be rational" for voltic v1.0

**Source:** Peter Jäckel, *Let's be rational*, Wilmott Magazine (Jan 2015);
canonical PDF at `www.jaeckel.org/LetsBeRational.pdf` (this version: 25 Mar 2016).

**Clean-room provenance:** This spec is derived from the paper PDF only.
Jäckel's reference C++ (`LetsBeRational.7z` from his site) and the
`py_lets_be_rational` Python wrapper have NOT been opened. Spec author reads
the paper, writes the spec; implementation derives from this spec; validation
against `py_lets_be_rational` occurs ONLY after the implementation converges on
its own. Any "TBD" below is a place where the paper alludes but does not
fully spell out — those are derived independently when the implementation
hits them, not back-filled from C++.

---

## 0. Goal

Replace voltic's current direct-Newton kernel with a SIMD implementation of
Jäckel's rational method. Required properties:

1. **Same six-slice API** as `voltic::implied_vol`: `(spot, strike, tte, rate, price, kind) → Vec<f64>`. Same NaN discipline.
2. **Coverage across all moneyness/expiry regimes** where the implied vol is
   recoverable to the f64 conditioning floor — including the deep-OTM-near-
   expiry corner where the current voltic kernel returns NaN.
3. **Accuracy to near-machine precision** in the well-conditioned regions
   (paper claim: "maximum attainable precision … on standard (64 bit floating
   point) hardware … for all possible inputs"). Target initial milestone:
   1e-10 in vol across all bands; eventual: ~1e-15 in central, conditioning-
   floor in the wings.
4. **Lane-packed SIMD**, branch-free where practical (mask-and-compute over
   region branches; segregate by region only if profiling demands it).
5. **Two iterations to convergence** per Jäckel's design — the initial guess
   carries most of the accuracy.

---

## 1. Notation and normalization (Paper §2)

The standard Black formula is

```
B(F, K, σ̂, T, θ) = θ · [F · Φ(θ · (ln(F/K)/(σ̂√T) + σ̂√T/2))
                     − K · Φ(θ · (ln(F/K)/(σ̂√T) − σ̂√T/2))]
```

with `θ = +1` for calls, `θ = −1` for puts, `F` the forward, `K` the strike,
`σ̂` the annualized volatility, `T` the time-to-expiry, and `Φ` the standard
normal CDF.

Jäckel's normalization (2.1)–(2.4):

| symbol | definition                       | role                         |
|--------|----------------------------------|------------------------------|
| `x`    | `ln(F/K)`                        | forward log-moneyness        |
| `σ`    | `σ̂ · √T`                        | total volatility (NOT annual)|
| `b`    | `B / √(F·K)`                     | normalized Black price       |
| `θ`    | `+1` call, `−1` put              | option kind                  |

In normalized form, (2.4):

```
b(x, σ, θ) = θ · [e^(x/2) · Φ(θ · (x/σ + σ/2)) − e^(−x/2) · Φ(θ · (x/σ − σ/2))]
```

**Crucial: throughout this spec, `σ` denotes TOTAL vol = `σ̂√T`, not annual
vol.** Conversion to/from annual vol happens at the API boundary only.

---

## 2. Invariances and canonical reduction (Paper §2, (2.5)–(2.10))

Two symmetries reduce every input to the canonical case `x ≤ 0, θ = +1`
(out-of-the-money calls):

**Reciprocal-strike put-call invariance** (2.5):
```
b(x, σ, θ) = b(−x, σ, −θ)
```

**Time-value put-call invariance** (2.6):
```
b(x, σ, θ) − ι(x, θ) = b(x, σ, −θ) − ι(x, −θ)
```

with the intrinsic-value-in-normalized-form (2.7):
```
ι(x, θ) := (b_max − b_max^(−1))_+    where b_max := e^(θ·x/2)
```

**Implementation: canonicalization routine** `canonicalize(x, β, θ) → (x', β')`
where the output is guaranteed to satisfy `x' ≤ 0` and the corresponding kind
is a call (θ' = +1). Four cases by (sign of x) × (call/put):

| input case            | x'    | β'                              | notes                          |
|-----------------------|-------|---------------------------------|--------------------------------|
| call (θ=+1), x ≤ 0    | x     | β                               | already canonical              |
| call (θ=+1), x > 0    | −x    | (5): b(−x, σ, +1) = b(x, σ, −1) — need the put-of-reciprocal price. Use (2.6): β' = β − ι(x,+1) + ι(−x,+1) |
| put (θ=−1), x ≤ 0     | x     | (2.6): β' = β + ι(x,+1) − ι(x,−1)   | time-value swap to call         |
| put (θ=−1), x > 0     | −x    | combine both: apply (2.5) then (2.6) |                                |

**TBD-1:** Verify the put → call mapping (case rows 3 and 4) via the
canonical (2.6) restated for the normalized form. The paper says "From here
on, we shall only deal with out-of-the-money call options"; the reader
constructs the canonical inverse. Derive carefully, validate on a
round-trip test (canonicalize then de-canonicalize the answer).

After canonicalization: `x ≤ 0`, `θ = +1`, and the bounds (2.9):
```
0 ≤ b ≤ b_max ≤ 1            with b_max = e^(x/2)
```

---

## 3. Asymptotics (Paper §3)

The two limits, accurate to first asymptotic order in σ and 1/σ respectively
(3.3)/(3.4):

**Small-σ limit** (lower-region anchor):
```
b ≈ (2π|x| / (3√3)) · Φ(−|x| / (√3·σ))³                                      (3.3)
```
This is invertible for σ from b — exactly how the lower-region initial
guess is constructed.

**Large-σ limit** (upper-region anchor):
```
b ≈ b_max − 2·Φ(−σ/2)                                                          (3.4)
```
Also invertible.

The crucial property that distinguishes this paper from prior work
(particularly [Jäc06]): both limits are not just dominant-term correct but
**first-order asymptotically invertible** — solving them for σ gives a guess
that is asymptotically correct as β → 0 or β → b_max, not merely bounded.

---

## 4. The four-branch initial guess (Paper §4)

### 4.1 The inflection point (4.1)–(4.2)

`b(x, σ)` (at fixed `x ≤ 0`, viewed as a function of σ ∈ [0, ∞)) has a single
point of inflexion:

```
σ_c = √(2·|x|)                                                                 (4.1)
b_c = b(x, σ_c)                                                                (4.2)
```

For `σ < σ_c`, `b` is convex; for `σ > σ_c`, `b` is concave. The function is
sigmoidal: zero slope at σ=0 and σ→∞, monotonically increasing in between.

### 4.2 The tangent intersections (4.3)–(4.4)

Tangent at the inflection point intersects `b = 0` at `σ_l` and `b = b_max`
at `σ_u`:

```
σ_l = σ_c − b_c / b'(σ_c)                                                      (4.3)
σ_u = σ_c + (b_max − b_c) / b'(σ_c)                                            (4.4)
```

with the vega-in-total-vol form (4.5)/(4.6):
```
b'(σ) = (1/√(2π)) · e^(−½·[(x/σ)² + (σ/2)²])                                   (4.6)
```

The points `(σ_l, b_l)` and `(σ_u, b_u)`:
```
b_l = b(x, σ_l)                                                                (4.7)
b_u = b(x, σ_u)                                                                (4.8)
```

### 4.3 Four-region partition

```
β ∈ [0, b_l]     → lower (asymptotic + rational refinement)
β ∈ [b_l, b_c]   → centre-left (Delbourgo-Gregory rational cubic)
β ∈ [b_c, b_u]   → centre-right (Delbourgo-Gregory rational cubic)
β ∈ [b_u, b_max] → upper (asymptotic + rational refinement)
```

The boundaries `b_l, b_c, b_u, b_max` all depend on `x` (figure 2 in the
paper shows them as functions of `|x|`).

### 4.4 The Delbourgo-Gregory rational cubic (Paper (4.10))

For a smooth interpolation on `[x_l, x_r]` matching levels `f_l, f_r` and
slopes `f'_l, f'_r`:

```
f^rc(x; x_l, x_r, f_l, f_r, f'_l, f'_r, r) =
    [f_r·s³ + (r·f_r − h·f'_r)·s²·(1−s) + (r·f_l + h·f'_l)·s·(1−s)² + f_l·(1−s)³]
  / [1 + (r−3)·s·(1−s)]                                                       (4.10)

h := x_r − x_l                                                                (4.11)
s := (x − x_l) / h                                                            (4.11)
```

Parameter `r > −1` controls the shape; `r → ∞` collapses to the linear form.
The Delbourgo-Gregory paper [DG85] gives monotonicity/convexity-preserving
conditions on `r`. Jäckel chooses `r` to match a desired second derivative
at one edge — formulae (4.12)/(4.13):

```
r_l(x_l, x_r, f_l, f_r, f'_l, f'_r, f''_l) =
    [½·h·f''_l + (f'_r − f'_l)] / (Δ − f'_l)                                  (4.12)
r_r(x_l, x_r, f_l, f_r, f'_l, f'_r, f''_r) =
    [½·h·f''_r + (f'_r − f'_l)] / (f'_r − Δ)                                  (4.13)
```

with `Δ := (f_r − f_l) / h`.

The choice of `r_l` vs `r_r` depends on which edge's second derivative we want
to match — see (4.16): at the inflection point `σ_c`, the natural choice is
the second derivative of `σ(β)` at `β = b_c`, which is `−b''(σ_c)/b'(σ_c)³ = 0`
because `b''(σ_c) ≡ 0` by definition of the inflection.

### 4.5 Centre-left initial guess (Paper (4.17)–(4.19))

```
σ_0(β)|[b_l, b_c] = f_cl^rc(β)                                                 (4.17)
f_cl^rc(β) = f^rc(β; b_l, b_c, σ_l, σ_c, 1/b'_l, 1/b'_c, r_[b_l, b_c])         (4.18)
r_[b_l, b_c] = r_r(b_l, b_c, σ_l, σ_c, 1/b'_l, 1/b'_c, 0)                      (4.19)
```

Interpretation: interpolating `σ(β)` (not `b(σ)`) with values at the two
edges (`σ_l, σ_c`), slopes `dσ/dβ = 1/b'` at the two edges, and second
derivative `d²σ/dβ²|_{β=b_c} = 0` (matched at the right edge → `r_r`).

### 4.6 Centre-right initial guess (Paper (4.20)–(4.22))

```
σ_0(β)|(b_c, b_u] = f_cr^rc(β)                                                 (4.20)
f_cr^rc(β) = f^rc(β; b_c, b_u, σ_c, σ_u, 1/b'_c, 1/b'_u, r_[b_c, b_u])         (4.21)
r_[b_c, b_u] = r_l(b_c, b_u, σ_c, σ_u, 1/b'_c, 1/b'_u, 0)                      (4.22)
```

Same structure as centre-left, mirror-symmetric: the `d²σ/dβ² = 0` match is
now at the left edge → `r_l`.

### 4.7 Upper-region initial guess (Paper (4.23)–(4.30))

Transform via the asymptotic (3.4): define
```
f_u(β) := Φ(−σ(β)/2)                                                           (4.23)
```
which is asymptotically linear in β as β → b_max. Approximate by a
Delbourgo-Gregory rational cubic on `[b_u, b_max]`. The required derivatives
of `f_u` at β = b_max (4.26)/(4.27):
```
lim_{β → b_max} f_u(β) = 0                                                     (4.26)
lim_{β → b_max} f'_u(β) = −½                                                  (4.27)
```

At the left edge (β = b_u), need `f_u(b_u)`, `f'_u(b_u)`, `f''_u(b_u)`
(4.24)/(4.25):
```
f'_u(β) = −½ · e^(½·(z²/σ²))            with z := σ                            (4.24)?
f''_u(β) = √(π/2) · (z²/σ³) · e^((z²/σ²) + (σ²/8))                            (4.25)?
```

**TBD-2 (RESOLVED):** Derive directly from (4.23) and (4.6):
```
f_u(β) = Φ(−σ/2)
f'_u(β) = d/dβ Φ(−σ/2) = −½·φ(σ/2)·(dσ/dβ) = −½·φ(σ/2)·(1/b'(σ))    [φ even]
       = −½ · (1/√(2π))·exp(−σ²/8) · √(2π)·exp((x/σ)²/2 + σ²/8)
       = −½ · exp((x/σ)²/2)                                          (TBD-2.a)
f''_u(β) = d/dβ f'_u = (df'_u/dσ)·(1/b'(σ))
        = (−½·exp((x/σ)²/2))·(−x²/σ³)·(1/b'(σ))
        = (x²/(2σ³))·exp((x/σ)²/2)·√(2π)·exp((x/σ)²/2 + σ²/8)
        = √(π/2)·(x²/σ³)·exp((x/σ)² + σ²/8)                          (TBD-2.b)
```
Sanity: at β → b_max (σ → ∞), `(x/σ)² → 0`, so `f'_u → −½`. Matches (4.27) ✓.

Coefficient check: `½·√(2π) = √(2π)/2 = √π·√2/2 = √π/√2 = √(π/2)` ✓.

These match paper (4.24)/(4.25) IF the paper's `z` in (4.25) is read as
`z := |x|` (local to the upper region, distinct from lower-region z).

Then:
```
f_u^rc(β) := f^rc(β; b_u, b_max, f_u(b_u), 0, f'_u(b_u), −½, r_[b_u, b_max])   (4.28)
r_[b_u, b_max] = r_l(b_u, b_max, f_u(b_u), 0, f'_u(b_u), −½, f''_u(b_u))       (4.29)
σ_0(β)|(b_u, b_max] = −2·Φ⁻¹(f_u^rc(β))                                        (4.30)
```

### 4.8 Lower-region initial guess (Paper (4.31)–(4.38))

Transform via the asymptotic (3.3): define
```
f_l(β) := (2π·|x| / (3√3)) · Φ(z)³           with z := −|x| / (√3·σ)           (4.31)
```
The asymptotic equality `b ≈ f_l(σ)` holds as σ → 0. Inverting (4.31)
gives σ in closed form (modulo Φ⁻¹).

Derivatives at β → 0 (4.34)/(4.35):
```
lim_{β → 0} f_l(β) = 0                                                         (4.34)
lim_{β → 0} f'_l(β) = 1                                                        (4.35)
```

At the right edge (β = b_l), need `f_l(b_l)`, `f'_l(b_l)`, `f''_l(b_l)`
(4.32)/(4.33):
```
f'_l(β) = 2π · z² · Φ(z)² · e^(z²/2)                                          (4.32)
f''_l(β) = (π/6) · (z²/σ³) · Φ(z) · e^(2z² + a²/4) ·
           [8√3·σ·|x| + (3σ²·(σ²−8) − 8x²) · Φ(z)/φ(z)]                       (4.33)
```

**TBD-3 (PARTIAL — paper formula appears to have typo):**

Re-deriving from (4.31) `f_l(β) := (2π|x|/(3√3))·Φ(z)³` with `z := −|x|/(√3·σ)`:

```
df_l/dσ = (2π|x|/(√3))·Φ(z)²·φ(z)·(dz/dσ) = (2π·x²/(3σ²))·(1/√(2π))·Φ(z)²·exp(−z²/2)
f'_l(β) = (df_l/dσ)·(1/b'(σ))
       = (2π·x²/(3σ²))·(1/√(2π))·Φ(z)²·exp(−z²/2) · √(2π)·exp((x/σ)²/2 + σ²/8)
       = (2π·x²/(3σ²))·Φ(z)²·exp((x/σ)²/2 − z²/2 + σ²/8)
       = (2π·x²/(3σ²))·Φ(z)²·exp(z² + σ²/8)        [(x/σ)² = 3z², so (x/σ)²/2 − z²/2 = z²]
       = 2π·z²·Φ(z)²·exp(z² + σ²/8)                                 (TBD-3.a)
```

**This differs from paper (4.32)** `f'_l = 2π·z²·Φ(z)²·exp(z²/2)`. The paper's
form gives `lim_{β→0} f'_l = 0` (since `2π·z²·Φ(z)²·exp(z²/2) ≈ exp(−z²/2) → 0`
as `z → −∞`), but paper (4.35) states `lim_{β→0} f'_l = 1`. My form gives:
`2π·z²·Φ(z)²·exp(z²+σ²/8) ≈ z²·(z⁻²)·exp(σ²/8) → exp(0) = 1` ✓.

Concluding: paper (4.32) has a typesetting typo (`exp(z²/2)` should be
`exp(z² + σ²/8)`). Use the corrected form.

`f''_l(β)` is then derived by `d/dσ` of (TBD-3.a) times `(1/b'(σ))`. The
derivation is mechanical but lengthy; resolves the `a` symbol in paper
(4.33) by direct computation rather than trying to interpret the typo.
Pin against finite differences in tests; documented in code at point of
use.

Then the inversion (4.38):
```
σ_0(β)|[0, b_l] = | x/√3 · [Φ⁻¹(√3 · ∛(f_l^rc(β) / (2π·|x|)))]⁻¹ |              (4.38)
```

with the rational cubic for `f_l` itself:
```
f_l^rc(β) := f^rc(β; 0, b_l, 0, f_l(b_l), 1, f'_l(b_l), r_[0, b_l])           (4.36)
r_[0, b_l] = r_r(0, b_l, 0, f_l(b_l), 1, f'_l(b_l), f''_l(b_l))                (4.37)
```

### 4.9 Net initial guess (Paper (4.39))

```
σ_0(β) = {
    expression (4.38)       for β ∈ [0, b_l]
    f_cl^rc(β)              for β ∈ [b_l, b_c]
    f_cr^rc(β)              for β ∈ [b_l, b_u]      [paper says [b_l, b_u], probably (b_c, b_u]]
    −2 · Φ⁻¹(f_u^rc(β))     for β ∈ (b_u, b_max)
}                                                                              (4.39)
```

C¹ overall; σ_0'' discontinuous at b_l and b_u; σ_0''' discontinuous at b_c.

---

## 5. The iteration (Paper §5)

### 5.1 Three objective functions by region (Paper (5.1))

```
g(σ) = {
    1/ln(b(σ)) − 1/ln(β)                  for β ∈ [0, b_l]      (lower)
    b(σ) − β                              for β ∈ [b_l, b̄_u]   (middle)
    ln((b_max − β)/(b_max − b(σ)))        for β ∈ (b̄_u, b_max) (upper)
}                                                                              (5.1)
b̄_u := max(b_u, b_max/2)                                                      (5.2)
```

The transformations are chosen to make the Lagrange inversion series of
`g(σ)` converge well in each region — i.e., to make `σ(g)` near-linear in
`g` so that a low-order rational iteration step lands near the root.

### 5.2 Householder's method of order 3 (paper text, §5; Householder 1970)

Convergence order 4. The iteration takes the form (paraphrased):

```
σ_{n+1} = σ_n − [g · (1/(g')) · (1 − (g · g'')/(2 · (g')²))]
         / [1 − (g · g''/(g')²) + (g·g'''/(6·(g')³))]                        (paper text, no equation #)
```

**TBD-4 (RESOLVED):** The paper exhibits Halley (5.4) and Chebyshev (5.3)
as examples of order-3 methods (convergence order 3), but only NAMES the
Householder order-3 method (convergence order 4) without explicit formula.
Derivation from Householder's general construction:

Householder's method of order n applies the recursion
```
σ_{n+1} = σ_n + n · (1/g)^{(n−1)} / (1/g)^{(n)}
```
where `(1/g)^{(k)}` denotes the k-th derivative of `1/g` w.r.t. σ.

For order 3 (convergence order 4):
```
(1/g)''   = [2(g')² − g·g'']   / g³
(1/g)'''  = [6g·g'·g'' − g²·g''' − 6(g')³] / g⁴
```
giving the iteration
```
σ_{n+1} = σ_n + 3·[(2(g')² − g·g'')/g³] / [(6g·g'·g'' − g²·g''' − 6(g')³)/g⁴]
        = σ_n + 3·g·(2(g')² − g·g'')  /  (6g·g'·g'' − g²·g''' − 6(g')³)
        = σ_n − 3·g·(2(g')² − g·g'')  /  (6(g')³ + g²·g''' − 6g·g'·g'')
```

In compact form with `a := g, b := g', c := g'', d := g'''`:
```
σ_{n+1} = σ_n − 3·a·(2·b² − a·c) / (6·b³ + a²·d − 6·a·b·c)              (Householder-3)
```

**Sanity check (reduction to Newton when c = d = 0):**
```
σ_{n+1} = σ_n − 3·a·(2·b²) / (6·b³) = σ_n − a/b   ✓
```

**Numerical-safety note:** the denominator `6b³ + a²d − 6abc` can vanish in
pathological cases (the same condition that would make Halley's denominator
small). Implementation guards with a small floor and falls back to a
Halley step (or one extra Newton step) if the denominator's magnitude is
near the f64 floor. Validate the fallback path against finite differences
in `tests/jackel_iteration.rs`.

### 5.3 Derivatives of the objective functions

For each region, we need g, g', g'', g''' at the current σ.

**Middle region** (`g = b − β`):
- `g(σ) = b(x,σ) − β`
- `g'(σ) = b'(σ)`  [already have, (4.6)]
- `g''(σ) = b''(σ)`
- `g'''(σ) = b'''(σ)`

So we need analytic forms for b'', b'''. Let `A(σ) := −½·(x²/σ² + σ²/4)`.
Then `b'(σ) = (1/√(2π))·exp(A(σ))` and:
```
A'(σ)  = −½·(−2·x²/σ³ + σ/2)  = x²/σ³ − σ/4         [σ/4 not σ/2: d(σ²/4)/dσ = σ/2, then ·(−½) = −σ/4]
A''(σ) = −3·x²/σ⁴ − 1/4

b''(σ)  = b'(σ) · A'(σ)
        = b'(σ) · (x²/σ³ − σ/4)
b'''(σ) = b''(σ) · A'(σ) + b'(σ) · A''(σ)
        = b'(σ) · [ (x²/σ³ − σ/4)² + (−3x²/σ⁴ − 1/4) ]
        = b'(σ) · [ (x²/σ³ − σ/4)² − 3x²/σ⁴ − 1/4 ]
```

**Lower region** (`g = 1/ln(b) − 1/ln(β)`):

Let `u(σ) := 1/ln(b(σ))`. Then `g = u − const`, so `g^(k) = u^(k)` for k ≥ 1.
```
u'(σ) = −b'(σ) / (b(σ) · ln(b)²)
u''(σ) = compute by quotient/chain
u'''(σ) = compute by quotient/chain
```
**TBD-5:** Derive `u'', u'''` symbolically. Verify against finite differences.

**Upper region** (`g = ln((b_max − β)/(b_max − b(σ)))`):

Let `w(σ) := ln(b_max − b(σ))`. Then `g = ln(b_max − β) − w`, so
`g^(k) = −w^(k)` for k ≥ 1.
```
w'(σ) = −b'(σ) / (b_max − b(σ))
w''(σ) = compute
w'''(σ) = compute
```
**TBD-6:** Derive `w'', w'''` symbolically. Same finite-difference check.

### 5.4 Convergence after two iterations

Paper claim: two iterations suffice for f64. Implementation policy: run
exactly two iterations always (avoids the iteration-count divergence the
paper criticizes prior work for). Post-iteration acceptance check: same
priced_ok / inside / vega_ok gate voltic already uses (lib.rs).

---

## 6. The high-precision normalized Black function (Paper abstract, footnote)

The abstract emphasizes: "of crucial importance for the precision of the
implied volatility is a highly accurate Black function that minimizes
round-off errors and numerical truncations in the various parameter limits.
We implement the Black call option price by the aid of Cody's [Cod69, Cod90]
rational approximation for the complementary error function erfc(·) and its
little known cousin, the scaled complementary error function erfcx(·)."

### 6.1 The cancellation problem in b(x, σ)

The naïve form
```
b(x, σ) = e^(x/2)·Φ(x/σ + σ/2) − e^(−x/2)·Φ(x/σ − σ/2)
```
loses catastrophic precision in the centre region when both Φ values are
close to ½ — their difference is what we need, but cancellation kills
~50% of f64 mantissa bits in the worst case.

The cure: rewrite via the **erfc / erfcx pair**.

Standard normal CDF:
```
Φ(y) = ½ · erfc(−y/√2)
1 − Φ(y) = ½ · erfc(y/√2)
```

Scaled erfc:
```
erfcx(z) := e^(z²) · erfc(z)        [well-conditioned at large z]
```

### 6.2 The b(x, σ) algorithm via erfcx (Paper, abstract)

The paper does not give the explicit erfcx-based formula for `b`. Derive:
canonicalize to `x ≤ 0`; let `η := −x/σ + σ/2` (so the OTM call has `η > 0`
for the "long" Φ argument). Then:
```
b(x, σ) = e^(x/2)·[1 − ½·erfc((x/σ + σ/2)/√2)]
        − e^(−x/2)·[1 − ½·erfc((x/σ − σ/2)/√2)]
```
Rearrange and group with `erfcx` to factor the `exp` factors that would
otherwise underflow:

**TBD-7:** Derive the precision-preserving erfcx grouping that gives a
form free of catastrophic cancellation across the full sigmoid. Pin against
naive `b(x,σ)` in the well-conditioned centre (must agree to ~1e-15) and
against the (3.3)/(3.4) asymptotics in the tails (must converge to those as
σ → 0 / σ → ∞).

### 6.3 Cody's erfc / erfcx (voltic has this)

voltic's `src/norm.rs` already implements Cody's rational-Chebyshev erfc
(`phi_cody`). erfcx can be reduced to erfc via `erfcx(z) = e^(z²)·erfc(z)`
for `z` small enough to not overflow `e^(z²)`; for large `z`, use the
asymptotic series form directly without ever forming `e^(z²)`.

**Implementation note:** Extend `src/norm.rs` with a vectorized `erfcx`
function. The three-band Cody structure (`y ≤ 0.5`, `0.5 < y ≤ 4`, `y > 4`)
is already in place; erfcx is the same partition with the `e^(−y²)` factor
omitted.

---

## 7. SIMD implementation strategy

### 7.1 Region branching

Four initial-guess branches × three objective-function branches = up to 12
distinct code paths per lane. SIMD divergence is the cost.

**Strategy A: mask-and-compute (default for v1.0 first cut).** Compute all
four initial-guess values and all three objective-function residuals per
chunk; select by region mask. Cost: 4× the initial-guess work, 3× the
iteration work, but zero divergence. Acceptable for a first implementation;
optimize if profiling demands.

**Strategy B: lane-segregation.** Pre-classify the batch by region, repack
lanes into homogeneous chunks before SIMD dispatch. Caller-visible order
preserved via index tracking. More code; lower wasted work. Defer to a
v1.0.1 optimization pass.

### 7.2 Lane safety

The b-function evaluation must never produce NaN/Inf in the masked-off
lanes (those are computing junk inputs and the result is discarded, but a
NaN poisons downstream arithmetic if the mask isn't applied correctly).
Cure: clamp inputs to safe ranges before transcendentals, then mask the
result.

### 7.3 The chunk loop structure

```rust
// One chunk:
// 1. Load 8 lanes of (S, K, T, r, P, kind).
// 2. Apply BS-side normalization: compute x, β = price/√(F·K).
// 3. Canonicalize: map all lanes to x ≤ 0, θ = +1.
// 4. Compute region boundaries: σ_c = √(2|x|); σ_l, σ_u via tangent; b_c, b_l, b_u.
// 5. Region classification: 0=lower, 1=centre-left, 2=centre-right, 3=upper.
// 6. Initial guess: compute all four formulas, mask-select per lane.
// 7. Iterate twice (Householder order 3) with region-specific g, g', g'', g'''.
// 8. De-canonicalize σ → σ̂ via σ̂ = σ / √T.
// 9. Acceptance gate (priced_ok / inside / finite); NaN otherwise.
```

### 7.4 New module layout

```
src/
  lib.rs           — existing public API (implied_vol, implied_vol_one, bs_price)
                     gains implied_vol_rational (new) — same signature
  norm.rs          — existing; extend with erfcx
  schadner.rs      — existing; kept for benchmark comparison
  jackel.rs        — NEW: the entire LbR kernel
    canonicalize
    region_boundaries
    initial_guess_lower / _centre_left / _centre_right / _upper
    objective_g / g_prime / g_double_prime / g_triple_prime
    householder3_step
    solve_rational      — orchestrator (initial guess + 2 iterations + acceptance)
    public: implied_vol_rational
  black.rs         — NEW (or extend norm.rs): precision-preserving b(x, σ)
                     via erfcx, plus b', b'', b''' analytic forms
```

---

## 8. Tests and validation gates

### 8.1 Unit tests (clean-room, no oracle)

1. **Canonicalization round-trip:** for arbitrary (x, β, θ), canonicalize then
   de-canonicalize, expect identity to f64 round-off.
2. **Inflection-point invariants:** at `σ_c = √(2|x|)`, verify `b''(σ_c) ≈ 0`
   numerically (finite differences of `b'`).
3. **Asymptotic limits:** for σ → 0, `b(x, σ) / [(2π|x|/(3√3))·Φ(−|x|/(√3·σ))³] → 1`.
   For σ → ∞, `(b_max − b(x, σ)) / (2·Φ(−σ/2)) → 1`.
4. **Round-trip:** for σ_true ∈ {0.01, 0.05, 0.20, 0.50, 1.5, 4.0} × x ∈ {0,
   ±0.1, ±0.5, ±2, ±8}, compute β = b(x, σ_true), solve back, expect
   `|σ_solved − σ_true| < 1e-14·max(σ_true, 1e-3)` after 2 iterations.
5. **Reproducibility against voltic's direct solver on the well-conditioned
   centre:** for moneyness within ±0.5 and T in [1 week, 1 year], the
   rational solver and the existing Newton solver must agree to ~1e-12.
6. **Coverage extension:** on inputs where the existing voltic returns NaN
   (deep OTM near expiry from `tests::deep_otm_short_expiry_is_handled_or_nan`),
   the rational solver returns a finite value matching the conditioning floor.

### 8.2 Cross-validation (post-implementation only)

After all clean-room unit tests pass, set up the py_lets_be_rational oracle
in `bench/python/`. Run on `bench/data.rs` (the existing 1M-option dataset).
Required: max |σ_voltic_rational − σ_py_lbr| < 1e-13 in the central regions,
< conditioning floor in the wings. Diagnose any divergence by re-reading
the paper, NOT by reading Jäckel's C++.

### 8.3 Performance gate

Single-core ns/option must not be more than 3× the existing direct-Newton
kernel (152.8 ns/option). Stretch target: within 1.5×. If the rational
solver is significantly slower than direct Newton on the well-conditioned
centre, the right v1.0 shape may be a hybrid (direct Newton centre +
rational wings) rather than rational everywhere.

---

## 9. Out of scope

- **Adjoint mode / AAD differentiation through the solver.** Not part of
  voltic v1.0.
- **Greeks / sensitivities computed off the solved σ.** Caller's job.
- **Local volatility / implied surface fitting.** Higher-level concern.
- **Beyond f64 precision (long double / mpfr).** No.
- **Schadner kernel replacement.** `src/schadner.rs` stays for benchmark
  comparison purposes; it documents the IG-method approach honestly and is
  cited in the README.

---

## 10. Open questions list (the TBDs to derive before/during implementation)

| #     | Topic                                                                    |
|-------|--------------------------------------------------------------------------|
| TBD-1 | Put↔call canonicalization formula via (2.5)+(2.6), validated by round-trip|
| TBD-2 | `f'_u`, `f''_u` re-derivation from (4.23) + (4.6) by direct diff         |
| TBD-3 | `f''_l` re-derivation from (4.32); resolve the `a` symbol in paper (4.33)|
| TBD-4 | Explicit Householder-order-3 (conv. order 4) formula in (g, g', g'', g''')|
| TBD-5 | `u'(σ), u''(σ), u'''(σ)` for the lower-region objective                  |
| TBD-6 | `w'(σ), w''(σ), w'''(σ)` for the upper-region objective                  |
| TBD-7 | Precision-preserving `b(x, σ)` via erfcx, no cancellation                |

Each TBD has the same protocol: derive symbolically from the paper's
equations, validate numerically against finite differences in the
well-conditioned regime, document the derivation in a comment alongside the
implementation.

---

## 11. Phasing

| phase | scope                                                               | gate                                             |
|-------|---------------------------------------------------------------------|--------------------------------------------------|
| 1     | `black.rs`: high-precision `b(x, σ)` + `b'(σ)` via erfcx           | Unit tests 1–3 of §8.1; pin vs naive in centre.   |
| 2     | `jackel.rs`: canonicalize + region boundaries + initial guess only  | Unit tests 1–4 of §8.1 (round-trip on guess + 0 iters expects ~1e-3 — paper figure 3).|
| 3     | `jackel.rs`: add Householder-3 + objective functions for all 3 regions | Unit test 4 of §8.1 (2-iter convergence to ~1e-14).|
| 4     | Wire `implied_vol_rational` into `lib.rs`; bench harness add row    | Unit tests 5–6; bench throughput within 3× of direct.|
| 5     | py_lets_be_rational oracle; cross-validate                          | §8.2 accuracy bar; bug-fix iteration.            |
| 6     | README v1.0, writeup, release tag                                   | Ryan sign-off.                                   |

Phases 1–3 are the bulk; each has a clean numerical gate that's independent
of any C++ oracle. The oracle (phase 5) is the final cross-check, not the
guide.

---

## 12. Glossary

| symbol           | meaning                                                            |
|------------------|--------------------------------------------------------------------|
| `F`              | forward price `F = S · e^(rT)`                                     |
| `K`              | strike                                                             |
| `T`              | time-to-expiry, years                                              |
| `σ̂`             | annualized vol (the answer we return)                              |
| `σ`              | total vol `= σ̂ · √T` (used throughout the algorithm)              |
| `x`              | log-moneyness `= ln(F/K)`                                          |
| `θ`              | +1 call / −1 put                                                   |
| `B`              | raw Black option price                                             |
| `b`              | normalized Black price `= B/√(F·K)`                                |
| `b_max`          | upper bound on b given x, θ; `e^(θx/2)`                            |
| `β`              | the input normalized price (the thing we're inverting on)          |
| `Φ, φ`           | standard normal CDF and PDF                                        |
| `erfc, erfcx`    | complementary error function and scaled version                    |
| `(σ_c, b_c)`     | inflection point of `b(x, σ)` viewed as a function of σ             |
| `(σ_l, b_l)`     | left tangent intersection (b=0 side)                               |
| `(σ_u, b_u)`     | right tangent intersection (b=b_max side)                          |
| `b'(σ)`          | dB/dσ (the "normalized vega")                                      |
| `f^rc`           | Delbourgo-Gregory rational cubic interpolant (4.10)                |
| `r_l, r_r`       | Delbourgo-Gregory control parameter, edge-specific (4.12/4.13)     |
| `g(σ)`           | region-specific objective whose root is the implied vol            |
