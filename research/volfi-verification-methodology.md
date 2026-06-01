# volfi catastrophic-tail finding: verification methodology

Before publishing the volfi finding we ran six independent checks designed to catch the most common cause of false "library X has a bug" claims: a misconfigured adapter. All six agreed. Source for each check is in `bench/python/` and `tests/`.

## 1. Hand-coded direct repro outside the oracle adapter

We pulled the top 10 worst volfi rows from the oracle output and called `volfi.iv_call(F, K, disc, T, c)` directly in a standalone script, with no adapter layer of any kind. All 10 rows reproduced ULP-identical to the oracle run. Maximum disagreement across the 10 rows was 0.0 floats.

## 2. Put-call parity via two independent paths

For each put-side row we computed σ two ways. Path A: our oracle's parity transform `c = p + S - K*disc`, then `volfi.iv_call(c)`. Path B: volfi's own `bs_call` from their published `bench_vollib.py` evaluated at `sigma_true` to produce a call price, then `volfi.iv_call` on that price. Path A and Path B agreed to better than 1e-13 absolute σ on every row tested, and both produced the same catastrophic answer. The error does not depend on which side of parity the row originated on.

## 3. Alternate volfi entry point

We re-ran the worst rows through `volfi.iv_otm` instead of `volfi.iv_call`. The error reproduced on both paths. On Row 10 (true σ ≈ 0.786), `iv_otm` returned 0.4903 and `iv_call` returned 0.4961; both are off by roughly 0.29-0.30 absolute. The failure is not confined to a single volfi entry point.

## 4. volfi-self-priced, volfi-self-inverted

To remove our BS-pricer from the loop entirely, we used volfi's own `bs_call` formula to construct the call price `c` at `sigma_true`, then fed that price back into `volfi.iv_call`. The returned σ was the same wrong value. The failure mode is independent of which Black-Scholes formula produced the input price.

## 5. Bench pattern equivalence to volfi's own `bench_vollib.py`

The call shape used in our benchmark, `volfi.iv_call(F, K, disc, T, c)`, is the same pattern volfi's own published benchmark `bench_vollib.py` uses. The prices we feed volfi agree with what volfi's own `bs_call` produces to within 3.55e-15 absolute, which is the f64 ULP at the relevant magnitudes.

## 6. Documented OTM-native API equivalence

Volfi's documented OTM-native API is `volfi.otm_context(h)` returning a context, then `ctx.iv(c, t)`. The v1.0.1 verification used `volfi.iv_call`; v1.0.2 also exercised `volfi.iv_otm`. Both produced catastrophic σ on the deep-wing failure regime. For v1.1.0 the 20 worst rows were retested via the precomputed-context API: `volfi.otm_context(h)` followed by `ctx.iv(c, t)`. Result: identical σ to `iv_call` within 2.3e-12 (machine precision) on all 20 rows.

Reading volfi's Python binding source (`bindings/python/src/volfi_py.cpp` lines 95-110) confirms `iv_call`, `ctx.iv`, `iv_otm`, and `iv_call_norm` are all wrappers over the same `volfi::implied_volatility_otm` core. The precomputed-context API caches moneyness-dependent quantities; it does not invoke a different numerical algorithm. Volfi's own documented test reproduces `ctx.iv([0.05], [1.25]) = 0.43990879...`. The defect lives in the core OTM solver, not in any wrapper or parity-adapter convention.

## Aggregate

The failure mode is intrinsic to volfi at this regime; 913 of 99,996 rows show err > 1e-3, with floor-ratio (volfi_err / f64_BS_inversion_floor) median 1.34e+11, max 3.04e+15. Reproduced independently on the CLY-3D 2021 standard grid (5,488 catastrophic / 51,321 rows) and the v×Δ wing-saturated stress grid.
