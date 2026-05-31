//! PyO3 bindings — one module, built with `maturin`. This exists as a signal
//! ("the kernel ships as a Python extension if you want it"), not because the
//! repo expects `pip install` traffic. Build: `maturin develop --release` from
//! the repo root with the `python` feature.
//!
//! Exposes three functions — `implied_vol` (direct Newton), `implied_vol_rational`
//! (Jäckel Let's-be-rational, the v1.0 headline), and `implied_vol_explicit`
//! (Schadner inverse Gaussian) — each taking equal-length sequences of
//! (spot, strike, tte, rate, price) and a bytes/str of `'c'`/`'p'` per option;
//! each returns a list of f64 (NaN where voltic returns NaN). Kept deliberately
//! list-based (no `numpy` build dep) — a real release would take `&[f64]` views
//! via `numpy::PyReadonlyArray1`; this is the minimal viable wrapper.

use pyo3::prelude::*;
use pyo3::types::PyList;

use crate::OptionKind;

fn parse_kinds(kinds: &[String]) -> Vec<OptionKind> {
    kinds
        .iter()
        .map(|s| {
            if s.eq_ignore_ascii_case("p") || s.eq_ignore_ascii_case("put") {
                OptionKind::Put
            } else {
                OptionKind::Call
            }
        })
        .collect()
}

fn check_lengths(
    spot: &[f64],
    strike: &[f64],
    tte: &[f64],
    rate: &[f64],
    price: &[f64],
    kinds: &[String],
) -> PyResult<()> {
    let n = spot.len();
    if strike.len() != n
        || tte.len() != n
        || rate.len() != n
        || price.len() != n
        || kinds.len() != n
    {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "all input sequences must have the same length",
        ));
    }
    Ok(())
}

/// `implied_vol(spot, strike, tte, rate, price, kinds) -> list[float]`
///
/// The direct Newton kernel. Fastest of the three; returns `NaN` in the
/// deep-OTM-near-expiry corner where Newton stalls.
#[pyfunction]
#[pyo3(name = "implied_vol")]
fn implied_vol_py(
    spot: Vec<f64>,
    strike: Vec<f64>,
    tte: Vec<f64>,
    rate: Vec<f64>,
    price: Vec<f64>,
    kinds: Vec<String>,
) -> PyResult<Vec<f64>> {
    check_lengths(&spot, &strike, &tte, &rate, &price, &kinds)?;
    let kind = parse_kinds(&kinds);
    Ok(crate::implied_vol(
        &spot, &strike, &tte, &rate, &price, &kind,
    ))
}

/// `implied_vol_rational(spot, strike, tte, rate, price, kinds) -> list[float]`
///
/// The Jäckel "Let's be rational" kernel (Wilmott 2015), clean-room SIMD
/// implementation. Machine precision across every moneyness band; zero
/// `NaN` on the canonical synthetic dataset; cross-validated against
/// `py_lets_be_rational` to median 1 ULP, max 2.2e-11. Use this when you
/// need the precision/coverage guarantee; use `implied_vol` when you don't.
#[pyfunction]
#[pyo3(name = "implied_vol_rational")]
fn implied_vol_rational_py(
    spot: Vec<f64>,
    strike: Vec<f64>,
    tte: Vec<f64>,
    rate: Vec<f64>,
    price: Vec<f64>,
    kinds: Vec<String>,
) -> PyResult<Vec<f64>> {
    check_lengths(&spot, &strike, &tte, &rate, &price, &kinds)?;
    let kind = parse_kinds(&kinds);
    Ok(crate::implied_vol_rational(
        &spot, &strike, &tte, &rate, &price, &kind,
    ))
}

/// `implied_vol_explicit(spot, strike, tte, rate, price, kinds) -> list[float]`
///
/// The Schadner (2026) explicit inverse-Gaussian kernel. Same SIMD machinery
/// as the direct kernel; the residual is the IG CDF rather than the BS price.
/// Provided for reproducibility of the head-to-head comparison in the README.
#[pyfunction]
#[pyo3(name = "implied_vol_explicit")]
fn implied_vol_explicit_py(
    spot: Vec<f64>,
    strike: Vec<f64>,
    tte: Vec<f64>,
    rate: Vec<f64>,
    price: Vec<f64>,
    kinds: Vec<String>,
) -> PyResult<Vec<f64>> {
    check_lengths(&spot, &strike, &tte, &rate, &price, &kinds)?;
    let kind = parse_kinds(&kinds);
    Ok(crate::implied_vol_explicit(
        &spot, &strike, &tte, &rate, &price, &kind,
    ))
}

#[pymodule]
fn voltic(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(implied_vol_py, m)?)?;
    m.add_function(wrap_pyfunction!(implied_vol_rational_py, m)?)?;
    m.add_function(wrap_pyfunction!(implied_vol_explicit_py, m)?)?;
    // touch PyList so the import is not flagged unused on older pyo3
    let _ = std::any::type_name::<PyList>();
    Ok(())
}
