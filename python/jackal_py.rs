//! PyO3 bindings — one module, built with `maturin`. This exists as a signal
//! ("the kernel ships as a Python extension if you want it"), not because the
//! repo expects `pip install` traffic. Build: `maturin develop --release` from
//! the repo root with the `python` feature.
//!
//! Exposes one function, `implied_vol`, taking NumPy-or-list arrays of
//! (spot, strike, tte, rate, price) and a bytes/str of `'c'`/`'p'` per option;
//! returns a list of f64 (NaN where jackal returns NaN). Kept deliberately
//! list-based (no `numpy` build dep) — a real release would take `&[f64]`
//! views via `numpy::PyReadonlyArray1`; this is the minimal viable wrapper.

use pyo3::prelude::*;
use pyo3::types::PyList;

use jackal::OptionKind;

/// `implied_vol(spot, strike, tte, rate, price, kinds) -> list[float]`
///
/// `spot`/`strike`/`tte`/`rate`/`price`: equal-length sequences of floats.
/// `kinds`: a sequence the same length, each item truthy-or-`'c'` for a call,
/// falsy-or-`'p'` for a put. Returns the implied vols (NaN where unsolvable).
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
    let kind: Vec<OptionKind> = kinds
        .iter()
        .map(|s| {
            if s.eq_ignore_ascii_case("p") || s.eq_ignore_ascii_case("put") {
                OptionKind::Put
            } else {
                OptionKind::Call
            }
        })
        .collect();
    Ok(jackal::implied_vol(
        &spot, &strike, &tte, &rate, &price, &kind,
    ))
}

#[pymodule]
fn jackal(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(implied_vol_py, m)?)?;
    // touch PyList so the import is not flagged unused on older pyo3
    let _ = std::any::type_name::<PyList>();
    Ok(())
}
