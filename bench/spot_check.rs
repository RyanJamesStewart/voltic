// Spot-check: 10 rows verifying typed status ↔ legacy f64 NaN agreement.
use voltic::{bs_price, implied_vol, implied_vol_typed_batch, ImpliedVolStatus, OptionKind};
fn main() {
    // 10 hand-picked rows: 5 well-conditioned + 5 boundary/failure cases.
    let s = [
        100.0, 95.0, 110.0, 120.0, 80.0, 100.0, 100.0, -1.0, 100.0, 100.0,
    ];
    let k = [
        100.0, 100.0, 100.0, 110.0, 90.0, 100.0, 100.0, 100.0, 100.0, 100.0,
    ];
    let t = [1.0, 0.5, 0.25, 2.0, 0.75, 1.0, 0.0, 1.0, 1.0, 1.0];
    let r = [0.02, 0.03, 0.01, 0.04, 0.025, 0.0, 0.0, 0.0, 0.0, 0.0];
    let v_compute = [0.30, 0.25, 0.18, 0.40, 0.22, 0.005, 0.0, 0.0, 0.0, 0.0];
    let kind = [
        OptionKind::Call,
        OptionKind::Put,
        OptionKind::Call,
        OptionKind::Put,
        OptionKind::Call,
        OptionKind::Call,
        OptionKind::Call,
        OptionKind::Call,
        OptionKind::Call,
        OptionKind::Call,
    ];
    let p_compute = bs_price(&s, &k, &t, &r, &v_compute, &kind);
    let mut p = p_compute;
    // Override rows 6-10 with failure-mode prices
    p[6] = 5.0; // T=0 ⇒ NonFinite
    p[7] = 5.0; // negative spot ⇒ NonFinite
    p[8] = 200.0; // price > S ⇒ AboveMaximum
    p[9] = 100.0; // price == S ⇒ AboveMaximum
                  // Row 5 (idx 5): σ=0.005 forward-priced ⇒ BelowVolMin

    let typed = implied_vol_typed_batch(&s, &k, &t, &r, &p, &kind);
    let f_direct = implied_vol(&s, &k, &t, &r, &p, &kind);

    println!("idx | typed_status         typed_value      | implied_vol(f64) | OK?");
    println!("----+--------------------------------------+------------------+----");
    for i in 0..10 {
        let sl = match typed[i].status {
            ImpliedVolStatus::Computed => "Computed".to_string(),
            ImpliedVolStatus::BelowVolMin { computed } => {
                format!("BelowVolMin(c={:.4e})", computed)
            }
            ImpliedVolStatus::AboveVolMax { computed } => {
                format!("AboveVolMax(c={:.4e})", computed)
            }
            other => format!("{:?}", other),
        };
        let ok = match typed[i].status {
            ImpliedVolStatus::Computed => !f_direct[i].is_nan(),
            _ => f_direct[i].is_nan(),
        };
        println!(
            "{:2}  | {:35} {:.6e}    | {:.6e}     | {}",
            i,
            sl,
            typed[i].value,
            f_direct[i],
            if ok { "PASS" } else { "FAIL" }
        );
    }
}
