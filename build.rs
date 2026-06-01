fn main() {
    // Sleef is only used by the AVX-512 extern path in src/norm.rs (gated on
    // target_feature=avx512f). Match the gate here so non-AVX-512 builds
    // don't try to link libsleef.
    let features = std::env::var("CARGO_CFG_TARGET_FEATURE").unwrap_or_default();
    if features.split(',').any(|f| f == "avx512f") {
        let home = std::env::var("HOME").expect("HOME not set");
        println!("cargo:rustc-link-search=native={}/.local/lib", home);
        println!("cargo:rustc-link-lib=static=sleef");
    }
    println!("cargo:rerun-if-changed=build.rs");
}
