fn main() {
    let home = std::env::var("HOME").expect("HOME not set");
    println!("cargo:rustc-link-search=native={}/.local/lib", home);
    println!("cargo:rustc-link-lib=static=sleef");
    println!("cargo:rerun-if-changed=build.rs");
}
