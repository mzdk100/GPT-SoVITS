use std::env::var;

fn main() {
    if let Ok(v) = var("ORT_LIB_LOCATION") {
        println!("cargo:rustc-flags=-L mobile/{}", v);
    }
}
