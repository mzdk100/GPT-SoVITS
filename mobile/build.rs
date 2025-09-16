use cargo_emit::rustc_flags;
use std::env::var;

fn main() {
    rustc_flags!(format!("-L mobile/{}", var("ORT_LIB_LOCATION").unwrap()));
}