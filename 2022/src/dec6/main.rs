use std::collections::HashSet;
use std::slice::Windows;

type Result<T> = std::result::Result<T, anyhow::Error>;

fn find_marker(s: &str) -> Option<usize> {
    s.as_bytes()
        .windows(4)
        .enumerate()
        .filter(|(_, s)| HashSet::<&u8>::from_iter(s.iter()).len() == 4)
        .next()
        .map(|(i, _)| i + 4)
}

fn main() -> Result<()> {
    let total: usize = std::io::stdin()
        .lines()
        .map(|line| find_marker(line.unwrap().trim()).expect("expected marker"))
        .sum();

    println!("Total chars: {}", total);

    Ok(())
}
