type Result<T> = std::result::Result<T, anyhow::Error>;

#[derive(PartialEq, Copy, Clone)]
enum Symbol {
    Rock,
    Paper,
    Scissors,
}

enum Outcome {
    Loss,
    Tie,
    Win,
}

fn outcome_for_me(me: Symbol, them: Symbol) -> Outcome {
    match (me, them) {
        (Symbol::Rock, Symbol::Scissors) => Outcome::Win,
        (Symbol::Rock, Symbol::Paper) => Outcome::Loss,
        (Symbol::Paper, Symbol::Rock) => Outcome::Win,
        (Symbol::Paper, Symbol::Scissors) => Outcome::Loss,
        (Symbol::Scissors, Symbol::Paper) => Outcome::Win,
        (Symbol::Scissors, Symbol::Rock) => Outcome::Loss,
        _ => {
            assert!(me == them);
            Outcome::Tie
        }
    }
}

fn symbol_score(x: Symbol) -> i32 {
    match x {
        Symbol::Rock => 1,
        Symbol::Paper => 2,
        Symbol::Scissors => 3,
    }
}

fn round_score(me: Symbol, them: Symbol) -> i32 {
    symbol_score(me)
        + match outcome_for_me(me, them) {
            Outcome::Win => 6,
            Outcome::Tie => 3,
            Outcome::Loss => 0,
        }
}

fn main() -> Result<()> {
    let mut running_total: i32 = 0;

    for line in std::io::stdin().lines() {
        let line = line?;
        let (them, me) = line.split_once(" ").unwrap();
        let them = match them {
            "A" => Symbol::Rock,
            "B" => Symbol::Paper,
            "C" => Symbol::Scissors,
            _ => panic!("bad symbol for them"),
        };

        let me = match me {
            "X" => Symbol::Rock,
            "Y" => Symbol::Paper,
            "Z" => Symbol::Scissors,
            _ => panic!("bad symbol for me"),
        };

        running_total += round_score(me, them)
    }

    println!("Total score: {}", running_total);

    Ok(())
}
