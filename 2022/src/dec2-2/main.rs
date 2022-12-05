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

fn symbol_that_wins_against(x: Symbol) -> Symbol {
    match x {
        Symbol::Rock => Symbol::Paper,
        Symbol::Paper => Symbol::Scissors,
        Symbol::Scissors => Symbol::Rock,
    }
}

fn symbol_that_loses_against(x: Symbol) -> Symbol {
    match x {
        Symbol::Paper => Symbol::Rock,
        Symbol::Scissors => Symbol::Paper,
        Symbol::Rock => Symbol::Scissors,
    }
}

fn outcome_for_me(me: Symbol, them: Symbol) -> Outcome {
    if symbol_that_wins_against(me) == them {
        Outcome::Loss
    } else if symbol_that_wins_against(them) == me {
        Outcome::Win
    } else {
        Outcome::Tie
    }
}

fn symbol_to_choose(opponent: Symbol, desired_outcome: Outcome) -> Symbol {
    match desired_outcome {
        Outcome::Tie => opponent,
        Outcome::Win => symbol_that_wins_against(opponent),
        Outcome::Loss => symbol_that_loses_against(opponent),
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
        let (them, desired_outcome) = line.split_once(" ").unwrap();
        let them = match them {
            "A" => Symbol::Rock,
            "B" => Symbol::Paper,
            "C" => Symbol::Scissors,
            _ => panic!("bad symbol for them"),
        };

        let desired_outcome = match desired_outcome {
            "X" => Outcome::Loss,
            "Y" => Outcome::Tie,
            "Z" => Outcome::Win,
            _ => panic!("bad symbol for desired outcome"),
        };

        let me = symbol_to_choose(them, desired_outcome);

        running_total += round_score(me, them)
    }

    println!("Total score: {}", running_total);

    Ok(())
}
