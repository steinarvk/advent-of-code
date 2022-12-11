type Result<T> = std::result::Result<T, anyhow::Error>;

use anyhow::bail;
use regex::Regex;
use std::collections::VecDeque;
use std::io::Read;

#[derive(Debug)]
enum Atom {
    OldValue,
    Literal(i32),
}

#[derive(Debug)]
enum Expr {
    Add(Atom, Atom),
    Mul(Atom, Atom),
}

impl Atom {
    fn eval(&self, old_value: i32) -> i32 {
        match *self {
            Atom::OldValue => old_value,
            Atom::Literal(x) => x,
        }
    }
}

impl Expr {
    fn eval(&self, old_value: i32) -> i32 {
        match self {
            Expr::Add(a, b) => a.eval(old_value) + b.eval(old_value),
            Expr::Mul(a, b) => a.eval(old_value) * b.eval(old_value),
        }
    }
}
fn parse_atom(s: &str) -> Result<Atom> {
    match s {
        "old" => Ok(Atom::OldValue),
        _ => Ok(Atom::Literal(s.parse()?)),
    }
}

fn parse_expr(s: &str) -> Result<Expr> {
    let tokens: Vec<&str> = s.trim().split_whitespace().collect();

    assert!(tokens.len() == 3);

    let left = parse_atom(tokens[0])?;
    let right = parse_atom(tokens[2])?;

    match tokens[1] {
        "+" => Ok(Expr::Add(left, right)),
        "*" => Ok(Expr::Mul(left, right)),
        _ => bail!("bad operator: {}", tokens[1]),
    }
}

#[derive(Debug)]
struct Monkey {
    index: usize,
    inspect_count: i32,
    expr: Expr,
    items: VecDeque<i32>,
    divisor: i32,
    target_if_divisible: usize,
    target_if_not_divisible: usize,
}

fn run_round(monkeys: &mut Vec<Monkey>) -> Result<()> {
    let mut item_queue: Vec<VecDeque<i32>> = monkeys.iter().map(|_| VecDeque::new()).collect();

    for monkey in monkeys.iter_mut() {
        println!("Monkey {}'s turn.", monkey.index);

        println!("Incoming items: {:?}", item_queue[monkey.index]);
        monkey.items.append(&mut item_queue[monkey.index]);
        item_queue[monkey.index] = VecDeque::new();

        while !monkey.items.is_empty() {
            monkey.inspect_count += 1;

            let worry = monkey.items.pop_front().unwrap();
            println!("Monkey inspecting new item with worry level {}.", worry);
            let worry = monkey.expr.eval(worry);
            println!("Worry level adjusted to {}.", worry);
            let worry = worry / 3;
            println!("Worry level divided by 3 to {}.", worry);

            let divisible = worry % monkey.divisor == 0;
            println!("Divisible by {}? {}", monkey.divisor, divisible);

            let target_monkey = if divisible {
                monkey.target_if_divisible
            } else {
                monkey.target_if_not_divisible
            };
            println!(
                "Throwing item with worry level {} to monkey {}",
                worry, target_monkey
            );

            item_queue[target_monkey].push_back(worry);
        }
    }

    for monkey in monkeys.iter_mut() {
        monkey.items.append(&mut item_queue[monkey.index]);
    }

    Ok(())
}

fn main() -> Result<()> {
    let mut buffer = String::new();

    std::io::stdin().read_to_string(&mut buffer)?;

    let re = Regex::new(r"Monkey (?P<index>[0-9]+):.*\n.*Starting items: (?P<items>.*)\n.*Operation: new = (?P<expr>.*)\n.*Test: divisible by (?P<divisor>[0-9]+)\n.*If true: throw to monkey (?P<yes>[0-9]+).*\n.*If false: throw to monkey (?P<no>[0-9]+)").unwrap();

    let mut monkeys: Vec<Monkey> = buffer
        .split("\n\n")
        .map(|text| {
            let captures = re.captures(text.trim()).unwrap();

            Monkey {
                index: captures["index"].parse().unwrap(),
                inspect_count: 0,
                expr: parse_expr(&captures["expr"]).unwrap(),
                items: VecDeque::from_iter(
                    captures["items"]
                        .split(",")
                        .map(|x| x.trim().parse::<i32>().unwrap()),
                ),
                divisor: captures["divisor"].parse().unwrap(),
                target_if_divisible: captures["yes"].parse().unwrap(),
                target_if_not_divisible: captures["no"].parse().unwrap(),
            }
        })
        .collect();

    for _ in 0..20 {
        run_round(&mut monkeys)?;
    }

    let mut counts: Vec<i32> = monkeys.iter().map(|x| x.inspect_count).collect();
    counts.sort();

    println!("All counts: {:?}", counts);

    let most_active_counts: Vec<i32> = counts.iter().rev().take(2).map(|x| *x).collect();

    println!("Most active counts: {:?}", most_active_counts);

    println!("Product: {}", most_active_counts.iter().product::<i32>());

    Ok(())
}
