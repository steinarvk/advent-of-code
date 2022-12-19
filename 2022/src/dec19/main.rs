use regex::Regex;
use std::collections::HashMap;
use std::collections::VecDeque;
use std::io::Read;

type Result<T> = std::result::Result<T, anyhow::Error>;

const NUM_RESOURCES: usize = 4; // Ore Clay Obsidian Geode
const ORE: usize = 0;
const CLAY: usize = 1;
const OBSIDIAN: usize = 2;
const GEODES: usize = 3;

type ResourceVector = [i64; NUM_RESOURCES];
type RobotVector = [i64; NUM_RESOURCES];
type StateVector = [i64; NUM_RESOURCES * 2];
type Blueprint = [ResourceVector; NUM_RESOURCES];

fn resources_as_score(res: &ResourceVector) -> [i64; 4] {
    [res[GEODES], res[OBSIDIAN], res[CLAY], res[ORE]]
}

fn submul(a: &ResourceVector, b: &ResourceVector, n: i64) -> ResourceVector {
    let mut rv: ResourceVector = *a;
    for i in 0..NUM_RESOURCES {
        rv[i] -= n * b[i];
    }
    rv
}

fn add(a: &ResourceVector, b: &ResourceVector) -> ResourceVector {
    let mut rv: ResourceVector = *a;
    for i in 0..NUM_RESOURCES {
        rv[i] += b[i];
    }
    rv
}

fn can_afford(a: &ResourceVector, b: &ResourceVector) -> i64 {
    a.iter()
        .zip(b.iter())
        .filter_map(|(x, y)| match y {
            0 => None,
            y => Some(x / y),
        })
        .min()
        .unwrap()
}

#[derive(Clone, Debug)]
struct State {
    robots: [i64; NUM_RESOURCES],
    resources: ResourceVector,
    minutes_passed: i32,
    minutes_target: i32,
}

impl State {
    fn initial_state(target: i32) -> State {
        State {
            robots: [1, 0, 0, 0],
            resources: [1, 0, 0, 0],
            minutes_passed: 1,
            minutes_target: target,
        }
    }

    fn minutes_left(&self) -> i32 {
        self.minutes_target - self.minutes_passed
    }

    fn geodes(&self) -> i64 {
        self.resources[GEODES]
    }

    fn next_dream_state(&self, blueprint: &Blueprint) -> State {
        let i = can_afford(&self.resources, &blueprint[0]).min(1);
        let j = can_afford(&self.resources, &blueprint[1]).min(1);
        let k = can_afford(&self.resources, &blueprint[2]).min(1);
        let l = can_afford(&self.resources, &blueprint[3]).min(1);

        let new_robots: [i64; NUM_RESOURCES] = [i, j, k, l];
        State {
            robots: add(&new_robots, &self.robots),
            resources: add(&self.resources, &self.robots),
            minutes_passed: self.minutes_passed + 1,
            minutes_target: self.minutes_target,
        }
    }

    fn resources_at_end_if_inactive(&self) -> ResourceVector {
        let mut rv = self.resources;
        let mut n = self.minutes_passed;
        while n < self.minutes_target {
            for i in 0..NUM_RESOURCES {
                rv[i] += self.robots[i];
            }
            n += 1;
        }
        rv
    }

    fn next_greedy_state(&self, blueprint: &Blueprint) -> State {
        let mut best: Option<([i64; 4], State)> = None;
        self.next_states(blueprint).iter().for_each(|st| {
            let at_end = st.resources_at_end_if_inactive();
            let score = resources_as_score(&at_end);
            if let Some((best_score, _)) = best {
                if score > best_score {
                    best = Some((score, st.clone()));
                }
            } else {
                best = Some((score, st.clone()));
            }
        });
        best.unwrap().1
    }

    fn greedy_score(&self, blueprint: &Blueprint) -> [i64; 4] {
        let mut current = self.clone();
        while current.minutes_passed < self.minutes_target {
            current = current.next_greedy_state(blueprint);
        }
        resources_as_score(&current.resources)
    }

    fn upper_bound_geodes(&self, blueprint: &Blueprint) -> i64 {
        let mut current = self.clone();
        while current.minutes_passed < self.minutes_target {
            current = current.next_dream_state(blueprint);
        }
        current.geodes()
    }

    fn next_states(&self, blueprint: &Blueprint) -> Vec<State> {
        let remaining_resources = self.resources;
        let mut rv = Vec::new();

        for i in 0..=can_afford(&remaining_resources, &blueprint[0]).min(1) {
            let remaining_resources = submul(&remaining_resources, &blueprint[0], i);
            for j in 0..=can_afford(&remaining_resources, &blueprint[1]).min(1) {
                let remaining_resources = submul(&remaining_resources, &blueprint[1], j);
                for k in 0..=can_afford(&remaining_resources, &blueprint[2]).min(1) {
                    let remaining_resources = submul(&remaining_resources, &blueprint[2], k);
                    for l in 0..=can_afford(&remaining_resources, &blueprint[3]).min(1) {
                        let remaining_resources = submul(&remaining_resources, &blueprint[3], l);

                        let new_robots: [i64; NUM_RESOURCES] = [i, j, k, l];

                        if new_robots.iter().sum::<i64>() > 1 {
                            continue;
                        }

                        rv.push(State {
                            robots: add(&new_robots, &self.robots),
                            resources: add(&remaining_resources, &self.robots),
                            minutes_passed: self.minutes_passed + 1,
                            minutes_target: self.minutes_target,
                        });
                    }
                }
            }
        }

        rv
    }
}

struct Memo {
    memo: HashMap<(ResourceVector, RobotVector, i32), i64>,
    iter: i64,
    hits: i64,
    misses: i64,
    max_geodes_ever_seen: i64,
}

impl Memo {
    fn new() -> Memo {
        Memo {
            memo: HashMap::new(),
            iter: 0,
            hits: 0,
            misses: 0,
            max_geodes_ever_seen: 0,
        }
    }

    fn evaluate(&mut self, blueprint: &Blueprint, state: &State) -> i64 {
        self.iter += 1;

        let geodes = state.resources[GEODES];
        self.max_geodes_ever_seen = self.max_geodes_ever_seen.max(geodes);

        let upper_bound = state.upper_bound_geodes(blueprint);

        if self.iter % 10_000_000 == 0 {
            let greedy_score = state.greedy_score(blueprint);
            eprintln!(
                "iteration {}: {:?} hits {} misses {} memosz {} geodes {} upper-bound-geodes {} greedy-score {:?} max-geodes {}",
                self.iter,
                state,
                self.hits,
                self.misses,
                self.memo.len(),
                geodes,
                upper_bound,
                greedy_score,
                self.max_geodes_ever_seen,
            );
            self.hits = 0;
            self.misses = 0;
        }

        if upper_bound < self.max_geodes_ever_seen {
            return 0;
        }

        let minutes_left = state.minutes_left();

        if minutes_left == 2 {
            let number_of_affordable_geode_robots = can_afford(&state.resources, &blueprint[3]);
            let geodes = geodes + 2 * state.robots[GEODES] + number_of_affordable_geode_robots;
            self.max_geodes_ever_seen = self.max_geodes_ever_seen.max(geodes);
            return geodes;
        }

        if minutes_left == 1 {
            let geodes = geodes + state.robots[GEODES];
            self.max_geodes_ever_seen = self.max_geodes_ever_seen.max(geodes);
            return geodes;
        }

        if minutes_left == 0 {
            return geodes;
        }

        let key = (state.resources, state.robots, minutes_left);

        if let Some(result) = self.memo.get(&key) {
            self.hits += 1;
            return *result;
        }
        self.misses += 1;

        let result = state
            .next_states(&blueprint)
            .iter()
            .map(|x| self.evaluate(blueprint, x))
            .max()
            .unwrap();

        // eprintln!("result at {:?} time_left={} : {}", state, time_left, result);

        self.memo.insert(key, result);

        result
    }

    fn evaluate_blueprint(&mut self, blueprint: &Blueprint, target: i32) -> i64 {
        self.evaluate(blueprint, &State::initial_state(target))
    }
}

fn parse_scenario(s: String) -> Result<Vec<Blueprint>> {
    let re = Regex::new(
        r"Blueprint [0-9]+: Each ore robot costs (?P<ore_ore>[0-9]+) ore. Each clay robot costs (?P<clay_ore>[0-9]+) ore. Each obsidian robot costs (?P<obs_ore>[0-9]+) ore and (?P<obs_clay>[0-9]+) clay. Each geode robot costs (?P<geo_ore>[0-9]+) ore and (?P<geo_obs>[0-9]+) obsidian.",
    )?;

    let lines: Vec<String> = s.trim().split('\n').map(|line| line.to_string()).collect();

    Ok(lines
        .iter()
        .map(|line| {
            let captures = re.captures(line.trim()).unwrap();
            [
                [captures["ore_ore"].parse().unwrap(), 0, 0, 0],
                [captures["clay_ore"].parse().unwrap(), 0, 0, 0],
                [
                    captures["obs_ore"].parse().unwrap(),
                    captures["obs_clay"].parse().unwrap(),
                    0,
                    0,
                ],
                [
                    captures["geo_ore"].parse().unwrap(),
                    0,
                    captures["geo_obs"].parse().unwrap(),
                    0,
                ],
            ]
        })
        .collect())
}

fn main() -> Result<()> {
    let mut s = String::new();
    std::io::stdin().read_to_string(&mut s)?;

    let blueprints: Vec<Blueprint> = parse_scenario(s)?;

    for b in blueprints.iter() {
        println!("{:?}", b);
    }

    let mut sum_of_qualities = 0;
    let mut product_of_geodes = 1;
    let mut total_iterations = 0;

    for (i, blueprint) in blueprints.iter().enumerate() {
        let mut memo = Memo::new();
        let id = (i + 1) as i64;
        let geodes_24 = memo.evaluate_blueprint(&blueprint, 24);
        let quality = id * geodes_24;
        sum_of_qualities += quality;

        println!(
            "Blueprint {} (after {} iterations): {} after 24",
            i + 1,
            memo.iter,
            geodes_24,
        );

        if i < 3 {
            let geodes_32 = memo.evaluate_blueprint(&blueprint, 32);
            println!(
                "Blueprint {} (after {} iterations): {} after 32",
                i + 1,
                memo.iter,
                geodes_32,
            );
            product_of_geodes *= geodes_32;
        }

        total_iterations += memo.iter;
    }

    println!("Total iterations used: {}", total_iterations);

    println!("Answer to part A: {}", sum_of_qualities);
    println!("Answer to part B: {}", product_of_geodes);
    Ok(())
}
