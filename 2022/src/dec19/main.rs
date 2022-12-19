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

fn sub(a: &ResourceVector, b: &ResourceVector) -> ResourceVector {
    let mut rv: ResourceVector = *a;
    for i in 0..NUM_RESOURCES {
        rv[i] -= b[i];
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

enum DominationResult {
    Left,
    Right,
    Equal,
    Neither,
}

fn dominates(left: &StateVector, right: &StateVector) -> DominationResult {
    let mut left_bigger = false;
    let mut right_bigger = false;

    for i in 0..(2 * NUM_RESOURCES) {
        if left[i] > right[i] {
            left_bigger = true;
        } else if left[i] < right[i] {
            right_bigger = true;
        }
    }

    match (left_bigger, right_bigger) {
        (true, true) => DominationResult::Neither,
        (true, false) => DominationResult::Left,
        (false, true) => DominationResult::Right,
        (false, false) => DominationResult::Equal,
    }
}

#[derive(Clone, Debug)]
struct State {
    robots: [i64; NUM_RESOURCES],
    resources: ResourceVector,
    minutes_passed: i32,
    trail: String,
}

impl State {
    fn initial_state() -> State {
        State {
            robots: [1, 0, 0, 0],
            resources: [1, 0, 0, 0],
            minutes_passed: 1,
            trail: "".to_string(),
        }
    }

    fn format(&self) -> String {
        format!(
            "resources={:?} robots={:?} minutes_passed={}",
            self.resources, self.robots, self.minutes_passed
        )
    }

    fn next_state_by_robots(
        &self,
        blueprint: &Blueprint,
        new_robots: &RobotVector,
    ) -> Result<State> {
        let desired = add(&self.robots, new_robots);
        for x in self.next_states(&blueprint) {
            if desired == x.robots {
                return Ok(x);
            }
        }
        anyhow::bail!("no such option: {:?}", new_robots);
    }

    fn geodes(&self) -> i64 {
        self.resources[GEODES]
    }

    fn next_dream_state(&self, blueprint: &Blueprint) -> State {
        let i = can_afford(&self.resources, &blueprint[0]);
        let j = can_afford(&self.resources, &blueprint[1]);
        let k = can_afford(&self.resources, &blueprint[2]);
        let l = can_afford(&self.resources, &blueprint[3]);

        let new_robots: [i64; NUM_RESOURCES] = [i, j, k, l];
        State {
            robots: add(&new_robots, &self.robots),
            resources: add(&self.resources, &self.robots),
            minutes_passed: self.minutes_passed + 1,
            trail: "<dream>".to_string(),
        }
    }

    fn resources_at_end_if_inactive(&self) -> ResourceVector {
        let mut rv = self.resources;
        let mut n = self.minutes_passed;
        while n < 32 {
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
        while current.minutes_passed < 32 {
            current = current.next_greedy_state(blueprint);
        }
        resources_as_score(&current.resources)
    }

    fn upper_bound_geodes(&self, blueprint: &Blueprint) -> i64 {
        let mut current = self.clone();
        while current.minutes_passed < 32 {
            current = current.next_dream_state(blueprint);
        }
        current.geodes()
    }

    fn next_states(&self, blueprint: &Blueprint) -> Vec<State> {
        /*
        // Collect income
        let mut new_resources = self.resources;

        for i in 0..NUM_RESOURCES {
            new_resources[i] += self.robots[i];
        }
        */

        let remaining_resources = self.resources;
        let mut rv = Vec::new();

        // TODO some of these are likely not possibly optimal
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
                            trail: "".to_string(),
                            // trail: format!("{}\n{}", self.trail, self.format()),
                        });
                    }
                }
            }
        }

        //eprintln!("hmm?");
        let mut rv: Vec<([i64; 4], State)> = rv
            .iter()
            .map(|st| {
                (
                    resources_as_score(&st.resources_at_end_if_inactive()),
                    st.clone(),
                )
            })
            .collect();
        //eprintln!("sorting {}", rv.len());
        rv.sort_by(|(a_score, _), (b_score, _)| a_score.cmp(b_score).reverse());
        //eprintln!("sorted {}", rv.len());
        //for (score, x) in rv.iter() {
        //    eprintln!("hmm {:?} {:?}", score, x);
        // }
        let rv: Vec<State> = rv.iter().map(|x| x.1.clone()).collect();
        //eprintln!("mapped {}", rv.len());

        /*
        let mut rng = rand::thread_rng();
        rv.shuffle(&mut rng);
        */

        // eprintln!("Generated {} options (pruned {})", rv.len(), prunecount);

        // eprintln!("generated {} successor states", rv.len());

        rv
    }
}

struct ChampionSet {
    champions: VecDeque<StateVector>,
}

impl ChampionSet {
    fn new() -> ChampionSet {
        ChampionSet {
            champions: VecDeque::new(),
        }
    }

    fn update(&mut self, state: &State) -> bool {
        let state: StateVector = [
            state.resources[0],
            state.resources[1],
            state.resources[2],
            state.resources[3],
            state.robots[0],
            state.robots[1],
            state.robots[2],
            state.robots[3],
        ];

        let mut already_dominated = false;
        let mut i = 0;

        while i < self.champions.len() {
            match dominates(&self.champions[i], &state) {
                DominationResult::Equal => {
                    return false;
                }
                DominationResult::Left => {
                    return false;
                }
                DominationResult::Right => {
                    if already_dominated {
                        // We already inserted this, no need to insert it again.
                        if (i + 1) != self.champions.len() {
                            self.champions[i] = *self.champions.back().unwrap();
                        }
                        self.champions.pop_back();
                        continue;
                    } else {
                        self.champions[i] = state;
                    }

                    already_dominated = true;
                }
                DominationResult::Neither => (),
            }

            i += 1;
        }

        if already_dominated {
            return true;
        }

        self.champions.push_front(state);

        true
    }
}

struct Memo {
    memo: HashMap<(ResourceVector, RobotVector, i32), i64>,
    champions: Vec<ChampionSet>,
    iter: i64,
    hits: i64,
    misses: i64,
    max_geodes_ever_seen: i64,
}

impl Memo {
    fn new() -> Memo {
        let champions = (0..=32).map(|_| ChampionSet::new()).collect();
        Memo {
            memo: HashMap::new(),
            champions,
            iter: 0,
            hits: 0,
            misses: 0,
            max_geodes_ever_seen: 0,
        }
    }

    fn evaluate(&mut self, blueprint: &Blueprint, state: &State) -> i64 {
        self.iter += 1;

        let geodes = state.resources[GEODES];
        /*
        if geodes == 13 {
            eprintln!("wtf? did not expect 13 geodes {:?}", state);
            eprintln!("trail:\n{}\n{}", state.trail, state.format());
            panic!("oops");
        }
        */
        self.max_geodes_ever_seen = self.max_geodes_ever_seen.max(geodes);

        let upper_bound = state.upper_bound_geodes(blueprint);

        if self.iter % 1_000_000 == 0 {
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

        assert!(state.minutes_passed <= 32);

        /*
        if state.minutes_passed == 22 {
            let number_of_affordable_geode_robots = can_afford(&state.resources, &blueprint[3]);
            let geodes = geodes + 2 * state.robots[GEODES] + number_of_affordable_geode_robots;
            if geodes == 13 {
                eprintln!("wtf? did not expect 13 geodes {:?}", state);
                panic!("oops");
            }
            self.max_geodes_ever_seen = self.max_geodes_ever_seen.max(geodes);
            return geodes;
        }
        */

        /*
        if state.minutes_passed == 23 {
            let geodes = geodes + state.robots[GEODES];
            self.max_geodes_ever_seen = self.max_geodes_ever_seen.max(geodes);
            if geodes == 13 {
                eprintln!("wtf? did not expect 13 geodes {:?}", state);
                panic!("oops");
            }
            return geodes;
        }
        */

        if state.minutes_passed == 32 {
            return geodes;
        }

        /*
        if !self.champions[state.minutes_passed as usize].update(state) {
            // eprintln!("discarding {:?} -- already saw better or equal", state);
            return 0;
        }
        */

        let key = (state.resources, state.robots, state.minutes_passed);

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

    fn evaluate_blueprint(&mut self, blueprint: &Blueprint) -> i64 {
        self.evaluate(blueprint, &State::initial_state())
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
    let blueprint: Blueprint = [[4, 0, 0, 0], [2, 0, 0, 0], [3, 14, 0, 0], [2, 0, 7, 0]];

    let mut state = State::initial_state();

    let xs: Vec<RobotVector> = vec![
        [0, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ];

    for new_robots in xs {
        state = state.next_state_by_robots(&blueprint, &new_robots)?;
        println!(
            "{:?} [upper bound: {}]",
            state,
            state.upper_bound_geodes(&blueprint)
        );
    }

    for (i, nb) in state.next_states(&blueprint).iter().enumerate() {
        println!("  [{}] {:?}", i, nb);
    }

    let mut s = String::new();
    std::io::stdin().read_to_string(&mut s)?;

    let blueprints: Vec<Blueprint> = parse_scenario(s)?;
    let blueprints = &blueprints[0..3];
    assert!(blueprints.len() == 3);

    for b in blueprints.iter() {
        println!("{:?}", b);
    }

    let mut score = 0;
    let mut total_product = 1;
    for (i, blueprint) in blueprints.iter().enumerate() {
        let mut memo = Memo::new();
        let id = (i + 1) as i64;
        let geodes = memo.evaluate_blueprint(&blueprint);
        let quality = id * geodes;
        score += quality;
        total_product *= geodes;
        println!(
            "Blueprint {} (after {} iterations): {}",
            i + 1,
            memo.iter,
            geodes,
        );
    }

    println!("total score: {}", score);

    println!("product score: {}", total_product);
    Ok(())
}
