use regex::Regex;
use std::collections::HashMap;
use std::collections::HashSet;
use std::collections::VecDeque;

type Result<T> = std::result::Result<T, anyhow::Error>;

const MAX_NODES: usize = 51;
const MAX_NUMBER_OF_MINUTES: usize = 30;

struct Node {
    paths: Vec<usize>,
    rate: i64,
}

struct ScoreMemo {
    memo: HashMap<((usize, usize), [bool; MAX_NODES]), [Option<i64>; MAX_NUMBER_OF_MINUTES]>,
}

impl ScoreMemo {
    fn new() -> ScoreMemo {
        ScoreMemo {
            memo: HashMap::new(),
        }
    }

    fn update(&mut self, state: &State) -> bool {
        let (p0, p1) = state.positions;
        let canonical_pos = (p0.min(p1), p0.max(p1));
        let key = (canonical_pos, state.on);

        let value: &mut [Option<i64>; MAX_NUMBER_OF_MINUTES] = self
            .memo
            .entry(key)
            .or_insert_with(|| [None; MAX_NUMBER_OF_MINUTES]);

        for t in 0..=state.time {
            if let Some(previous_score) = value[t] {
                if previous_score >= state.score {
                    return false;
                }
            }
        }

        value[state.time] = Some(state.score);
        true
    }
}

#[derive(Debug)]
struct State {
    time: usize,
    positions: (usize, usize),
    on: [bool; MAX_NODES],
    score: i64,
    use_elephant: bool,
}

impl State {
    fn initial_state(starting_position: usize, use_elephant: bool) -> State {
        State {
            time: if use_elephant { 4 } else { 0 },
            positions: (starting_position, starting_position),
            on: [false; MAX_NODES],
            score: 0,
            use_elephant,
        }
    }

    fn move_is_pointless(
        &self,
        distances: &[Vec<Option<i32>>],
        before: usize,
        after: usize,
        time_left: usize,
    ) -> bool {
        // This optimization doesn't actually seem to matter much.
        let conclusion = self
            .on
            .iter()
            .enumerate()
            .filter_map(|(i, v)| {
                if !v && i < distances.len() {
                    Some(i)
                } else {
                    None
                }
            })
            .filter(|i| {
                let distance_before = distances[before][*i].unwrap();
                let distance_after = distances[after][*i].unwrap();
                let progress = distance_before - distance_after;
                let max_steps_left = time_left - 1; // Also need 1min to toggle.
                let has_enough_time = distance_before <= max_steps_left.try_into().unwrap();
                progress > 0 && has_enough_time
            })
            .next()
            .is_none();
        conclusion
    }

    fn minutes_left(&self) -> usize {
        MAX_NUMBER_OF_MINUTES - (self.time + 1)
    }

    fn upper_bound_for_score(&self, nodes: &[Node], distance_matrix: &[Vec<Option<i32>>]) -> i64 {
        let minutes_left: i64 = self.minutes_left() as i64;

        let upper_bound_extra_score = self
            .on
            .iter()
            .enumerate()
            .filter_map(|(i, on)| {
                if *on || i >= nodes.len() || nodes[i].rate == 0 {
                    return None;
                }

                let distance = distance_matrix[self.positions.0][i]
                    .unwrap()
                    .min(distance_matrix[self.positions.1][i].unwrap());

                let minutes_flowing = minutes_left - (distance as i64);

                if minutes_flowing <= 0 {
                    return None;
                }

                Some(nodes[i].rate * minutes_flowing)
            })
            .sum::<i64>();
        assert!(upper_bound_extra_score >= 0);

        upper_bound_extra_score + self.score
    }

    fn next_states(&self, nodes: &[Node], distance_matrix: &[Vec<Option<i32>>]) -> Vec<State> {
        assert!(nodes.len() <= MAX_NODES);
        assert!(self.time < MAX_NUMBER_OF_MINUTES);

        let my_node = &nodes[self.positions.0];
        let its_node = &nodes[self.positions.1];

        let minutes_left_to_flow_after_current_turn = MAX_NUMBER_OF_MINUTES - (self.time + 1);

        let mut rv: Vec<State> = Vec::new();

        let mut my_possible_actions: Vec<(usize, i64)> = Vec::new();
        let mut its_possible_actions: Vec<(usize, i64)> = Vec::new();

        for my_next_node in &my_node.paths {
            let pointless = self.move_is_pointless(
                distance_matrix,
                self.positions.0,
                *my_next_node,
                minutes_left_to_flow_after_current_turn,
            );
            if !pointless {
                my_possible_actions.push((*my_next_node, 0));
            }
        }

        for its_next_node in &its_node.paths {
            let pointless = self.move_is_pointless(
                distance_matrix,
                self.positions.1,
                *its_next_node,
                minutes_left_to_flow_after_current_turn,
            );

            if !pointless {
                its_possible_actions.push((*its_next_node, 0));
            }
        }

        if my_possible_actions.is_empty() {
            my_possible_actions.push((self.positions.0, 0));
        }

        if its_possible_actions.is_empty() {
            its_possible_actions.push((self.positions.1, 0));
        }

        if my_node.rate > 0 && !self.on[self.positions.0] {
            let extra_score = my_node.rate * (minutes_left_to_flow_after_current_turn as i64);
            my_possible_actions.push((self.positions.0, extra_score));
        }

        if its_node.rate > 0 && !self.on[self.positions.1] {
            let extra_score = its_node.rate * (minutes_left_to_flow_after_current_turn as i64);
            its_possible_actions.push((self.positions.1, extra_score));
        }

        if !self.use_elephant {
            its_possible_actions = vec![(0, 0)];
        }

        let mut position_pairs: HashSet<(usize, usize)> = HashSet::new();

        for (my_next_pos, _) in &my_possible_actions {
            for (its_next_pos, _) in &its_possible_actions {
                let p0: usize = *my_next_pos;
                let p1: usize = *its_next_pos;

                position_pairs.insert((p0, p1));
            }
        }

        for (my_next_pos, my_extra_score) in &my_possible_actions {
            for (its_next_pos, its_extra_score) in &its_possible_actions {
                let p0: usize = *my_next_pos;
                let p1: usize = *its_next_pos;

                if p1 > p0 && position_pairs.contains(&(p1, p0)) {
                    continue;
                }

                if p0 == p1 && *my_extra_score > 0 && *its_extra_score > 0 {
                    // We can't both flip the same valve.
                    continue;
                }

                let mut new_on = self.on;

                if *my_extra_score > 0 {
                    new_on[self.positions.0] = true;
                }

                if *its_extra_score > 0 {
                    new_on[self.positions.1] = true;
                }

                rv.push(State {
                    time: self.time + 1,
                    positions: (p0, p1),
                    on: new_on,
                    score: self.score + my_extra_score + its_extra_score,
                    use_elephant: self.use_elephant,
                });
            }
        }

        rv
    }
}

fn solve(nodes: &[Node], starting_position: usize, use_elephant: bool) -> Result<i64> {
    let distances = distance_matrix(nodes);

    let mut iteration_count: u64 = 0;

    let mut max_observed_score: i64 = 0;

    let mut memo = ScoreMemo::new();

    let mut q = VecDeque::new();
    q.push_back(State::initial_state(starting_position, use_elephant));

    let starting_time = q[0].time;

    while !q.is_empty() {
        iteration_count += 1;

        let state = q.pop_front().unwrap();

        max_observed_score = state.score.max(max_observed_score);

        let score_upper_bound = state.upper_bound_for_score(nodes, &distances);
        if score_upper_bound < max_observed_score {
            continue;
        }

        if iteration_count % 100_000 == 0 {
            eprintln!(
                "Iteration {}; queue size {}, time {}, cur score {}, upper-bound: {}, max score {}; oncount {}",
                iteration_count,
                q.len(),
                state.time,
                state.score,
                score_upper_bound,
                max_observed_score,
                state.on.iter().filter(|x| **x).count(),
            );
        }

        if state.time >= (MAX_NUMBER_OF_MINUTES - 1) {
            continue;
        }

        for new_state in state.next_states(nodes, &distances) {
            if memo.update(&new_state) {
                q.push_back(new_state);
            }
        }
    }

    eprintln!("Done after {} iterations.", iteration_count);
    eprintln!(
        "Max score (with elephant? {}) after {} minutes: {}",
        use_elephant,
        MAX_NUMBER_OF_MINUTES - starting_time,
        max_observed_score
    );

    Ok(max_observed_score)
}

fn distances_from(nodes: &[Node], start: usize) -> Vec<Option<i32>> {
    let n = nodes.len();
    let mut rv: Vec<Option<i32>> = std::iter::repeat(None).take(n).collect();
    let mut visited: Vec<bool> = std::iter::repeat(false).take(n).collect();
    let mut q = VecDeque::new();

    q.push_back((0, start));
    visited[start] = true;

    while let Some((distance, index)) = q.pop_front() {
        rv[index] = match rv[index] {
            Some(old_distance) => Some(old_distance.min(distance)),
            None => Some(distance),
        };
        visited[index] = true;

        for next_index in &nodes[index].paths {
            if !visited[*next_index] {
                q.push_back((distance + 1, *next_index));
            }
        }
    }

    rv
}

fn distance_matrix(nodes: &[Node]) -> Vec<Vec<Option<i32>>> {
    (0..nodes.len()).map(|i| distances_from(nodes, i)).collect()
}

fn main() -> Result<()> {
    let re = Regex::new(
        r"Valve (?P<name>[A-Z]+) has flow rate=(?P<rate>[0-9]+); tunnels? leads? to valves? (?P<out>[A-Z, ]+)",
    )?;

    let mut name_to_index: HashMap<String, usize> = HashMap::new();
    let mut nodes: Vec<(i64, Vec<String>)> = Vec::new();

    for line in std::io::stdin().lines() {
        let line = line?;
        let captures = re.captures(line.trim()).unwrap();
        let name: String = captures["name"].to_string();
        let rate: i64 = captures["rate"].parse().unwrap();
        let out: Vec<String> = captures["out"]
            .split(',')
            .map(|x| x.trim().to_string())
            .collect();
        let index: usize = nodes.len();
        name_to_index.insert(name, index);
        nodes.push((rate, out));
    }

    let nodes: Vec<Node> = nodes
        .iter()
        .map(|(rate, out_names)| {
            let rate = *rate;
            let paths: Vec<usize> = out_names
                .iter()
                .map(|name| *name_to_index.get(name).unwrap())
                .collect();
            Node { paths, rate }
        })
        .collect();

    let starting_position: usize = name_to_index["AA"];

    println!("AA = {}", starting_position);

    println!(
        "Answer to first question: {}",
        solve(&nodes, starting_position, false)?
    );
    println!(
        "Answer to second question: {}",
        solve(&nodes, starting_position, true)?
    );

    Ok(())
}
