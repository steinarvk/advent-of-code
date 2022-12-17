type Result<T> = std::result::Result<T, anyhow::Error>;

use std::collections::HashMap;
use std::collections::VecDeque;

use std::io::Read;

struct Map<T> {
    number_of_rows: i32,
    number_of_columns: i32,
    row_data: Vec<Vec<T>>,
}

impl Map<char> {
    fn from_stdin() -> Result<Map<char>> {
        let mut buffer = String::new();
        std::io::stdin().read_to_string(&mut buffer)?;
        Map::from_string(&buffer)
    }

    fn from_string(s: &str) -> Result<Map<char>> {
        let rows: Vec<&str> = s.trim().split('\n').collect();
        let number_of_rows = rows.len();
        let number_of_cols = rows[0].len();

        if !(rows.iter().all(|s| s.trim().len() == number_of_cols)) {
            anyhow::bail!("rows are not all the same length");
        }

        let row_data = rows
            .iter()
            .map(|row| row.trim().chars().collect())
            .collect();

        Ok(Map {
            number_of_rows: number_of_rows.try_into().unwrap(),
            number_of_columns: number_of_cols.try_into().unwrap(),
            row_data,
        })
    }
}

impl<T> Map<T>
where
    T: Clone,
{
    fn new(number_of_columns: i32, number_of_rows: i32, starting_value: &T) -> Map<T> {
        let row_data = (0..number_of_rows)
            .map(|_| {
                (0..number_of_columns)
                    .map(|_| starting_value.clone())
                    .collect()
            })
            .collect();
        Map {
            number_of_rows,
            number_of_columns,
            row_data,
        }
    }

    fn at(&self, (col, row): (i32, i32)) -> Option<&T> {
        if row < 0 || row >= self.number_of_rows || col < 0 || col >= self.number_of_columns {
            return None;
        }
        Some(&self.row_data[row as usize][col as usize])
    }

    fn at_mut(&mut self, (col, row): (i32, i32)) -> Option<&mut T> {
        if row < 0 || row >= self.number_of_rows || col < 0 || col >= self.number_of_columns {
            return None;
        }
        Some(&mut self.row_data[row as usize][col as usize])
    }

    fn indexed_for_each<F>(&self, mut f: F)
    where
        F: FnMut((i32, i32), &T),
    {
        for (y, row) in self.row_data.iter().enumerate() {
            for (x, value) in row.iter().enumerate() {
                f((x.try_into().unwrap(), y.try_into().unwrap()), value);
            }
        }
    }

    fn indexed_map<F, B>(&self, mut f: F) -> Map<B>
    where
        F: FnMut((i32, i32), &T) -> B,
        B: Clone,
    {
        let row_data = self
            .row_data
            .iter()
            .enumerate()
            .map(|(row_index, row)| {
                row.iter()
                    .enumerate()
                    .map(|(col_index, value)| {
                        f(
                            (col_index.try_into().unwrap(), row_index.try_into().unwrap()),
                            value,
                        )
                    })
                    .collect()
            })
            .collect();
        Map {
            number_of_rows: self.number_of_rows,
            number_of_columns: self.number_of_columns,
            row_data,
        }
    }

    fn in_bounds(&self, (col, row): (i32, i32)) -> bool {
        !(row < 0 || row >= self.number_of_rows || col < 0 || col >= self.number_of_columns)
    }

    fn neighbours(&self, (col, row): (i32, i32)) -> Vec<(i32, i32)> {
        [(0, -1), (0, 1), (-1, 0), (1, 0)]
            .iter()
            .filter_map(|(dx, dy)| {
                let p = (col + dx, row + dy);
                if self.in_bounds(p) {
                    Some(p)
                } else {
                    None
                }
            })
            .collect()
    }

    fn find(&self, needle: T) -> Vec<(i32, i32)>
    where
        T: PartialEq,
    {
        let mut rv: Vec<(i32, i32)> = Vec::new();

        for (y, row) in self.row_data.iter().enumerate() {
            for (x, value) in row.iter().enumerate() {
                if *value == needle {
                    rv.push((x.try_into().unwrap(), y.try_into().unwrap()));
                }
            }
        }

        rv
    }

    fn map<F, B>(&self, f: F) -> Map<B>
    where
        F: Fn(&T) -> B,
        B: Clone,
    {
        self.indexed_map(|_, x| f(x))
    }

    fn show<F>(&self, format_cell: F) -> String
    where
        F: Fn(&T) -> String,
    {
        let mut rv: Vec<String> = Vec::new();

        for row in &self.row_data {
            for col in row {
                rv.push(format_cell(col));
            }
            rv.push("\n".to_string());
        }

        rv.join("")
    }

    fn values(&self) -> Vec<T> {
        let mut rv = Vec::new();

        for row in &self.row_data {
            for col in row {
                rv.push(col.clone());
            }
        }

        rv
    }
}

fn parse_point(s: &str) -> Result<(i32, i32)> {
    let mut xy = s.split(',');
    let x = xy.next().unwrap().parse()?;
    let y = xy.next().unwrap().parse()?;
    assert!(xy.next().is_none());
    Ok((x, y))
}

fn simulate_sand(map: &mut Map<char>, mut x: i32, mut y: i32) -> Result<Option<(i32, i32)>> {
    if map.at((x, y)).cloned() == Some('o') {
        return Ok(None);
    }

    loop {
        if x < 0 || x > map.number_of_columns {
            anyhow::bail!("simulation went out of bounds on x-axis: {}", x);
        }

        if map.at((x, y + 1)).is_none() {
            return Ok(None);
        }

        if map.at((x, y + 1)).cloned() == Some('.') {
            y += 1;
            continue;
        }

        if map.at((x - 1, y + 1)).cloned() == Some('.') {
            y += 1;
            x -= 1;
            continue;
        }

        if map.at((x + 1, y + 1)).cloned() == Some('.') {
            y += 1;
            x += 1;
            continue;
        }

        *map.at_mut((x, y)).unwrap() = 'o';
        return Ok(Some((x, y)));
    }
}

fn drop_block(
    map: &mut Map<char>,
    block: &Map<char>,
    jets: &[char],
    jet_index: &mut usize,
    tower_height: i32,
) -> Result<(i32, Option<i32>)> {
    let from_bottom = tower_height + 3 + block.number_of_rows;

    let mut x = 2;
    let mut y = map.number_of_rows - from_bottom;

    let mut new_max_y = map.number_of_rows;

    loop {
        // Move by jet if possible
        let (ddx, ddy) = match jets[*jet_index] {
            '>' => (1, 0),
            '<' => (-1, 0),
            _ => panic!("oops"),
        };
        *jet_index = (*jet_index + 1) % jets.len();

        let mut is_blocked = false;
        block.indexed_for_each(|(dx, dy), v| {
            if *v != '.' {
                if match map.at((x + dx + ddx, y + dy + ddy)) {
                    Some('.') => false,
                    _ => true,
                } {
                    is_blocked = true;
                }
            }
        });
        if !is_blocked {
            x += ddx;
            y += ddy;
        }

        // Move down if possible
        let (ddx, ddy) = (0, 1);
        let mut is_blocked = false;
        block.indexed_for_each(|(dx, dy), v| {
            if *v != '.' {
                if match map.at((x + dx + ddx, y + dy + ddy)) {
                    Some('.') => false,
                    _ => true,
                } {
                    is_blocked = true;
                }
            }
        });
        if !is_blocked {
            x += ddx;
            y += ddy;
        } else {
            break;
        }
    }

    // freeze
    block.indexed_for_each(|(dx, dy), v| {
        if *v != '.' {
            new_max_y = new_max_y.min(y + dy);
            *map.at_mut((x + dx, y + dy)).unwrap() = *v;
        }
    });

    let mut full_row: Option<i32> = None;

    for dy in 0..block.number_of_rows {
        let y = y + (block.number_of_rows - 1 - dy);
        let row_is_full = (0..map.number_of_columns).all(|x| *map.at((x, y)).unwrap() != '.');
        if row_is_full {
            let at_tower_height = map.number_of_rows - y;
            full_row = Some(at_tower_height);
        }
    }

    let new_tower_height = map.number_of_rows - new_max_y;
    let new_tower_height = new_tower_height.max(tower_height);

    return Ok((new_tower_height, full_row));
}

fn scan_bottom(map: &Map<char>, last_full_row: i32, tower_height: i32) -> [u16; 7] {
    assert!(map.number_of_columns == 7);
    assert!(last_full_row <= tower_height);

    let mut rv = [0; 7];
    let max_value = tower_height - last_full_row;
    assert!(max_value <= 65536);
    let max_value: u16 = max_value.try_into().unwrap();

    for x in 0..7 {
        for dy in 0..=(max_value as i32) {
            let y = map.number_of_rows - (tower_height - dy);
            if *map.at((x, y)).unwrap() != '.' {
                rv[x as usize] = dy as u16;
                break;
            }
        }
        rv[x as usize] = max_value - rv[x as usize];
    }

    rv
}

fn main() -> Result<()> {
    let mut map: Map<char> = Map::new(7, 1_000_000, &'.');
    let blocks = vec![
        Map::from_string("0000")?,
        Map::from_string(".1.\n111\n.1.")?,
        Map::from_string("..2\n..2\n222")?,
        Map::from_string("3\n3\n3\n3")?,
        Map::from_string("44\n44")?,
    ];
    let mut tower_height = 0;
    let mut buffer = String::new();
    std::io::stdin().read_to_string(&mut buffer)?;
    let jets: Vec<char> = buffer.trim().chars().collect();
    let mut jet_index = 0;

    let mut blocks_dropped: i64 = 0;

    let mut repeats: HashMap<(u8, usize, [u16; 7]), (i64, i64)> = HashMap::new();

    let mut calculated_tower_heights: Vec<i32> = Vec::new();

    loop {
        calculated_tower_heights.push(tower_height);

        let block = &blocks[(blocks_dropped as usize) % blocks.len()];
        let (new_tower_height, maybe_full_row) =
            drop_block(&mut map, &block, &jets, &mut jet_index, tower_height)?;
        tower_height = new_tower_height;

        if tower_height > 1_000_000 {
            break;
        }

        blocks_dropped += 1;
        if blocks_dropped == 2022 {
            println!(
                "Tower height after {} blocks dropped: {}",
                blocks_dropped, tower_height
            );
        }

        if let Some(full_row) = maybe_full_row {
            let bottom = scan_bottom(&map, full_row, tower_height);
            let block_index: u8 = (blocks_dropped % (blocks.len() as i64)).try_into().unwrap();
            let key = (block_index, jet_index, bottom);

            if let Some((last_tower_height, last_blocks_dropped)) = repeats.get(&key) {
                // Note: there are some tricksy potential bottom-shapes where the shapes are NOT
                // all equivalent to each other. We should really filter these out.
                // However, in my test input, the first repeat doesn't have this problem, so
                // I can let that problem wait.
                println!(
                    "Found repeat!!: {:?} from blocks:{}->{} height:{}->{}",
                    key, last_blocks_dropped, blocks_dropped, last_tower_height, tower_height,
                );
                let delta_blocks: i64 = blocks_dropped - last_blocks_dropped;
                let delta_height: i64 = (tower_height as i64) - last_tower_height;
                let target: i64 = 1000000000000;

                let mut accumulated_height: i64 = tower_height.into();
                let mut left_to_target = target - blocks_dropped;

                while left_to_target > delta_blocks {
                    let mut shift = 0;
                    while left_to_target > (delta_blocks << (shift + 1)) {
                        shift += 1;
                    }
                    left_to_target -= delta_blocks << shift;
                    accumulated_height += delta_height << shift;
                    println!("With {} to go: {}", left_to_target, accumulated_height);
                }

                let j = *last_blocks_dropped;
                let k = last_blocks_dropped + left_to_target;
                let final_delta =
                    calculated_tower_heights[k as usize] - calculated_tower_heights[j as usize];
                println!(
                    "With 0 to go: {}",
                    accumulated_height + (final_delta as i64)
                );
                break;
            }

            repeats.insert(key, (tower_height.into(), blocks_dropped));

            /*
            println!("{}", map.show(|v| v.to_string()));
            println!("bottom: {:?}", bottom);
            println!(
                "full row discovered at {} with tower height {} after {} blocks: {} {}",
                full_row, tower_height, blocks_dropped, block_index, jet_index
            );
            assert!(*bottom.iter().max().unwrap() as i32 + full_row == tower_height);
            break;
            */
        }
    }

    Ok(())
}
