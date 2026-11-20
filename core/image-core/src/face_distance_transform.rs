//! Small dependency-free Euclidean distance transform for raster support.
//!
//! The face and photo-transport candidates use this as a clearance field. A
//! positive input sample is a distance-transform seed; every output entry is
//! the distance in pixels to the nearest seed. The implementation uses a
//! multi-source Dijkstra pass over the 8-connected raster, which is stable for
//! antialiased masks and avoids pulling another matrix crate into the WASM
//! core.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

#[derive(Copy, Clone, Debug)]
struct State {
    distance: f64,
    index: usize,
}

impl PartialEq for State {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance && self.index == other.index
    }
}

impl Eq for State {}

impl Ord for State {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .distance
            .total_cmp(&self.distance)
            .then_with(|| self.index.cmp(&other.index))
    }
}

impl PartialOrd for State {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

pub(super) fn support_distance(alpha: &[f32], width: usize, height: usize) -> Vec<f64> {
    let count = width.saturating_mul(height);
    if count == 0 || alpha.len() != count {
        return vec![f64::INFINITY; count];
    }
    let mut distances = vec![f64::INFINITY; count];
    let mut queue = BinaryHeap::new();
    for (index, &value) in alpha.iter().enumerate() {
        if value > 0.0 {
            distances[index] = 0.0;
            queue.push(State {
                distance: 0.0,
                index,
            });
        }
    }
    let neighbours = [
        (-1_i32, -1_i32, 2.0_f64.sqrt()),
        (0, -1, 1.0),
        (1, -1, 2.0_f64.sqrt()),
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (-1, 1, 2.0_f64.sqrt()),
        (0, 1, 1.0),
        (1, 1, 2.0_f64.sqrt()),
    ];
    while let Some(State { distance, index }) = queue.pop() {
        if distance > distances[index] {
            continue;
        }
        let x = index % width;
        let y = index / width;
        for &(dx, dy, weight) in &neighbours {
            let nx = x as i32 + dx;
            let ny = y as i32 + dy;
            if nx < 0 || ny < 0 || nx >= width as i32 || ny >= height as i32 {
                continue;
            }
            let neighbour = ny as usize * width + nx as usize;
            let candidate = distance + weight;
            if candidate < distances[neighbour] {
                distances[neighbour] = candidate;
                queue.push(State {
                    distance: candidate,
                    index: neighbour,
                });
            }
        }
    }
    distances
}
