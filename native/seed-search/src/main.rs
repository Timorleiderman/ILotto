//! Fast finite batches for the existing calendar-sha256-v1 generator.
//! Python owns the ledger; threads only process disjoint intervals in memory.

use serde::{Deserialize, Serialize};
use sha2::{compress256, digest::generic_array::GenericArray};
use std::io::{self, Read};
use std::thread;

const GENERATOR: &str = "calendar-sha256-v1";
const DOMAIN: &[u8] = b"calendar-sha256-v1:";
const OUTCOMES: u64 = 16_273_488;
const INITIAL: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];
const SURVIVOR_CAP: usize = 10_000;

#[derive(Deserialize)]
struct Observation {
    date: String,
    outcome: u64,
}

#[derive(Deserialize)]
struct Request {
    generator: String,
    start: String,
    stop: String,
    workers: usize,
    prefix: Vec<Observation>,
}

#[derive(Serialize, Debug, PartialEq)]
struct ResultBatch {
    generator: &'static str,
    start: String,
    stop: String,
    tested: u64,
    matched_prefix_counts: Vec<u64>,
    survivors: Vec<u64>,
    survivor_overflow: bool,
}

#[derive(Clone)]
struct Prepared {
    block: [u8; 64],
    expected: u64,
}

/// Canonical YYYY-MM-DD with a real calendar day, matching Python's
/// `date.fromisoformat(value).isoformat() == value` and the CUDA worker.
fn canonical_date(day: &str) -> bool {
    let bytes = day.as_bytes();
    if bytes.len() != 10 || bytes[4] != b'-' || bytes[7] != b'-' {
        return false;
    }
    if !bytes
        .iter()
        .enumerate()
        .all(|(i, b)| i == 4 || i == 7 || b.is_ascii_digit())
    {
        return false;
    }
    let number = |range: std::ops::Range<usize>| {
        bytes[range]
            .iter()
            .fold(0_u32, |n, b| n * 10 + u32::from(b - b'0'))
    };
    let (year, month, date) = (number(0..4), number(5..7), number(8..10));
    let leap = year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
    let days = match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if leap => 29,
        2 => 28,
        _ => return false,
    };
    year >= 1 && (1..=days).contains(&date)
}

fn prepare(day: &str, expected: u64) -> Result<Prepared, String> {
    if !canonical_date(day) || expected >= OUTCOMES {
        return Err("Invalid canonical date or outcome".into());
    }
    let mut block = [0_u8; 64];
    let length = DOMAIN.len() + 8 + 10 + 4;
    block[..DOMAIN.len()].copy_from_slice(DOMAIN);
    block[DOMAIN.len() + 8..DOMAIN.len() + 18].copy_from_slice(day.as_bytes());
    block[length] = 0x80;
    block[56..].copy_from_slice(&((length as u64) * 8).to_be_bytes());
    Ok(Prepared { block, expected })
}

/// 2^(32k) mod OUTCOMES for k = 0..8. Entry 8 is 2^256 mod OUTCOMES, the size
/// of the rejected tail. Computed at compile time, never in the hot loop.
const POWERS: [u64; 9] = {
    let mut powers = [1_u64; 9];
    let mut k = 1;
    while k < 9 {
        powers[k] = (powers[k - 1] << 32) % OUTCOMES;
        k += 1;
    }
    powers
};
const TAIL: u64 = POWERS[8];

#[inline]
fn reduce_hash(words: &[u32; 8]) -> Option<u64> {
    // The rejected tail is smaller than 2^32, so its upper seven words are all ones.
    if words[..7].iter().all(|&x| x == u32::MAX) && u64::from(words[7]) >= (1_u64 << 32) - TAIL {
        return None;
    }
    // sum(word_i * 2^(32*(7-i)) mod N) is congruent to the 256-bit value mod N.
    // Each term is < 2^32 * 2^24 = 2^56, so eight of them sum below 2^59: one
    // division replaces eight chained 64-bit divisions.
    let mut sum = 0_u64;
    for (i, &word) in words.iter().enumerate() {
        sum += u64::from(word) * POWERS[7 - i];
    }
    Some(sum % OUTCOMES)
}

#[inline]
fn outcome(seed: u64, prepared: &Prepared) -> u64 {
    let mut block = prepared.block;
    block[DOMAIN.len()..DOMAIN.len() + 8].copy_from_slice(&seed.to_be_bytes());
    let counter_start = DOMAIN.len() + 18;
    let mut counter = 0_u32;
    loop {
        block[counter_start..counter_start + 4].copy_from_slice(&counter.to_be_bytes());
        let mut state = INITIAL;
        compress256(&mut state, &[GenericArray::clone_from_slice(&block)]);
        if let Some(value) = reduce_hash(&state) {
            return value;
        }
        counter = counter
            .checked_add(1)
            .expect("Hash rejection counter exhausted");
    }
}

fn batch(start: u128, stop: u128, prefix: &[Prepared]) -> ResultBatch {
    let mut counts = vec![0; prefix.len() + 1];
    let mut survivors = Vec::new();
    let mut overflow = false;
    for seed in start..stop {
        let mut matched = 0;
        for observation in prefix {
            if outcome(seed as u64, observation) != observation.expected {
                break;
            }
            matched += 1;
        }
        counts[matched] += 1;
        if matched == prefix.len() {
            if survivors.len() < SURVIVOR_CAP {
                survivors.push(seed as u64);
            } else {
                overflow = true;
            }
        }
    }
    ResultBatch {
        generator: GENERATOR,
        start: start.to_string(),
        stop: stop.to_string(),
        tested: (stop - start) as u64,
        matched_prefix_counts: counts,
        survivors,
        survivor_overflow: overflow,
    }
}

fn execute(request: Request) -> Result<ResultBatch, String> {
    let start = request.start.parse::<u128>().map_err(|e| e.to_string())?;
    let stop = request.stop.parse::<u128>().map_err(|e| e.to_string())?;
    if request.generator != GENERATOR
        || start >= stop
        || stop > (1_u128 << 64)
        || stop - start > 1_000_000_000
        || !(1..=64).contains(&request.workers)
        || request.prefix.is_empty()
        || request.prefix.len() > 10_000
    {
        return Err("Invalid generator, range, worker count, or prefix".into());
    }
    let prefix = request
        .prefix
        .iter()
        .map(|x| prepare(&x.date, x.outcome))
        .collect::<Result<Vec<_>, _>>()?;
    let workers = request.workers.min((stop - start) as usize);
    let pieces = thread::scope(|scope| {
        let mut handles = Vec::new();
        for index in 0..workers {
            let a = start + (stop - start) * index as u128 / workers as u128;
            let b = start + (stop - start) * (index + 1) as u128 / workers as u128;
            let prefix_ref = &prefix;
            handles.push(scope.spawn(move || batch(a, b, prefix_ref)));
        }
        handles
            .into_iter()
            .map(|h| h.join().expect("Worker failed"))
            .collect::<Vec<_>>()
    });
    let mut result = ResultBatch {
        generator: GENERATOR,
        start: start.to_string(),
        stop: stop.to_string(),
        tested: 0,
        matched_prefix_counts: vec![0; prefix.len() + 1],
        survivors: Vec::new(),
        survivor_overflow: false,
    };
    for piece in pieces {
        result.tested += piece.tested;
        for (total, n) in result
            .matched_prefix_counts
            .iter_mut()
            .zip(piece.matched_prefix_counts)
        {
            *total += n;
        }
        result.survivors.extend(piece.survivors);
        result.survivor_overflow |= piece.survivor_overflow;
    }
    if result.survivors.len() > SURVIVOR_CAP {
        result.survivors.truncate(SURVIVOR_CAP);
        result.survivor_overflow = true;
    }
    Ok(result)
}

fn main() {
    let mut input = String::new();
    io::stdin()
        .read_to_string(&mut input)
        .expect("Cannot read batch request");
    let result = serde_json::from_str::<Request>(&input)
        .map_err(|e| e.to_string())
        .and_then(execute);
    match result {
        Ok(output) => println!("{}", serde_json::to_string(&output).unwrap()),
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};

    #[test]
    fn python_reference_outputs() {
        assert_eq!(outcome(0, &prepare("2020-01-01", 0).unwrap()), 15_611_497);
        assert_eq!(
            outcome(45_123, &prepare("2020-01-01", 0).unwrap()),
            6_083_258
        );
        assert_eq!(
            outcome(u64::MAX, &prepare("2026-11-06", 0).unwrap()),
            9_078_729
        );
    }

    #[test]
    fn compression_matches_standard_hash_api() {
        for seed in [0_u64, 1, 123, 101_000_000, u64::MAX] {
            let prepared = prepare("2026-11-03", 0).unwrap();
            let mut input = DOMAIN.to_vec();
            input.extend(seed.to_be_bytes());
            input.extend(b"2026-11-03");
            input.extend(0_u32.to_be_bytes());
            let digest = Sha256::digest(input);
            let words: [u32; 8] = std::array::from_fn(|i| {
                u32::from_be_bytes(digest[i * 4..i * 4 + 4].try_into().unwrap())
            });
            assert_eq!(Some(outcome(seed, &prepared)), reduce_hash(&words));
        }
    }

    /// The single-division reduction must equal the original chained reduction.
    fn chained_reduce(words: &[u32; 8]) -> u64 {
        let mut remainder = 0_u64;
        for &word in words {
            remainder = ((remainder << 32) | u64::from(word)) % OUTCOMES;
        }
        remainder
    }

    #[test]
    fn single_division_reduction_matches_chained() {
        let mut x = 0x9e37_79b9_7f4a_7c15_u64;
        for _ in 0..200_000 {
            let words: [u32; 8] = std::array::from_fn(|_| {
                // splitmix64: deterministic, well-spread test words.
                x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
                let mut z = x;
                z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
                (z ^ (z >> 31)) as u32
            });
            assert_eq!(reduce_hash(&words), Some(chained_reduce(&words)));
        }
        let extremes = [
            [0_u32; 8],
            [u32::MAX; 8],
            [u32::MAX, 0, u32::MAX, 0, 1, 2, 3, 4],
        ];
        for words in extremes {
            if let Some(value) = reduce_hash(&words) {
                assert_eq!(value, chained_reduce(&words));
            }
        }
    }

    #[test]
    fn invalid_calendar_dates_are_rejected() {
        for day in [
            "2020-13-01",
            "2021-02-29",
            "2020-1a-01",
            "2020/01/01",
            "0000-01-01",
        ] {
            assert!(prepare(day, 0).is_err(), "{day} was accepted");
        }
        assert!(prepare("2020-02-29", 0).is_ok());
    }

    #[test]
    fn rejection_boundary() {
        let tail = TAIL;
        let mut words = [u32::MAX; 8];
        assert!(reduce_hash(&words).is_none());
        words[7] = ((1_u64 << 32) - tail) as u32;
        assert!(reduce_hash(&words).is_none());
        words[7] -= 1;
        assert_eq!(reduce_hash(&words), Some(OUTCOMES - 1));
    }

    #[test]
    fn uneven_parallel_ranges_equal_serial() {
        fn request(workers: usize) -> Request {
            Request {
                generator: GENERATOR.into(),
                start: "1".into(),
                stop: "333".into(),
                workers,
                prefix: vec![Observation {
                    date: "2020-01-01".into(),
                    outcome: outcome(123, &prepare("2020-01-01", 0).unwrap()),
                }],
            }
        }
        assert_eq!(execute(request(1)).unwrap(), execute(request(14)).unwrap());
        assert_eq!(execute(request(14)).unwrap().survivors, vec![123]);
    }

    #[test]
    fn upper_seed_boundary_is_supported() {
        let request = Request {
            generator: GENERATOR.into(),
            start: u64::MAX.to_string(),
            stop: (1_u128 << 64).to_string(),
            workers: 14,
            prefix: vec![Observation {
                date: "2026-11-06".into(),
                outcome: 9_078_729,
            }],
        };
        assert_eq!(execute(request).unwrap().survivors, vec![u64::MAX]);
    }
}
