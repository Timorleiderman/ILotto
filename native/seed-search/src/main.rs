//! Fast finite batches for the existing calendar-sha256-v1 generator.
//! Python owns the ledger; threads only process disjoint intervals in memory.

use serde::{Deserialize, Serialize};
use sha2::{compress256, digest::generic_array::GenericArray};
use std::io::{self, Read};
use std::thread;

const GENERATOR: &str = "calendar-sha256-v1";
const DOMAIN: &[u8] = b"calendar-sha256-v1:";
const OUTCOMES: u64 = 16_273_488;
const ODD_MOD: u64 = OUTCOMES >> 4; // 1_017_093 (16 | OUTCOMES)
const ODD_INV: u64 = {
    let mut x = 1_u64;
    let mut i = 0;
    while i < 6 {
        x = x.wrapping_mul(2_u64.wrapping_sub(ODD_MOD.wrapping_mul(x)));
        i += 1;
    }
    x
};
const ODD_MAX_Q: u64 = u64::MAX / ODD_MOD;
const _: () = assert!(OUTCOMES & 15 == 0 && ODD_MOD & 1 == 1 && ODD_MOD.wrapping_mul(ODD_INV) == 1);

const INITIAL: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];
#[cfg(target_arch = "aarch64")]
const SHA256_K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
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
const POWERS_DIV16: [u64; 8] = {
    let mut p = [0_u64; 8];
    let mut k = 1;
    while k < 8 {
        p[k] = POWERS[k] >> 4;
        k += 1;
    }
    p
};

#[inline(always)]
fn in_rejected_tail(words: &[u32; 8]) -> bool {
    // The rejected tail is smaller than 2^32, so its upper seven words are all ones.
    if words[0] != u32::MAX {
        return false;
    }
    words[1..7].iter().all(|&x| x == u32::MAX) && u64::from(words[7]) >= (1_u64 << 32) - TAIL
}

#[inline]
fn reduce_hash(words: &[u32; 8]) -> Option<u64> {
    if in_rejected_tail(words) {
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

/// Fast exact match check against `expected`: returns `Some(true)` on match,
/// `Some(false)` on mismatch, or `None` if `words` falls in the rejected tail.
#[inline(always)]
fn matches_expected_words(words: &[u32; 8], expected: u64) -> Option<bool> {
    if in_rejected_tail(words) {
        return None;
    }
    // 16 | OUTCOMES and 16 | 2^32, so H mod 16 == words[7] & 15; rejects 15/16 immediately.
    if (words[7] & 15) != (expected as u32 & 15) {
        return Some(false);
    }
    let mut y = (u64::from(words[7]) + (OUTCOMES - expected)) >> 4;
    y += u64::from(words[0]) * POWERS_DIV16[7];
    y += u64::from(words[1]) * POWERS_DIV16[6];
    y += u64::from(words[2]) * POWERS_DIV16[5];
    y += u64::from(words[3]) * POWERS_DIV16[4];
    y += u64::from(words[4]) * POWERS_DIV16[3];
    y += u64::from(words[5]) * POWERS_DIV16[2];
    y += u64::from(words[6]) * POWERS_DIV16[1];
    Some(y.wrapping_mul(ODD_INV) <= ODD_MAX_Q)
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

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "sha2")]
unsafe fn filter_first_aarch64_x4(
    base_words: &[u32; 16],
    expected: u64,
    start: u128,
    stop: u128,
    first: &Prepared,
    hits: &mut Vec<u64>,
) {
    use std::arch::aarch64::*;

    let init0 = vld1q_u32(INITIAL.as_ptr());
    let init1 = vld1q_u32(INITIAL.as_ptr().add(4));
    let k_vecs: [uint32x4_t; 16] = std::array::from_fn(|i| vld1q_u32(SHA256_K.as_ptr().add(i * 4)));
    let w0_const = vld1q_u32(base_words.as_ptr());
    let w2_base = vld1q_u32(base_words.as_ptr().add(8));
    let w3_const = vld1q_u32(base_words.as_ptr().add(12));

    // Rounds 0..3 depend only on words 0..3 (fixed domain prefix), identical for every seed.
    let wk0 = vaddq_u32(w0_const, k_vecs[0]);
    let mid0 = vsha256hq_u32(init0, init1, wk0);
    let mid1 = vsha256h2q_u32(init1, init0, wk0);

    let w4_base = base_words[4];
    let w6_base = base_words[6];
    let w7_const = base_words[7];
    let exp_nibble = expected as u32 & 15;

    let mut seed = start;
    while seed + 4 <= stop {
        let s = [seed as u64, (seed + 1) as u64, (seed + 2) as u64, (seed + 3) as u64];
        let mut a = [mid0; 4];
        let mut e = [mid1; 4];
        let mut w0 = [w0_const; 4];
        let mut w1: [uint32x4_t; 4] = std::array::from_fn(|j| {
            let arr = [
                w4_base | ((s[j] >> 56) as u32),
                (s[j] >> 24) as u32,
                w6_base | ((s[j] as u32) << 8),
                w7_const,
            ];
            vld1q_u32(arr.as_ptr())
        });
        let mut w2 = [w2_base; 4];
        let mut w3 = [w3_const; 4];

        macro_rules! round4 {
            ($w:expr, $k:expr) => {{
                let wk0 = vaddq_u32($w[0], $k);
                let wk1 = vaddq_u32($w[1], $k);
                let wk2 = vaddq_u32($w[2], $k);
                let wk3 = vaddq_u32($w[3], $k);
                let t0 = a[0]; a[0] = vsha256hq_u32(a[0], e[0], wk0); e[0] = vsha256h2q_u32(e[0], t0, wk0);
                let t1 = a[1]; a[1] = vsha256hq_u32(a[1], e[1], wk1); e[1] = vsha256h2q_u32(e[1], t1, wk1);
                let t2 = a[2]; a[2] = vsha256hq_u32(a[2], e[2], wk2); e[2] = vsha256h2q_u32(e[2], t2, wk2);
                let t3 = a[3]; a[3] = vsha256hq_u32(a[3], e[3], wk3); e[3] = vsha256h2q_u32(e[3], t3, wk3);
            }};
        }
        macro_rules! sched4 {
            ($wa:expr, $wb:expr, $wc:expr, $wd:expr) => {{
                $wa[0] = vsha256su1q_u32(vsha256su0q_u32($wa[0], $wb[0]), $wc[0], $wd[0]);
                $wa[1] = vsha256su1q_u32(vsha256su0q_u32($wa[1], $wb[1]), $wc[1], $wd[1]);
                $wa[2] = vsha256su1q_u32(vsha256su0q_u32($wa[2], $wb[2]), $wc[2], $wd[2]);
                $wa[3] = vsha256su1q_u32(vsha256su0q_u32($wa[3], $wb[3]), $wc[3], $wd[3]);
            }};
        }

        // Rounds 4..15
        round4!(w1, k_vecs[1]);
        round4!(w2, k_vecs[2]);
        round4!(w3, k_vecs[3]);

        // Rounds 16..63 (12 groups of 4 rounds)
        for g in 0..3 {
            let base = 4 + g * 4;
            sched4!(w0, w1, w2, w3);
            round4!(w0, k_vecs[base]);
            sched4!(w1, w2, w3, w0);
            round4!(w1, k_vecs[base + 1]);
            sched4!(w2, w3, w0, w1);
            round4!(w2, k_vecs[base + 2]);
            sched4!(w3, w0, w1, w2);
            round4!(w3, k_vecs[base + 3]);
        }

        for j in 0..4 {
            let h0 = vaddq_u32(a[j], init0);
            let h1 = vaddq_u32(e[j], init1);
            let w7 = vgetq_lane_u32::<3>(h1);
            let w0_lane = vgetq_lane_u32::<0>(h0);
            if (w7 & 15) == exp_nibble || w0_lane == u32::MAX {
                let mut words = [0_u32; 8];
                vst1q_u32(words.as_mut_ptr(), h0);
                vst1q_u32(words.as_mut_ptr().add(4), h1);
                match matches_expected_words(&words, expected) {
                    Some(true) => hits.push(s[j]),
                    Some(false) => {}
                    None => {
                        if outcome(s[j], first) == expected {
                            hits.push(s[j]);
                        }
                    }
                }
            }
        }
        seed += 4;
    }

    while seed < stop {
        if outcome(seed as u64, first) == expected {
            hits.push(seed as u64);
        }
        seed += 1;
    }
}

fn filter_first_fallback(start: u128, stop: u128, first: &Prepared, hits: &mut Vec<u64>) {
    let mut block = GenericArray::clone_from_slice(&first.block);
    let expected = first.expected;
    for seed in start..stop {
        let s = seed as u64;
        block[DOMAIN.len()..DOMAIN.len() + 8].copy_from_slice(&s.to_be_bytes());
        let mut state = INITIAL;
        compress256(&mut state, std::slice::from_ref(&block));
        match matches_expected_words(&state, expected) {
            Some(true) => hits.push(s),
            Some(false) => {}
            None => {
                if outcome(s, first) == expected {
                    hits.push(s);
                }
            }
        }
    }
}

fn batch(start: u128, stop: u128, prefix: &[Prepared]) -> ResultBatch {
    let mut counts = vec![0; prefix.len() + 1];
    let mut survivors = Vec::new();
    let mut overflow = false;
    let first = &prefix[0];
    let mut first_hits = Vec::new();

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("sha2") {
            let words: [u32; 16] = std::array::from_fn(|i| {
                u32::from_be_bytes(first.block[i * 4..i * 4 + 4].try_into().unwrap())
            });
            unsafe {
                filter_first_aarch64_x4(&words, first.expected, start, stop, first, &mut first_hits);
            }
        } else {
            filter_first_fallback(start, stop, first, &mut first_hits);
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        filter_first_fallback(start, stop, first, &mut first_hits);
    }

    let tested = (stop - start) as u64;
    counts[0] = tested - first_hits.len() as u64;
    for seed in first_hits {
        let mut matched = 1;
        for observation in &prefix[1..] {
            if outcome(seed, observation) != observation.expected {
                break;
            }
            matched += 1;
        }
        counts[matched] += 1;
        if matched == prefix.len() {
            if survivors.len() < SURVIVOR_CAP {
                survivors.push(seed);
            } else {
                overflow = true;
            }
        }
    }
    ResultBatch {
        generator: GENERATOR,
        start: start.to_string(),
        stop: stop.to_string(),
        tested,
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
    let mut previous = "";
    let mut prefix = Vec::with_capacity(request.prefix.len());
    for item in &request.prefix {
        if item.date.as_str() <= previous {
            return Err("Invalid ordered prefix".into());
        }
        prefix.push(prepare(&item.date, item.outcome)?);
        previous = &item.date;
    }
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

    #[test]
    fn fast_match_matches_reduce_hash() {
        let mut x = 0x9e37_79b9_7f4a_7c15_u64;
        for _ in 0..200_000 {
            let words: [u32; 8] = std::array::from_fn(|_| {
                x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
                let mut z = x;
                z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
                (z ^ (z >> 31)) as u32
            });
            let exact = reduce_hash(&words).unwrap();
            assert_eq!(matches_expected_words(&words, exact), Some(true));
            assert_eq!(
                matches_expected_words(&words, (exact + 1) % OUTCOMES),
                Some(false)
            );
            assert_eq!(
                matches_expected_words(&words, (exact + 16) % OUTCOMES),
                Some(false)
            );
        }
        assert_eq!(matches_expected_words(&[u32::MAX; 8], 0), None);
    }

    #[test]
    fn unordered_or_duplicate_prefix_dates_are_rejected() {
        let req = Request {
            generator: GENERATOR.into(),
            start: "0".into(),
            stop: "10".into(),
            workers: 1,
            prefix: vec![
                Observation {
                    date: "2020-01-02".into(),
                    outcome: 0,
                },
                Observation {
                    date: "2020-01-02".into(),
                    outcome: 0,
                },
            ],
        };
        assert!(execute(req).is_err());
    }
}
