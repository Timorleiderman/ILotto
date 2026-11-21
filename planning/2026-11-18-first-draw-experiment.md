# Can one ticket start a generator that reproduces later draws?

## What the archive tells us

The local modern archive through November 14, 2026 has 1,664 draws. All
1,664 full tickets are distinct. A table that maps each of the first 1,663
tickets to its successor can replay the archive from draw one. Its entry
for draw 1,664 can point to any of the 16,273,488 valid tickets. Under
independent fair draws, the probability of seeing zero repeats among the
1,663 tickets with recorded successors is about 91.9%.

The earlier replay ruled out every fixed linear congruential rule on the
project's ticket index. Its odd/even contradiction depends on that encoding.
Another simple hypothesis can be checked directly on the balls. If each
draw uses one fixed relabeling of the previous six balls, the number shared
by adjacent draws must stay constant. Draws one and two share only 25.
Draws two and three share 25 and 28. That rules out every fixed permutation
of the 37 ball labels, independently of the strong number.

## Proposed pilot

Test whether a generator initialized once from the first complete ticket
can reproduce successive draws by selecting the balls one at a time.
A generator can carry state between draws even when each published ticket
shows only part of that state. This is a testable software model.

1. **Freeze four candidates.** Encode the first ticket with the existing
   zero-based `ticket_index`. Initialize either NumPy PCG64 or MT19937
   once with that integer. For each draw, start with a fresh ascending
   list of balls 1–37. Six times, sample a uniform zero-based index with
   `Generator.integers(len(pool))` and remove the ball at that index.
   Sort the six for comparison. Draw the strong number with
   `Generator.integers(1, 8)` from the same stream. Test two fixed orders
   for that one call, before or after the six main balls. Record the NumPy
   version. These choices give four candidates and require no seed search.
2. **Test the checker blind.** Have a fixture builder generate histories
   from each candidate using several initial tickets. Give the checker
   the starting ticket and a short fitting prefix. Keep the candidate
   identity and the next 1,000 tickets in the scoring fixture. Verify
   that the planted candidate survives and replays the withheld tickets.
   Also change one known ticket and verify the reported first mismatch.
   Run the identical four-candidate search on independent fair histories
   and record the longest exact streak across the entire search.
3. **Replay the archive once.** Start all four candidates from draw one.
   Record each complete forecast before comparing it with the corresponding
   archived draw. Eliminate a candidate on its first miss. Record each
   candidate's first failure and every survivor. The old archive is
   exploratory because its outcomes have already informed this project.
4. **Decide what follows.** If every candidate fails, close this pilot.
   A new hypothesis gets a new version and a reason for changing the rule.
   If a candidate replays the archive, freeze its code, initialization and
   state. If several survive, give PCG64 priority over MT19937 and
   strong-last priority within each generator. Timestamp the selected
   generator's one complete ticket before the next draw.
   Repeat for the next draw with the same generator state. A miss ends
   that version's exact-sequence claim. Retain every issued prediction.

For one fixed ticket at each of two predeclared independent fair draws,
the chance of two consecutive exact matches is 1 in 264,826,411,686,144.
That calculation applies to a single prospectively recorded sequence.
Historical fitting, testing more candidates or choosing a successful
window afterward changes what the result can support.
