# Biped evidence log

Every claim we might post gets a file here and a line below: date, file, what it shows, the number. Newest at the bottom.

| When (CDT) | File | What it shows |
|---|---|---|
| 2026-10-21 23:05 | `2026-10-21_gait-strip_first-walk_seed0.png` | First learned gait, numpy prototype, ARS on a 4x13 linear policy, P=32, 150 generations: 4.00 m in 5 s, upright the whole episode, 14 foot-contact switches, knees never touch the ground. Limps (right stance longer). |
| 2026-10-21 23:15 | `2026-10-21_gait-strip_gallop-exploit_seed1.png` | Failure worth showing: with uncapped position-based motors the optimizer found a two-footed gallop at ~3 m/s with 0.5 s flights (15.2 m in 5 s). Motors were effectively infinitely strong. |
| 2026-10-21 23:28 | `2026-10-21_gait-strip_capped_seed3.png`, `..._seed2.png` | After capping joint rotation per pass (MAX_ROT 0.05 rad, a strength limit): clean alternating left/right stance phases, 5.42 m in 5 s (seed 3), upright throughout. All 4 seeds 3.0 to 5.4 m. Logs: `2026-10-21_train-log_capped_seed*.txt`. |
| 2026-10-21 23:20 | (numbers, no file) | Desmos port equals numpy: actions and all six joint positions after one tick match to 6+ decimals; after 28 ticks (uncapped) and 55 ticks (capped) the Desmos state matches the numpy state with max abs difference 0.0. |
| 2026-10-21 23:22 | (numbers, no file) | Desmos cost model: per-tick latency flat across population 16/32/64 (~200-220 ms), scales with expression count; fused physics passes reached 6.5 ticks/s under the real ticker (window in front, rAF 61/s). |
| 2026-10-21 23:30 | `2026-10-21_desmos-first-render_trained-walker_stray-helper-curves.jpg` | The trained walker (orange, best individual) rendering inside Desmos with the expression list open. The blue diagonal and horizontal lines are a bug (Desmos plots one-variable helper functions); fixed afterwards by hiding all non-drawing expressions. Not for posting; kept as a build record. |
| 2026-10-22 00:31 | `2026-10-22_desmos-fused-6-steps-per-tick_population-running.jpg` | Fast generator: 32 walkers (grey, each with its own noise) and the current best (orange) inside Desmos, expressions grouped into collapsed folders. Same-load comparison (load 19-24, CPU at 1.5 GHz): old graph 0.88-1.37 ticks/s at 1 physics step per tick; new graph 5.2-6.0 ticks/s at 6 physics steps per tick, i.e. about 31-36 simulated steps per second vs about 1. Numerics: tick 1 bit-identical to numpy, 6.7e-15 after 162 ticks. The "Choose Files" button top-left is the page loader input, not part of the graph. |
