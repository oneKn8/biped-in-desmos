# A graphing calculator that teaches a ragdoll to walk

<p align="center">
  <img src="docs/learns-to-walk.gif" alt="Inside Desmos: 129 grey stick figures flail, the orange average walker improves generation by generation, and after 16 generations it walks; the graph stops learning by itself" width="820">
</p>

<p align="center"><i>Screen recording of the first graph: 129 bodies learn from zero (sped up 12x), the graph stops itself at generation 16 when the average walker (orange) passes the walking test, and its walk replays at real speed. Generation 13 went past the distance line but failed another part of the test (falls, steps or time in the air).</i></p>

A population of stick-figure ragdolls starts with blank controllers and learns to walk, entirely inside the [Desmos](https://www.desmos.com/calculator) graphing calculator. The physics, the controller, the learning rule and the test that decides "it walks" are all Desmos expressions. Python only writes those expressions and checks them against a numpy copy of the same model.

## Try it in your browser

| Graph | What happens | Time on a laptop |
|---|---|---|
| **[Learns to walk](https://www.desmos.com/calculator/zzd5dnayog)** | 129 bodies learn from zero. Learning stops by itself the first time the average walker passes the walking test (generation 16, 5.28 m in 5 s), then the graph replays that walk. | about 5 minutes |
| **[Learns to keep walking](https://www.desmos.com/calculator/yywath1bwc)** | 97 bodies practise on 5 s windows that continue their own walk and take a 20 s walk from a standing start as an exam. Learning stops after two exam passes in a row, then it replays the 20 s walk with the camera following. | about 40 minutes |

Collapse the expression list (the arrows at its top right), then press play on the ticker. Both runs are deterministic: the graph pins Desmos's random seed, so everyone sees the same run.

<p align="center">
  <img src="docs/keeps-walking.gif" alt="The walker learned in the second graph, walking from a standing start inside Desmos while the camera follows it" width="820">
</p>

<p align="center"><i>The second graph's learned walker replaying its 20 s exam inside Desmos: the first 16 s, at real speed (one frame per Desmos tick of 0.1 simulated seconds), cropped to follow the walker.</i></p>

## How it works

<p align="center"><img src="docs/body.png" alt="The ragdoll: head, hip, two knees and two feet joined by five bones, with motors at the hips and knees; next to it, the 13 inputs wired to 4 joint targets" width="820"></p>

- **Body.** A ragdoll in the style of Jakobsen's "Advanced Character Physics" (2001): six point masses, Verlet integration, bone-length constraints and joint-angle motors, solved with parallel (Jacobi) passes because Desmos evaluates whole lists at once. The motors are translation-neutral, so they cannot push the body along by themselves, and their speed is capped, which acts as a strength limit.
- **Brain.** A linear policy, one layer with tanh: 13 inputs (torso angle and spin, four joint angles, hip height and speed, two foot contacts, a clock as sine and cosine, a bias) mapped to 4 joint targets. That is 52 numbers.
- **Learning.** Augmented Random Search (Mania, Guy and Recht, 2018). Every episode, each random direction in weight space is tried with both signs next to the noise-free average, all bodies walk at once, and the average moves toward the directions that went farthest. Fitness is distance, minus a penalty for time with both feet off the ground beyond 15%, which rules out hopping and running.
- **The walking test.** The noise-free average must cover at least 0.8 m/s, never fall, take at least 3 alternating single-support steps per foot per 5 s, and keep both feet off the ground less than 25% of the time. The graph counts the steps itself.
- **Making it fit in a calculator.** Each physics family (all bones, all motors, all four policy outputs) is one vectorised formula over the whole population, the four constraint passes are nested function calls, and one ticker action advances six physics steps.

The second graph wraps the same learner in a practice-and-exam loop, because walkers that pass a 5 s test tend to fall soon after:

```mermaid
flowchart LR
    W["5 s practice window<br/>(continues the last walk)"] -->|"passed, but not twice in a row"| W
    W -->|"failed"| R["next window starts<br/>from a standing pose"]
    R --> W
    W -->|"passed twice in a row"| E["20 s exam<br/>from a standing pose"]
    E -->|"failed"| R
    E -->|"passed once"| E
    E -->|"passed twice in a row"| S(["learning stops,<br/>the walk replays"])
```

Every episode, practice or exam, also updates the weights; after the first pass the random search takes smaller steps.

## How we know it is real

`proto/` holds a numpy reference of the same model (`sim.py` physics, `walkbar.py` walking test, `ars.py` learning), and `check_desmos.py` compares the graphs against it. It bakes numpy's random noise into a test graph, lets Desmos run one episode, and compares every body.

- One 5 s episode of all 129 bodies agrees to about 2e-12 in fitness and 2e-14 in the learned update, with the same falls, step counts and verdict.
- For the second graph, a practice window that continues a walk (including the carried pose and clock) and a 20 s exam also match, and the weights learned inside Desmos, re-run in numpy, walk 17.683 m in 20 s against the graph's 17.684 m.
- Rounding differences grow over long walks because the gait is chaotic: they reached 1e-9 to 1e-6 m over 20 s in our checks, and metres within 15 s for one unstable walker in the sweeps. So long walks are compared on falls and verdicts, not to the last digit.

## Results

<p align="center"><img src="evidence/2026-10-22_training-curve_recipe-vs-original.png" alt="Training curves: the chosen recipe reaches the walking bar in about 6,000 simulated steps, the first recipe needs about ten times more" width="820"></p>

| | Result |
|---|---|
| 5 s graph, inside Desmos | Two from-zero runs with different seeds both passed, at generation 17 (5.53 m) and 16 (5.28 m); the second is the saved graph. 4.9 minutes on a 15 W laptop CPU (Intel i7-1255U, no GPU). |
| Long-walk graph, inside Desmos | Passed the 20 s exam twice in a row after 193 episodes and 42 minutes. Its walker covers 17.7 m in 20 s from a standing start. |
| Training recipes | Chosen in numpy, not guessed: 62 configurations for the 5 s recipe (24 of 24 seeds pass, median 5.3k simulated steps, about ten times fewer than the first recipe), and a second sweep for the long walk (22 of 24 seeds, median 36k steps). Tables in `proto/sweep/results_all.txt` and `results_long.txt`. |
| Limits | The learned gait limps (one stance is longer) and the torso pitches a lot. The long walker falls at 42 s, and with its start pose nudged by up to 1 mm, 64% of its 20 s walks still pass. |

### Next: fixing the limp (built and checked, not yet finished inside Desmos)

<p align="center"><img src="evidence/2026-10-22_gait-strip_adopted-recipe-vs-symmetrized-gait-exam_seed0_20s.png" alt="Two 20 second walks compared as filmstrips: the current recipe limps, flies and pitches its torso; the symmetrized recipe steps evenly with the torso nearly upright" width="820"></p>

`gen_desmos.py --gait` adds a third recipe: after the first pass, the random search only explores mirror-symmetric changes (legs swapped, clock shifted half a period), each update pulls the weights toward their mirror image, the torso angle is penalised, and the exam also requires an even gait. In numpy (`proto/sweep/results_gait.txt`) its walkers have stance asymmetry 0.014 instead of 0.37, a mean torso tilt of 7 degrees instead of 27, and 13 of 14 walk the full 60 s; but only 14 of 24 seeds get there within 100k simulated steps. The graph matches numpy episode by episode. One 90-minute run inside Desmos passed one gait exam (25 m in 20 s) but not two in a row, and was stopped.

## Related work in Desmos

This builds on what the Desmos community has already shown is possible: the JellyCar soft-body physics graph (https://www.reddit.com/r/desmos/comments/1rmyv7z/), a policy network trained in a graph (https://www.desmos.com/calculator/khjvuysihr), Q-learning in Desmos (https://www.reddit.com/r/desmos/comments/s6t5z4/), and evolved neural-network dot agents (https://www.reddit.com/r/desmos/comments/1lr1sru/).

## Run it yourself

Requirements: Python 3 with numpy (and matplotlib for the figures).

```bash
python3 proto/ars.py 200 0                                   # train in numpy until the walking test passes (seed 0: generation 13)
python3 proto/render.py theta_s0.npy strip.png               # filmstrip of the learned gait
python3 gen_desmos.py learn.json                             # the 5 s learning graph (the first saved graph)
python3 gen_desmos.py long.json --long --dirs 48 --top 24    # the long-walk graph (the second saved graph)
python3 gen_desmos.py gait.json --gait --dirs 48 --top 24    # the gait recipe
python3 gen_desmos.py walker.json --theta theta_s0.npy --nu 0     # replay a numpy-trained walker in Desmos
python3 check_desmos.py prepare zero 12 test.json expected.json   # an episode to compare against numpy
cd proto/sweep && python3 test_train.py && python3 test_long_train.py   # trainer tests
```

To load a generated file: open https://www.desmos.com/calculator, open the browser console, run `Calc.setState(` followed by the file's contents and `)`, collapse the expression list, and press play on the ticker.

## Repository map

| Path | What it is |
|---|---|
| `gen_desmos.py` | Writes the Desmos graphs (5 s, `--long`, `--gait`). |
| `check_desmos.py` | Builds test episodes with numpy's noise baked in, and the numpy results to compare with. |
| `proto/` | The numpy reference: `sim.py` (physics), `walkbar.py` (walking test), `ars.py` (learning), `render.py` (filmstrips). |
| `proto/sweep/` | The recipe sweeps behind every training choice, their result tables, tests, and the trained walkers. |
| `evidence/` | Screenshots, filmstrips and curves; `evidence/LOG.md` gives the numbers behind every claim. |
| `docs/`, `tools/` | The README figures and the script that draws the body diagram. |

## Things we learned about the Desmos engine

- After fusing the physics into a few vectorised functions, the cost per tick grows with list length (about 7, 5, 4.3 and 3 ticks per second at 65, 97, 129 and 193 bodies on this laptop). Run several physics steps per tick, and keep the state written back by actions small.
- `random(n, seed)` is reproducible only together with the graph's top-level `randomSeed`. A state loaded without one gets a fresh seed on every load.
- A browser tab that has loaded several heavy graphs can run three times slower than a freshly loaded one; a background tab, or a screen recorder, slows it much more.
- The ticker's elapsed-time variable is `\operatorname{dt}`; plain `dt` parses as d times t. `\tau` is a built-in constant, so it cannot be a slider. Any one-variable function definition gets plotted, so hide helpers.
- Multi-assignment actions are simultaneous, conditional actions work (`\left\{k<T:A_{1},A_{2}\right\}`), and a number written next to a bracket (`0.08[1...n]`) parses as indexing; write `0.08\cdot[1...n]`.

## License

MIT, see [LICENSE](LICENSE). Author: Shifat Islam Santo ([@oneKn8](https://github.com/oneKn8)).
