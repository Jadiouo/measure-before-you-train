# Provenance of the predictions

This repository is a clean export and carries none of the private development history. The table below lists, for each prediction file and for the results that answered it, the commit abbreviation in the author's private repository, the author time (UTC+8) and the commit title. The times come from the author's own machine clock; no third party timestamped them. The hashes refer to the private repository, which readers of this repository cannot open. The table is therefore the author's statement of the order of events, not independently verifiable proof.

The prediction files are in [`predictions/`](predictions/). They are copies of the versions stored in the prediction commits in which a few passages are masked (internal role wording about the author, and local resource settings; see [`predictions/README.md`](predictions/README.md)). The column "blob of the unmasked original" gives the abbreviated git blob identifier (`git rev-parse <commit>:<path>` in the private repository) of the original file, so the author can show that an unmasked original matches. Per-claim outcomes are in the paper appendix and the supplement.

| Family | Prediction file (this repo) | blob of the unmasked original | Prediction commit (hash, author time, subject) | Result commit (hash, author time, subject) | Gap |
|---|---|---|---|---|---|
| Birds v2 | [`predictions/01_birds_v2.md`](predictions/01_birds_v2.md) | `a66324d692` | `2838e68`, 2026-10-31T14:21:54+08:00, survival v2 prereg: cheaper swap + observable heterogeneity | `98937d9`, 2026-10-31T19:56:45+08:00, survival v2 RL seeds per prereg: A2 sustains 3/3 but energy-blind; IPPO/alpha0 collapse | 5 h 35 min |
| Rotation, basic | [`predictions/02_rotation_basic.md`](predictions/02_rotation_basic.md) | `2d9b977819` | `a0619ab`, 2026-10-31T22:38:54+08:00, drone bucket model: prediction | `a7e76aa`, 2026-10-31T23:09:18+08:00, Drone energy-bucket model: time rotation loses up to 12% under heterogeneity; energy gate near-optimal | 31 min |
| Rotation, richer | [`predictions/03_rotation_richer.md`](predictions/03_rotation_richer.md) | `b898594e6a` | `79b4dbc`, 2026-11-01T11:15:13+08:00, drone L1: prediction | `3c774e7`, 2026-11-01T13:26:38+08:00, Drone L1: positional savings; best rule within 2% of LP bound in 767/768 cells | 2 h 11 min |
| Rotation, noisy | [`predictions/04_rotation_noisy.md`](predictions/04_rotation_noisy.md) | `f73767f2cf` | `e8d6059`, 2026-11-01T13:28:52+08:00, drone L2/L3: prediction | `ae62fc8`, 2026-11-01T15:00:02+08:00, Drone L2/L3: no RL headroom in rotation scheduling; L3 gap is a bubble-rule artifact (local rot closes it) | 1 h 32 min |
| Wake surfing | [`predictions/05_wake_surfing.md`](predictions/05_wake_surfing.md) | `baa225357e` | `f49e6e8`, 2026-11-01T19:59:12+08:00, wake headroom: prediction | `3437930`, 2026-11-01T20:05:40+08:00, Wake-surfing headroom: classical KF setpoint within 2pp of privileged bound everywhere (with matched GPS) | 6 min |
| Soaring, round 1 | [`predictions/06_soaring_round1.md`](predictions/06_soaring_round1.md) | `e17ba516d7` | `071574f`, 2026-11-01T20:24:36+08:00, soaring headroom: prediction | `a477e3b`, 2026-11-01T22:48:33+08:00, soaring headroom: results (reference is map-omniscient; gap mostly information) | 2 h 24 min |
| Soaring, round 2 | [`predictions/07_soaring_round2.md`](predictions/07_soaring_round2.md) | `08437721fe` | `4eacb94`, 2026-11-01T22:58:50+08:00, soaring headroom 2: prediction | `3219473`, 2026-11-01T23:27:16+08:00, soaring headroom 2: results (no info-matched method beats shared rule) | 29 min |
| Soaring, round 3 | [`predictions/08_soaring_round3.md`](predictions/08_soaring_round3.md) | `912c705a5a` | `02dc53f`, 2026-11-02T10:44:29+08:00, soaring headroom 3: prediction | `7c215c2`, 2026-11-02T11:04:23+08:00, soaring headroom 3: decomposition; gap is information, line closed | 20 min |
| Language-model commander | [`predictions/09_llm_commander.md`](predictions/09_llm_commander.md) | `3bf5583f09` | `5c53a88`, 2026-11-02T23:29:27+08:00, llm commander: prediction | `f1c439d`, 2026-11-02T23:32:02+08:00, llm commander: recorded Haiku/Sonnet answers and scores (14/14 compliant both) | 3 min |
| Neighbours-only learner, first | [`predictions/10_learner_first.md`](predictions/10_learner_first.md) | `aa0facd6c8` | `38ce89e`, 2026-11-04T21:33:57+08:00, prereg: L3 MAPPO validation | `74245e9`, 2026-11-06T00:18:26+08:00, l3 mappo: positive control failed at U=500/1000/2000; L3 main run not executed per prereg | 26 h 45 min |
| Neighbours-only learner, second | [`predictions/11_learner_second.md`](predictions/11_learner_second.md) | `ae3b8211af` | `0e16639`, 2026-11-06T00:22:11+08:00, prereg v2: L3 MAPPO exploration fix | `040f391`, 2026-11-06T14:06:23+08:00, l3 mappo v2: positive control not passed (1/3 cells at U=2000); gap unresolved | 13 h 44 min |

Gap is the result author time minus the prediction author time, both cut to the minute, which is how the paper's timeline table computes it; the two Gap columns agree. For Rotation, basic, the exact difference is 30 min 24 s.

Commit titles are quoted as they were written at the time; they are not statements of the paper. One was later corrected. The title of `ae62fc8` says that rotation scheduling has no RL headroom and that the neighbours-only (L3) gap is a rule artifact. The paper claims neither: it reports the neighbours-only gap as real in the 23 selected settings and unresolved (Section 4.3).

## Predictions that were not blind

Facts as recorded in the project's scorecard notes, family by family.

- Birds v2: the prediction file says it is not blind. The model change followed the author's reading of the v1 results, and the thresholds were set after seeing script baselines.
- Rotation, basic: the file states that no code had been run. The prediction covered 192 settings and 192 were measured.
- Rotation, richer: the file says only earlier model results and a note on horseshoe vortices had been read. It predicted for 576 settings; 768 were measured, because a further +-5% individual-difference level was added during implementation.
- Rotation, noisy: the prediction commit came 2 minutes after the richer-rotation result commit, and the file says only that result had been read. Predicted and measured grids are the same, 160 settings.
- Wake surfing: the file is headed as written before any code was run. The same-GPS reference was a diagnostic added later and was not part of the prediction.
- Soaring, round 1: the file says the model had not been run. The result notes record that the rule was changed three times after seeing results (centring shift, shared rule, leave threshold), so the tuned rule was not predicted blind.
- Soaring, round 2: written after two small trial runs on a quarter-size grid; the file says its writer had seen rough numbers from them, and that the predictions were not adjusted because of them.
- Soaring, round 3: written after the round-2 results were seen, and the decomposition design was chosen after seeing round-2 data.
- Language-model commander: the file says only the overall oracle size had been seen, not any model output; the results notes add that it was written after seeing that the energy gate was almost always best. The two answering models were told not to use tools but each made one tool call, whose content was not inspected.
- Neighbours-only learner, first: the file says a wiring-debug run of at most 20 updates had been seen, and no trained results.
- Neighbours-only learner, second: written after the first attempt failed, with per-seed results and training logs of the first attempt in view; it was the one further attempt.

## What a commit does and does not freeze

From `git show --stat` of each prediction commit (files added):

- Birds v2: the prediction text only. No training script, and no grid file.
- Rotation basic, richer, noisy; Wake surfing; Soaring rounds 1, 2, 3: the prediction text only. The scripts that produced the results first appear in the result commits, so neither code nor grid was frozen by the prediction commit.
- Language-model commander: the prediction, plus `instructions.json` (the 14 instructions) and `prompt_template.md`. The scoring script is not in the commit. The instructions and prompt are frozen; the code is not.
- Neighbours-only learner, first: the prediction (preregistration) together with the training code (`drone_l3_jax.py`, `l3_mappo.py`), the cell list (`l3_cells.py`) and a test. Code and cell grid were committed with it.
- Neighbours-only learner, second: the preregistration plus a 9-line edit to `scripts/l3_mappo_vmap.py`; no new cell list. The training code change is committed, the grid file is not touched.

A prediction commit freezes the text of the prediction and whatever else it contains; it does not show that the code and grids used later were the ones the prediction had in mind.
