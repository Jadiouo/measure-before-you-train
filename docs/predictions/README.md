# Predictions

These 11 files are the prediction (or preregistration) documents of the study. They are copies of the files saved in the commit that introduced each one, with a few passages masked (see below); everything else is as written, so most of the text is in Chinese and typos are original. File numbers follow the order of the paper's timeline table.

## What is masked

Two kinds of text were replaced. Nothing else was changed, including the first person ("I") in the files.

1. **Internal role wording about the author.** The word that the working notes used for the author's role (a Chinese word for "user") is replaced by `〔作者〕` ("author"). The sentences around it are untouched, because several of them are substantive commitments made before a run (for example "only one more attempt").
2. **Local resource settings.** Names and options of the author's local job queue and machine setup (queue names and numbers, memory and GPU-memory limits, the name of a virtual environment, a core count, a monitoring command) are replaced by `〔已遮蔽：本機資源設定〕` ("masked: local resource settings"). Budgets that bear on the prediction itself (number of updates, hour budgets, the budget ladder) were kept.

Every masked place carries one of these two markers. The masks were produced by a script that lives in the author's private repository: it reads the original from the commit, applies an explicit rule table, checks that each rule matched the expected number of times, and checks that putting the original text back at the markers gives the original file byte for byte. The rule table is not published because it contains the masked text.

| File | Role wording masked | Local resource settings masked |
|---|---|---|
| `01_birds_v2.md` | 2 | 1 |
| `02_rotation_basic.md` | 0 | 0 |
| `03_rotation_richer.md` | 0 | 0 |
| `04_rotation_noisy.md` | 0 | 0 |
| `05_wake_surfing.md` | 0 | 0 |
| `06_soaring_round1.md` | 0 | 0 |
| `07_soaring_round2.md` | 0 | 0 |
| `08_soaring_round3.md` | 0 | 0 |
| `09_llm_commander.md` | 0 | 0 |
| `10_learner_first.md` | 1 | 3 |
| `11_learner_second.md` | 3 | 5 |
| Total | 6 | 9 |

The git blob identifier of each unmasked original is listed in [../provenance.md](../provenance.md) (column "blob of the unmasked original"); the author can show an original and a reader can check that it hashes to that identifier.

The commit table is in [../provenance.md](../provenance.md). Claim-by-claim outcomes are in the paper appendix and the supplement.
