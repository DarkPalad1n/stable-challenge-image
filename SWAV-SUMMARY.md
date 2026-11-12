# SwAV track: summary of all experiments (as of 2026-11-12)

Pascal's SwAV track for the image challenge. This file collects everything done so far in one
place: setup, how we measure, every ablation, and the leaderboard uploads. `PROGRESS.md` holds the
detailed log of the laptop phase (up to 2026-11-10); the Hydra results below are not in it yet.

## Short version

- **Best configuration: batch 128, 200 epochs**, all other settings at their defaults. Local
  challenge score 0.683 +/- 0.009 over seeds 0-5 (untrained encoder: 0.386 +/- 0.011; paired gain
  +29.8 +/- 1.7 points).
- **Uploaded** on 2026-11-12: seed 4 of that configuration (local score 0.698). The leaderboard has
  not refreshed yet, so there is no official score.
- **What helps:** longer training and batch 128, and the two add up. From 30 to 200 epochs at batch
  128 the score rises by 17 points.
- **What hurts:** stronger colour jitter (-2.3), blur (-1.4), learning rate 1e-3 instead of 5e-4
  (-2.2), a feature queue (-3.6, laptop phase).
- **No measurable effect:** 100 or 1000 prototypes instead of 300; batch 256 at 30 epochs.
- **Weakest domain:** Galaxy10 (0.52 in-domain, 0.43 few-shot, against 0.73-0.85 for the others).

## 1. Setup

| Setting | Value |
|---|---|
| Encoder | ResNet9, 1024-dim output, 3.49M parameters, 13.3 MiB ONNX (cap 20 MiB) |
| Head | Projector 1024-1024-128, then prototypes (default 300) |
| Crops | 2 global (96 px, scale 0.14-1) + 4 local (48 px, scale 0.05-0.14) |
| Augmentation | Colour jitter 0.4, 0.4, 0.2, 0.1 (p = 0.8), grayscale; no blur, no solarize |
| SwAV | Sinkhorn epsilon 0.05, 3 iterations, temperature 0.1; prototypes frozen in epoch 1; targets from the global crops only |
| Optimiser | AdamW, learning rate 5e-4, weight decay 0.05, constant (no schedule) |
| Defaults in `main.py` | Batch 32, 10 epochs, seed 0 |
| Data | `data/train.parquet` only, labels never in the loss, no pretrained weights |

## 2. How we measure

- **Challenge score** (`challenge_score.py`, main measure): mirrors the pre-release evaluator. Mean of
  an in-domain part (ridge probe per domain, leave-one-out) and a few-shot part (10 images per class,
  20 fixed episodes), each averaged over EuroSAT, Galaxy10 and Imagenette. All classes were seen in
  training, so the few-shot part is not a test on new classes as on the leaderboard.
- **`main.py` score** (`evaluate.py`): one leave-one-out ridge probe over all 750 labelled images.
  Reported for completeness. It ranks some runs differently because it does not average per domain.
- **Seeds:** every configuration is trained with seeds 0-5 (laptop phase: 0-2).
- **Comparisons:** difference of means with standard error SE = sqrt(s1²/6 + s2²/6). We call a
  difference real when it is at least about 2 SE. A single run varies by about +/-1 point.
- **Last epoch only:** we never pick the best epoch, since that would use labels for model selection.

## 3. Phase 1: laptop (MacBook, MPS), 2026-11-06 to 2026-11-10

Three seeds each, batch 32, 30 epochs unless noted. Scores are `main.py` scores (see `PROGRESS.md`).

| Run | `main.py` score | Challenge score | Result |
|---|---:|---:|---|
| Untrained ResNet9 (floor) | 0.374 | 0.393 | Reference |
| SwAV baseline | 0.518 +/- 0.003 | 0.481 | +14.4 over the floor |
| Queue of 480 from epoch 2 | 0.483 +/- 0.013 | 0.457 | -3.6, lower for every seed: queue stays off |
| Paper augmentation (jitter 0.8, 0.8, 0.8, 0.2 + blur 0.5) | 0.457 +/- 0.032 | 0.455 | -6.1, lower for every seed: defaults stay |

The laptop and Hydra runs are not bit-identical (different hardware), so we compare only within
one phase.

## 4. Phase 2: Brown Hydra cluster, 2026-11-10 to 2026-11-12

One configuration per job array, seeds 0-5 in parallel, one GPU per seed. All runs use the same
training code (commits `757c0cd` and `b684337` differ only in `challenge_score.py`'s thread count).
Scoring ran afterwards in one CPU job. Batch 128 does not fit on an RTX 2080 Ti (11 GB) and ran on
Titan RTX (24 GB).

### All configurations (means over seeds 0-5)

| Configuration | Changed setting | Challenge score | In-domain | Few-shot | EuroSAT id / fs | Galaxy10 id / fs | Imagenette id / fs | `main.py` score | GPU |
|---|---|---:|---:|---:|---|---|---|---:|---|
| `b128-e200` | batch 128, 200 epochs | **0.683 +/- 0.009** | 0.703 | 0.664 | 0.850 / 0.820 | 0.524 / 0.435 | 0.734 / 0.738 | not read | Titan RTX |
| `b128-e100` | batch 128, 100 epochs | 0.612 +/- 0.009 | 0.647 | 0.577 | 0.823 / 0.782 | 0.495 / 0.357 | 0.624 / 0.591 | 0.625 | Titan RTX |
| `e200` | 200 epochs | 0.592 +/- 0.011 | 0.619 | 0.564 | 0.806 / 0.765 | 0.415 / 0.326 | 0.636 / 0.602 | 0.625 | RTX 2080 Ti |
| `e100` | 100 epochs | 0.548 +/- 0.012 | 0.593 | 0.502 | 0.778 / 0.727 | 0.435 / 0.291 | 0.566 / 0.489 | 0.594 | RTX 2080 Ti |
| `b128-e30` | batch 128 | 0.511 +/- 0.012 | 0.559 | 0.463 | 0.728 / 0.697 | 0.463 / 0.298 | 0.486 / 0.394 | 0.515 | Titan RTX |
| `b256-e30` | batch 256 | 0.499 +/- 0.013 | 0.540 | 0.457 | 0.713 / 0.690 | 0.449 / 0.317 | 0.458 / 0.365 | 0.513 | L40S |
| `baseline-e30` | none (batch 32, 30 epochs) | 0.497 +/- 0.008 | 0.549 | 0.446 | 0.719 / 0.686 | 0.420 / 0.272 | 0.508 / 0.379 | 0.522 | RTX 2080 Ti |
| `k100-e30` | 100 prototypes | 0.491 +/- 0.012 | 0.540 | 0.442 | 0.716 / 0.677 | 0.409 / 0.269 | 0.497 / 0.380 | 0.509 | RTX 2080 Ti |
| `b128-lr1e-3-e30` | batch 128, learning rate 1e-3 | 0.489 +/- 0.008 | 0.540 | 0.438 | 0.709 / 0.685 | 0.461 / 0.281 | 0.450 / 0.347 | not read | Titan RTX |
| `k1000-e30` | 1000 prototypes | 0.487 +/- 0.017 | 0.539 | 0.436 | 0.715 / 0.674 | 0.425 / 0.266 | 0.476 / 0.367 | 0.510 | RTX 2080 Ti |
| `blur-e30` | blur p = 0.5 | 0.484 +/- 0.007 | 0.543 | 0.424 | 0.719 / 0.662 | 0.413 / 0.254 | 0.497 / 0.356 | 0.513 | RTX 2080 Ti |
| `jitter-e30` | jitter 0.8, 0.8, 0.8, 0.2 | 0.475 +/- 0.010 | 0.530 | 0.419 | 0.694 / 0.638 | 0.423 / 0.266 | 0.472 / 0.354 | 0.495 | RTX 2080 Ti |

"+/-" is the standard deviation over the 6 seeds. "id / fs" = in-domain / few-shot.

Floor: the untrained ResNet9 for seeds 0-5 (`make_random_init.py`, CPU, so the same weights as the
start of every Hydra run of that seed) scores 0.386 +/- 0.011 (challenge) and 0.366 (`main.py`). For
seeds 0-2 the files are byte-identical to the laptop ones.

### Ablations, one factor at a time (challenge score, difference in points)

| Question | Comparison | Difference (SE) | Verdict |
|---|---|---:|---|
| Longer training? | `e100` vs `baseline-e30` | +5.0 (0.6) | Helps; largest gain on Imagenette few-shot (+11.0) |
| | `e200` vs `e100` | +4.4 (0.7) | Still helps |
| Larger batch? | `b128-e30` vs `baseline-e30` | +1.4 (0.6) | Small gain, mostly Galaxy10 in-domain (+4.3) |
| | `b128-e100` vs `e100` | +6.4 (0.6) | Gain grows with training length |
| | `b128-e200` vs `e200` | +9.2 (0.6) | Gain grows further; Galaxy10 +10.9 in both parts |
| | `b256-e30` vs `b128-e30` | -1.2 (0.7) | No measurable difference |
| Longer training at batch 128? | `b128-e100` vs `b128-e30` | +10.1 (0.6) | Helps |
| | `b128-e200` vs `b128-e100` | +7.1 (0.5) | Helps; not yet saturated |
| Learning rate 1e-3 at batch 128? | `b128-lr1e-3-e30` vs `b128-e30` | -2.2 (0.6) | Hurts, mostly Imagenette; keep 5e-4 |
| Number of prototypes? | `k100-e30` vs `baseline-e30` | -0.6 (0.6) | No measurable difference |
| | `k1000-e30` vs `baseline-e30` | -1.0 (0.8) | No measurable difference |
| Blur? | `blur-e30` vs `baseline-e30` | -1.4 (0.5) | Hurts, mostly few-shot |
| Stronger colour jitter? | `jitter-e30` vs `baseline-e30` | -2.3 (0.5) | Hurts, most on EuroSAT few-shot (-4.8) |

Notes on interpretation:
- Batch 128 makes 4 times fewer optimiser steps per epoch than batch 32, so the batch comparisons are
  at equal epochs, not equal steps. Even so, batch 128 wins, and its lead grows with training length.
- The laptop augmentation ablation changed jitter and blur together (-6.1). On Hydra each hurts on
  its own.
- The two scorers disagree on `b128-e100` vs `e200`: equal by `main.py` score (0.625 each), +2.0 for
  `b128-e100` by challenge score. The per-domain average weights Galaxy10, where batch 128 helps
  most, as much as the other domains.
- Not tested: why batch 128 helps, and Galaxy10 most. One guess: Sinkhorn spreads each batch evenly
  over the 300 prototypes, so with 32 images per batch the targets are very soft (laptop phase:
  assignment entropy 4.2-4.4 nats, maximum 5.70). The SwAV paper warns about small batches.

## 5. Leaderboard uploads

Only the latest upload counts. All uploads so far come from one participant ID.

| Date | Model | Local challenge score (seed / configuration mean) | Leaderboard score |
|---|---|---|---|
| 2026-11-11 | `b128-e100`, seed 2 | 0.623 / 0.612 | Not yet published (replaced) |
| 2026-11-12 | `b128-e200`, seed 4 | 0.698 / 0.683 | Pending refresh |

Both files passed `test_submission.py` (13.3 MiB). The chosen seed is the best of six by local score,
so the leaderboard will likely land nearer the configuration mean.

## 6. Open points and next steps

1. **Leaderboard check:** when it refreshes, compare the official score with the local 0.683-0.698.
   This tells us how well `challenge_score.py` predicts the leaderboard.
2. **Retrain budget:** Randall retrains `main.py` offline. `b128-e200` takes 2.7-4.7 hours per seed on
   a Titan RTX. We need the retrain time and hardware limit before training even longer.
3. **`main.py` defaults:** they still describe the old baseline (batch 32, 10 epochs). Before the code
   hand-in they must match the configuration we submit.
4. **Longer training:** the score still rises from 100 to 200 epochs, so 300-400 epochs at batch 128
   are the obvious next test, if the retrain budget allows it.
5. **Galaxy10** is still the weakest domain and a third of the score.
6. **Group:** agree on the final model and on who uploads, since only the latest upload counts.
7. **Reproducibility:** the Hydra job scripts (6-seed array job, summary scripts) live only on the
   cluster so far.

## 7. Practical notes on Hydra

- Job array with one GPU per seed: `--array=0-5`, 4 CPUs, 16 GB RAM.
- RTX 2080 Ti (11 GB) runs batch 32. Batch 128 ran out of memory there and needs a Titan RTX (24 GB)
  or larger; batch 256 ran on an L40S.
- 200-epoch runs used a 16-hour limit. They took 2.7-4.7 hours, but one `e200` seed ran at 110 s
  per epoch on one node instead of about 38 s and was restarted on another node.
- `challenge_score.py` must use only the CPUs the job may use (fixed in `b684337`). Scoring all six
  models in one CPU job takes a few minutes.
- `evaluate.py` (upstream) still starts one thread per node CPU, which slows the evaluations during
  training.

## Appendix: challenge score per seed

| Configuration | s0 | s1 | s2 | s3 | s4 | s5 |
|---|---:|---:|---:|---:|---:|---:|
| `b128-e200` | 0.669 | 0.680 | 0.684 | 0.686 | 0.698 | 0.682 |
| `b128-e100` | 0.621 | 0.614 | 0.623 | 0.606 | 0.599 | 0.609 |
| `e200` | 0.588 | 0.593 | 0.583 | 0.601 | 0.578* | 0.606 |
| `e100` | 0.559 | 0.557 | 0.546 | 0.552 | 0.525 | 0.548 |
| `b128-e30` | 0.502 | 0.530 | 0.520 | 0.510 | 0.509 | 0.495 |
| `b256-e30` | 0.490 | 0.490 | 0.523 | 0.496 | 0.499 | 0.494 |
| `baseline-e30` | 0.487 | 0.500 | 0.489 | 0.509 | 0.497 | 0.503 |
| `k100-e30` | 0.489 | 0.478 | 0.506 | 0.505 | 0.489 | 0.480 |
| `b128-lr1e-3-e30` | 0.494 | 0.487 | 0.477 | 0.484 | 0.492 | 0.499 |
| `k1000-e30` | 0.477 | 0.495 | 0.473 | 0.519 | 0.478 | 0.481 |
| `blur-e30` | 0.474 | 0.483 | 0.484 | 0.492 | 0.491 | 0.477 |
| `jitter-e30` | 0.479 | 0.462 | 0.481 | 0.467 | 0.470 | 0.488 |

\* Restarted on another node after the first attempt ran slowly (same seed and settings).
