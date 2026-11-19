# SwAV track: summary of all experiments (as of 2026-11-19)

Pascal's SwAV track for the image challenge. This file collects everything done so far in one
place: setup, how we measure, every ablation, and the leaderboard uploads. `PROGRESS.md` holds the
detailed log of the laptop phase (up to 2026-11-10); for everything later, this file is the record.

## Short version

- **Best measured configuration: batch 128, constant learning rate, 800 epochs.** Local challenge
  score 0.751 +/- 0.007 over seeds 0-5 (untrained encoder: 0.386 +/- 0.011). 1600 epochs give
  0.759 +/- 0.010, a gain of +0.8 points that is not significant (about 1.6 SE), so 800 epochs is the
  working default from 2026-11-16 on.
- **Plateau:** every long run ends at about 0.75-0.76, whatever we change: 800 or 1600 epochs,
  Sinkhorn epsilon 0.05 or 0.03, larger local crops, or a larger encoder (narrow ResNet18). A cosine
  schedule ends lower (0.737). Two of these changes only speed up the start.
- **Larger encoder:** a narrow ResNet18 (4.37M parameters) gives the same challenge score as the
  ResNet9 (0.751 each) but a higher `main.py` score (0.762 against 0.748, about 3 SE). The two local
  scores disagree here for the first time; an upload on 2026-11-19 is meant to settle it.
- **Larger local crops** (0.14-0.4 of the image instead of 0.05-0.14) do not help (-0.8, not
  significant) and lower Galaxy10 few-shot by 3.9 points (about 2.5 SE), against our prediction.
- **Sinkhorn epsilon:** sharper targets learn faster. At 200 epochs epsilon 0.01-0.03 beats the
  paper's 0.05 by about 3 points (about 7 SE); at 800 epochs the difference is gone (+0.3). Epsilon
  0.1 collapses (0.270, below the untrained encoder), as the SwAV paper warns.
- **Leaderboard:** four scored uploads. The local scores rank them in the same order as the
  leaderboard (Spearman 1.0; Pearson 0.96 for the challenge score, 0.99 for the `main.py` score), but
  the leaderboard compresses differences: one local point is worth about 0.4-0.5 leaderboard points.
  The 1600-epoch upload (seed 0) scored 0.6233; we predicted 0.61-0.63 before the refresh. The
  ResNet18 upload of 2026-11-19 is not scored yet.
- **What hurts:** stronger colour jitter (-2.3), blur (-1.4), learning rate 1e-3 instead of 5e-4
  (-2.2), a feature queue (-3.6, laptop phase), cosine decay instead of a constant rate (-1.4); larger
  local crops for Galaxy10 few-shot (-3.9).
- **No measurable effect:** 100 or 1000 prototypes instead of 300; batch 256 at 30 epochs.
- **Weakest domain:** Galaxy10 (0.52 in-domain, 0.43 few-shot at `b128-e200`, against 0.73-0.85 for
  the others).

## 1. Setup

| Setting | Value |
|---|---|
| Encoder | ResNet9 (`stable_pretraining.backbone.Resnet9`), 1024-dim output, 3.49M parameters, 13.3 MiB ONNX (cap 20 MiB). Option `--backbone resnet18-narrow` (branch `resnet18`): ResNet18 layout with stages of 40-80-160-320 channels, 3x3 stem, BatchNorm momentum 0.9 as in the ResNet9, 320-dim output zero-padded to 1024, 4.37M parameters, 16.7 MiB ONNX (results in 4.2) |
| Head | Projector 1024-1024-128, then prototypes (default 300) |
| Crops | 2 global (96 px, scale 0.14-1) + 4 local (48 px, scale 0.05-0.14; set with `--local-crop-area`) |
| Augmentation | Colour jitter 0.4, 0.4, 0.2, 0.1 (p = 0.8), grayscale; no blur, no solarize |
| SwAV | Sinkhorn epsilon 0.05, 3 iterations (log space), temperature 0.1; prototypes frozen in epoch 1; targets from the global crops only |
| Optimiser | AdamW, learning rate 5e-4, weight decay 0.05, constant. Optional: `--warmup-epochs` and `--lr-schedule cosine` (tested, worse) |
| Defaults in `main.py` | Batch 128, 200 epochs, seed 0; data loading in the main process. To be raised to 800 epochs (section 6) |
| Data | `data/train.parquet` only, labels never in the loss, no pretrained weights |

`main.py` without options trains batch 128 for 200 epochs. It needs a GPU with more than 11 GB of
memory. Every result in this file is repeated by passing its changed settings explicitly; the
30-epoch baseline on Hydra, for example, is `--batch-size 32 --epochs 30 --num-workers 4`.

Loader workers: all Hydra runs used `--num-workers 4`, the laptop runs and the default use 0 (the main
process). The default must run in an unknown retrain environment: on macOS a worker process can abort
when it exits (onnxruntime's telemetry thread, loaded through kornia), which ends the run; this hit
runs with very short epochs. Each worker draws its own augmentations, so a run with the defaults draws
other augmentations than the measured Hydra runs. It is the same recipe, but not the same run, and it
loads the data more slowly.

## 2. How we measure

- **Challenge score** (`challenge_score.py`, main measure): mirrors the pre-release evaluator. Mean of
  an in-domain part (ridge probe per domain, leave-one-out) and a few-shot part (10 images per class,
  20 fixed episodes), each averaged over EuroSAT, Galaxy10 and Imagenette. All images, including those
  of the few-shot classes, were seen (without labels) in training, so the few-shot part is not a test
  on new classes as on the leaderboard.
- **`main.py` score** (`evaluate.py`): one leave-one-out ridge probe over all 750 labelled images.
  It ranks some runs differently because it does not average per domain.
- **Seeds:** every configuration is trained with seeds 0-5 (laptop phase: 0-2).
- **Comparisons:** difference of means with standard error SE = sqrt(s1²/6 + s2²/6). We call a
  difference real when it is at least about 2 SE. A single run varies by about +/-1 point.
- **Last epoch only:** we never pick the best epoch or the best seed by score, since that would use
  labels for model selection. Uploads from 2026-11-16 on use seed 0 by rule.
- **Training curves:** with a constant learning rate, the kept model of epoch N of a long run
  (`--keep-evaluated-models`) stands for a run of N epochs. The `b128-e800` curve confirms this at
  epoch 200: 0.684, against 0.683 for the separate 200-epoch runs. This does not hold for the cosine
  schedule, whose rate depends on the planned length.

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

## 4. Phase 2: Brown Hydra cluster, from 2026-11-10

One configuration per job array, seeds 0-5 in parallel, one GPU per seed. Scoring ran afterwards in
one CPU job. Batch 128 does not fit on an RTX 2080 Ti (11 GB); it ran on Titan RTX (24 GB), L40,
L40S and RTX A6000 (48 GB). The seeds of one configuration sometimes ran on different GPU types.

### 4.1 First wave, 2026-11-10 to 2026-11-12 (means over seeds 0-5)

| Configuration | Changed setting | Challenge score | In-domain | Few-shot | EuroSAT id / fs | Galaxy10 id / fs | Imagenette id / fs | `main.py` score | GPU |
|---|---|---:|---:|---:|---|---|---|---:|---|
| `b128-e200` | batch 128, 200 epochs | **0.683 +/- 0.009** | 0.703 | 0.664 | 0.850 / 0.820 | 0.524 / 0.435 | 0.734 / 0.738 | 0.684 | Titan RTX |
| `b128-e100` | batch 128, 100 epochs | 0.612 +/- 0.009 | 0.647 | 0.577 | 0.823 / 0.782 | 0.495 / 0.357 | 0.624 / 0.591 | 0.625 | Titan RTX |
| `e200` | 200 epochs | 0.592 +/- 0.011 | 0.619 | 0.564 | 0.806 / 0.765 | 0.415 / 0.326 | 0.636 / 0.602 | 0.625 | RTX 2080 Ti |
| `e100` | 100 epochs | 0.548 +/- 0.012 | 0.593 | 0.502 | 0.778 / 0.727 | 0.435 / 0.291 | 0.566 / 0.489 | 0.594 | RTX 2080 Ti |
| `b128-e30` | batch 128 | 0.511 +/- 0.012 | 0.559 | 0.463 | 0.728 / 0.697 | 0.463 / 0.298 | 0.486 / 0.394 | 0.515 | Titan RTX |
| `b256-e30` | batch 256 | 0.499 +/- 0.013 | 0.540 | 0.457 | 0.713 / 0.690 | 0.449 / 0.317 | 0.458 / 0.365 | 0.513 | L40S |
| `baseline-e30` | none (batch 32, 30 epochs) | 0.497 +/- 0.008 | 0.549 | 0.446 | 0.719 / 0.686 | 0.420 / 0.272 | 0.508 / 0.379 | 0.522 | RTX 2080 Ti |
| `k100-e30` | 100 prototypes | 0.491 +/- 0.012 | 0.540 | 0.442 | 0.716 / 0.677 | 0.409 / 0.269 | 0.497 / 0.380 | 0.509 | RTX 2080 Ti |
| `b128-lr1e-3-e30` | batch 128, learning rate 1e-3 | 0.489 +/- 0.008 | 0.540 | 0.438 | 0.709 / 0.685 | 0.461 / 0.281 | 0.450 / 0.347 | 0.501 | Titan RTX |
| `k1000-e30` | 1000 prototypes | 0.487 +/- 0.017 | 0.539 | 0.436 | 0.715 / 0.674 | 0.425 / 0.266 | 0.476 / 0.367 | 0.510 | RTX 2080 Ti |
| `blur-e30` | blur p = 0.5 | 0.484 +/- 0.007 | 0.543 | 0.424 | 0.719 / 0.662 | 0.413 / 0.254 | 0.497 / 0.356 | 0.513 | RTX 2080 Ti |
| `jitter-e30` | jitter 0.8, 0.8, 0.8, 0.2 | 0.475 +/- 0.010 | 0.530 | 0.419 | 0.694 / 0.638 | 0.423 / 0.266 | 0.472 / 0.354 | 0.495 | RTX 2080 Ti |

"+/-" is the standard deviation over the 6 seeds. "id / fs" = in-domain / few-shot.

Floor: the untrained ResNet9 for seeds 0-5 (`make_random_init.py`, CPU, so the same weights as the
start of every Hydra run of that seed) scores 0.386 +/- 0.011 (challenge) and 0.366 (`main.py`). For
seeds 0-2 the files are byte-identical to the laptop ones.

### 4.2 Second wave, 2026-11-12 to 2026-11-19 (all batch 128, means over seeds 0-5)

| Configuration | Changed setting | Challenge score | `main.py` score | GPU |
|---|---|---:|---:|---|
| `b128-e1600` | 1600 epochs | 0.759 +/- 0.010 | 0.754 +/- 0.011 | Titan RTX (s0-2), RTX A6000 (s3-5) |
| `b128-e800-eps0.03` | 800 epochs, epsilon 0.03 | 0.754 +/- 0.005 | 0.746 +/- 0.011 | Titan RTX |
| `b128-e800-r18` | 800 epochs, narrow ResNet18 encoder | 0.751 +/- 0.007 | **0.762 +/- 0.005** | L40 (s0-2), RTX A6000 (s3-5) |
| `b128-e800` | 800 epochs | **0.751 +/- 0.007** | 0.748 +/- 0.009 | Titan RTX (s0-4), L40 (s5) |
| `b128-e800-local0.14-0.4` | 800 epochs, local crops 0.14-0.4 of the image | 0.742 (5 seeds) | not read | L40 (s0-3), Titan RTX (s5) |
| `b128-e800-cos` | 800 epochs, 10 warm-up epochs, cosine decay to 0 | 0.737 +/- 0.009 | 0.731 +/- 0.006 | RTX A6000 (s0-2), L40 (s3-5) |
| `b128-e200-eps0.02` | epsilon 0.02 | 0.715 +/- 0.007 | 0.721 +/- 0.007 | Titan RTX |
| `b128-e200-eps0.03` | epsilon 0.03 | 0.715 +/- 0.007 | 0.714 +/- 0.008 | Titan RTX |
| `b128-e200-eps0.01` | epsilon 0.01 | 0.711 +/- 0.005 | 0.717 +/- 0.011 | RTX A6000, Titan RTX |
| `b128-e200` (from 4.1) | epsilon 0.05 (default) | 0.683 +/- 0.009 | 0.684 +/- 0.008 | Titan RTX |
| `b128-e200-eps0.1` | epsilon 0.1 | 0.270 +/- 0.020 | 0.264 +/- 0.021 | RTX A6000, Titan RTX |

`b128-e800-local0.14-0.4` uses seeds 0-3 and 5. Seed 4 is left out: its cluster job was restarted, so
its run is not certain to be complete.

Untrained narrow ResNet18 (floor, `make_random_init.py SEED resnet18-narrow`, seeds 0-5, the same
weights the runs start from): challenge score 0.366 +/- 0.010, against 0.386 +/- 0.011 for the ResNet9.

Per domain at 800 epochs (challenge score parts, means over seeds; difference to `b128-e800` in points,
SE in brackets):

| Domain, part | `b128-e800` | `b128-e800-local0.14-0.4` (5 seeds) | `b128-e800-r18` |
|---|---:|---:|---:|
| EuroSAT in-domain | 0.883 | 0.881 (-0.1, 0.6) | 0.890 (+0.8, 0.6) |
| EuroSAT few-shot | 0.874 | 0.866 (-0.7, 0.5) | 0.875 (+0.1, 0.4) |
| Galaxy10 in-domain | 0.582 | 0.590 (+0.8, 1.3) | 0.591 (+0.9, 1.1) |
| Galaxy10 few-shot | 0.548 | **0.509 (-3.9, 1.6)** | 0.548 (-0.0, 1.2) |
| Imagenette in-domain | 0.804 | 0.804 (-0.1, 1.0) | 0.810 (+0.5, 0.8) |
| Imagenette few-shot | 0.812 | 0.803 (-0.9, 0.5) | **0.792 (-2.0, 0.3)** |

Per-domain scores of the other second-wave runs were not read.

Training curves (challenge score, means over seeds 0-5, every 50 epochs for the 800-epoch runs and
every 100 for `b128-e1600`):

| Epoch | 50 | 100 | 200 | 300 | 400 | 500 | 600 | 700 | 800 | 1000 | 1200 | 1400 | 1600 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `b128-e800` | 0.535 | 0.604 | 0.684 | 0.717 | 0.730 | 0.738 | 0.743 | 0.748 | 0.751 | | | | |
| `b128-e800-eps0.03` | 0.614 | 0.675 | 0.713 | 0.731 | 0.737 | 0.739 | 0.747 | 0.747 | 0.754 | | | | |
| `b128-e800-cos` | 0.558 | 0.629 | 0.691 | 0.718 | 0.732 | 0.738 | 0.737 | 0.737 | 0.737 | | | | |
| `b128-e800-r18` | 0.539 | 0.604 | 0.677 | 0.710 | 0.728 | 0.734 | 0.746 | 0.747 | 0.751 | | | | |
| `b128-e1600` | | 0.604 | 0.683 | 0.713 | 0.729 | 0.738 | 0.744 | 0.745 | 0.749 | 0.751 | 0.756 | 0.762 | 0.759 |

The cosine curve's points before epoch 800 are not runs of that length: its learning rate depends on
the planned 800 epochs.

### 4.3 Ablations, one factor at a time (challenge score, difference in points)

| Question | Comparison | Difference (SE) | Verdict |
|---|---|---:|---|
| Longer training? | `e100` vs `baseline-e30` | +5.0 (0.6) | Helps; largest gain on Imagenette few-shot (+11.0) |
| | `e200` vs `e100` | +4.4 (0.7) | Still helps |
| Larger batch? | `b128-e30` vs `baseline-e30` | +1.4 (0.6) | Small gain, mostly Galaxy10 in-domain (+4.3) |
| | `b128-e100` vs `e100` | +6.4 (0.6) | Gain grows with training length |
| | `b128-e200` vs `e200` | +9.2 (0.6) | Gain grows further; Galaxy10 +10.9 in both parts |
| | `b256-e30` vs `b128-e30` | -1.2 (0.7) | No measurable difference |
| Longer training at batch 128? | `b128-e100` vs `b128-e30` | +10.1 (0.6) | Helps |
| | `b128-e200` vs `b128-e100` | +7.1 (0.5) | Helps |
| | `b128-e800` vs `b128-e200` | +6.8 (0.5) | Helps; the curve flattens after about 400-500 epochs |
| | `b128-e1600` vs `b128-e800` | +0.8 (0.5) | Not significant (`main.py` score: +0.6, SE 0.6) |
| Learning rate 1e-3 at batch 128? | `b128-lr1e-3-e30` vs `b128-e30` | -2.2 (0.6) | Hurts, mostly Imagenette; keep 5e-4 |
| Cosine schedule? | `b128-e800-cos` vs `b128-e800` | -1.4 (0.5) | Hurts (`main.py` score: -1.7, SE 0.4); ahead early, flat from epoch 400 |
| Sinkhorn epsilon at 200 epochs? | `eps0.03` vs `b128-e200` (0.05) | +3.2 (0.5) | Helps |
| | `eps0.02` vs `b128-e200` | +3.2 (0.5) | Helps; same as 0.03 |
| | `eps0.01` vs `b128-e200` | +2.8 (0.4) | Helps; same as 0.02 and 0.03 within noise |
| | `eps0.1` vs `b128-e200` | -41.3 (0.9) | Collapse, below the untrained encoder |
| Sinkhorn epsilon at 800 epochs? | `b128-e800-eps0.03` vs `b128-e800` | +0.3 (0.4) | No measurable difference |
| Larger local crops? | `b128-e800-local0.14-0.4` (5 seeds) vs `b128-e800` | -0.8 (0.5) | No measurable difference overall; Galaxy10 few-shot -3.9 (1.6) |
| Larger encoder? | `b128-e800-r18` vs `b128-e800` | +0.1 (0.4) | No difference by challenge score; `main.py` score +1.4 (0.4) |
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
- **Epsilon.** The SwAV paper sets epsilon to 0.05 for all runs and does not vary it; it states that a
  high epsilon "generally leads to a trivial solution" in which all samples collapse, and therefore
  keeps epsilon low (Caron et al. 2020, section 3.1 and appendix A.1). Our sweep confirms the collapse
  at 0.1 and adds that 0.01-0.03 learn faster than 0.05. The 800-epoch curves show that the gain is
  speed, not a higher end point: epsilon 0.03 leads by +7.9 at epoch 50, +2.9 at 200 and +0.7 at 400,
  and both runs end at 0.75. `sinkhorn()` in `swav.py` works in log space, so a small epsilon cannot
  overflow.
- **Cosine.** The warm-up plus cosine schedule leads early (+2.3 at epoch 50, probably the warm-up;
  not tested separately) and stops improving at about epoch 400, while the constant rate still gains
  2 points. A guess, not tested: the model still learns at epoch 400, and cosine lowers the rate
  towards 0 before it is done.
- **Local crops.** Hypothesis: 8 of the 10 loss terms predict a target from a local crop, and at
  5-14% of a 96-pixel image a local crop covers only 21-36 pixels (in the paper's 224-pixel images,
  50-84), so it may miss the structure, above all of the galaxies. Prediction, fixed before the run:
  larger local crops raise the end point by at least 2 SE, most on Galaxy10. Result: the run leads
  early (curve over all six seeds: +4.8 at epoch 50) and ends 0.8 lower, and Galaxy10 few-shot drops
  by 3.9 points. The prediction failed in the opposite direction. A guess, not tested: predicting the
  whole image from a small crop is what makes the encoder learn the fine galaxy structure; larger
  local crops overlap the range of the global crops (0.14-1) and make the task easier.
- **Larger encoder.** Prediction, fixed before the run: if capacity sets the limit, the narrow
  ResNet18 ends at least 2 SE above 0.751. Result: 0.751, the same; its curve is a little slower
  early and meets the ResNet9 curve from epoch 400 on. The `main.py` score, however, rises by 1.4
  points (about 3 SE). The per-domain table shows why the two scores disagree: the ResNet18 is
  slightly better in-domain everywhere, but 2.0 points worse on Imagenette few-shot, and the challenge
  score gives few-shot half the weight. A guess, not tested: the ResNet18 outputs 320 features instead
  of 1024, which may matter for a probe fit on 10 images per class.
- **Plateau.** No setting tried so far ends above about 0.76: neither twice the epochs, sharper
  targets, larger local crops nor a larger encoder. Our reading is that the 7,500 training images or
  SwAV itself set the limit at this size. This is an inference, not a measured result.

### 4.4 Diagnostics (`swav_diagnostics.py`)

The script loads a run's `checkpoint.pt` and reports the entropy of the Sinkhorn targets and how the
training images spread over the 300 prototypes, overall and per domain (labels are used for this
split only). Run on the six `e200` (batch 32) and six `b128-e200` checkpoints, in eval mode, means
over seeds:

| | `e200` (batch 32) | `b128-e200` (batch 128) |
|---|---:|---:|
| Prototypes used (of 300) | 245 | 296 |
| Effective number of prototypes | 143 | 239 |
| Prototypes used by the labelled Galaxy10 images | 62 | 98 |
| Target entropy, measured at batch 32 (nats) | 3.26 | 2.84 |
| Target entropy, measured at batch 128 (nats) | 3.08 | 2.24 |

The batch-128 models spread the images over more prototypes and produce sharper targets. The
laptop observation that Galaxy10 shares only 4 prototypes (one 30-epoch run) does not hold for the
200-epoch models. The epsilon sweep (4.3) then changed the target sharpness directly: sharper targets
speed up learning, but do not raise the end point.

## 5. Leaderboard uploads

Only the latest upload counts. All uploads so far come from one participant ID. `uploads.json` holds
one record per upload (file hash, both local scores, the leaderboard score once published);
`proxy_validation.py` writes it and prints the correlation of the local scores with the leaderboard.

| Date | Model | Local challenge score | Local `main.py` score | Leaderboard score |
|---|---|---:|---:|---:|
| 2026-11-11 | `b128-e100`, seed 2 | 0.623 | 0.647 | Not scored (replaced before the refresh) |
| 2026-11-12 | `b128-e200`, seed 4 | 0.698 | 0.680 | 0.5747 |
| 2026-11-13 | `b128-e800`, seed 5 | 0.761 | 0.744 | 0.6157 |
| 2026-11-14 | `b128-e800`, seed 5, kept model of epoch 100 | 0.592 | 0.616 | 0.5520 |
| 2026-11-16 | `b128-e1600`, seed 0 | 0.766 | 0.768 | 0.6233 |
| 2026-11-19 | `b128-e800-r18`, seed 0 (narrow ResNet18) | 0.755 | 0.765 | Pending |

All files passed `test_submission.py` (13.3 MiB; the ResNet18 16.7 MiB). The 2026-11-12 seed was the best of six by local
score; the 2026-11-13 seed was the first to finish; from 2026-11-16 on we upload seed 0 by rule. The
epoch-100 model was uploaded on purpose as a weaker point, to spread the local scores for the
correlation.

Correlation over the four scored uploads (`proxy_validation.py report`):

| Local score | Pearson | Spearman | Slope (leaderboard points per local point) |
|---|---:|---:|---:|
| Challenge score | 0.957 | 1.000 | 0.40 |
| `main.py` score | 0.992 | 1.000 | 0.49 |

- The local scores rank all four uploads in the leaderboard's order.
- The leaderboard compresses differences: a local gain of 1 point is worth about 0.4-0.5 leaderboard
  points. In absolute terms the local challenge scores are higher, by 4 points for the weakest upload
  and 12-15 points for the others. A likely reason: the leaderboard tests held-out images and new
  classes, our scorer tests images seen in training.
- Before the 2026-11-17 refresh we predicted 0.61-0.63 for the 1600-epoch upload from a line through
  the first three points; it scored 0.6233.
- For the ResNet18 upload the two lines predict different scores, fixed before the refresh: 0.612
  from the challenge score, 0.623 from the `main.py` score. The gap is only about 1 point, since seed 0
  happens to have a `main.py` score close to the previous upload.
- Four points are weak evidence. More points, for example from the group's other methods scored with
  the same `challenge_score.py`, would make the check stronger.

## 6. Open points and next steps

1. **Encoder choice.** The narrow ResNet18 (branch `resnet18`) is done and tested; `main` keeps the
   ResNet9. By challenge score both are equal, by `main.py` score the ResNet18 is better. The
   leaderboard score of the 2026-11-19 upload decides which local score to trust here. The group's DINO
   track describes the same layout; sharing the exact code with the group is still open.
2. **`main.py` defaults:** still batch 128 and 200 epochs. To change to 800 epochs (decision
   2026-11-16). Epsilon can stay at the paper's 0.05, since 0.03 makes no difference at 800 epochs.
   Still to do: confirm the recipe exactly as handed in (defaults, so no loader workers) once with
   seeds 0-5 on Hydra.
3. **Retrain budget:** unknown; treated as no constraint for now (decision 2026-11-12). 800 epochs take
   about 4.5 hours on an L40 or RTX A6000 and 8.5-9 hours on a Titan RTX. If the budget turns out to
   be tight, epsilon 0.03 reaches the same level in fewer epochs (0.746 at epoch 550).
4. **Proxy:** our scorer tests images seen in training. A held-out-image variant (pretrain on 5/6 of
   the images, probe on the rest, rotating over 6 folds instead of 6 seeds) would match the
   leaderboard better at the same cost. New classes cannot be simulated: 90% of the images have no
   label, so we cannot remove a class from training.
5. **Galaxy10** is still the weakest domain and a third of the score (0.582 in-domain, 0.548 few-shot
   at 800 epochs). Neither the larger encoder nor larger local crops help it; the larger crops hurt its
   few-shot part.
6. **Group:** one shared encoder, one scorer and one `main.py`; agree on the final model and on who
   uploads, since only the latest upload counts. The best model must be uploaded again last.
7. **Step-matched batch control:** open. Batch 32 for 200 epochs makes as many optimiser steps as
   batch 128 for 800.

## 7. Practical notes on Hydra

- `sbatch cluster/train_seeds.sh NAME [main.py options]` trains seeds 0-5 as a job array with one GPU
  per seed (4 CPUs, 16 GB RAM, `--num-workers 4`, an evaluation every 5 epochs) and writes
  `runs/hydra-NAME-sSEED/`. `sbatch cluster/score_models.sh NAME` then scores the six models in one CPU
  job (`NAME curve`: every kept model), and `uv run cluster/summarize.py` prints mean, standard
  deviation, minimum and maximum per configuration (`--curve NAME`: per evaluated epoch).
- GPUs for batch 128: Titan RTX (24 GB, about 38 s per epoch), L40 (about 17 s), RTX A6000 (about
  20 s); the L40, L40S and A6000 are often fully booked. The RTX 2080 Ti (11 GB) runs out of memory
  at batch 128; the GTX 1080 and 1080 Ti are not supported by the installed PyTorch. Check free GPUs
  with `sinfo -p gpus -N -O "NodeList:10,Gres:45,GresUsed:55,StateCompact:8"` and pass `--gres=...`
  on the command line; split the seeds over GPU types with `--array` if needed.
- Long runs need `--time` (30 hours for 800 epochs, 40 for 1600). Some runs slowed to 2-4 times
  their usual time per epoch on node `gpu2001` and repeatedly for array task 0; `--exclude=gpu2001`
  avoids the node. `main.py` cannot resume, so a slow run has to finish or restart from epoch 0.
- A job pending with `Reason=Priority` waits for higher-priority jobs, even when a GPU is idle.
- `challenge_score.py` and the evaluations during training use only the job's CPUs (`evaluate.py`
  upstream starts one thread per node CPU). Scoring six final models takes about 3 minutes, a full
  curve 15-30 minutes.

## Appendix: challenge score per seed (first wave)

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

Per-seed scores of the second wave are in `runs/hydra-challenge-scores-NAME.json` on Hydra.
