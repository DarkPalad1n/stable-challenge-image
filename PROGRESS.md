# SwAV progress log

Pascal's SwAV track for the image challenge.
Update this file after every experiment.

## Goal
Train a ResNet9 encoder with SwAV on `data/train.parquet` and export it to ONNX
(20 MiB cap). Compare it with the group's other methods using the same encoder,
training budget, seeds and scorer.

## Group
| Person | Method |
|---|---|
| Peter | LeJEPA |
| Chris | DINO |
| Patrick | VISReg |
| Pascal | SwAV |

## Rules we follow
- Train on `train.parquet` only: no outside data and no pretrained weights.
  Randall confirmed in class on 2026-11-10: no pretrained backbone. Ours starts
  from random weights (`spt.backbone.Resnet9` loads nothing).
- Use labels for monitoring only, never in the training objective.
- The final `main.py` must run offline from the data alone.
- Change one setting at a time and use at least 3 seeds per setting.

## Done
- 2026-11-06: Forked the repo and set up the environment with `uv sync`.
- 2026-11-06: Added `make_random_init.py` and measured the score floor.
- 2026-11-06: Added `swav_data.py` with 2 global crops (96 px) and 4 local
  crops (48 px).
- 2026-11-08: Added `swav.py` (model, Sinkhorn-Knopp, swapped-prediction loss)
  and `main.py` (seeded training, ONNX export, probe score per epoch).
  `swav_data.py` now returns unlabelled images and a seeded loader.
- 2026-11-08: Added tests for all three files. Sinkhorn matches
  `distributed_sinkhorn` from the official SwAV code.
- 2026-11-08: Baseline: 3 seeds x 30 epochs, 0.518 mean, 14.4 points over the
  floor.
- 2026-11-08: Added an optional feature queue (`--queue-length`,
  `--queue-start-epoch`). Ablation: a queue of 480 lowers the score by 3.6
  points, so it stays off.
- 2026-11-09: Split `main.py`: it now only reads the settings and runs the
  epochs; device, seeding, one training epoch and evaluation live in
  `training.py`. Checked: trained weights are bit-identical to before.
- 2026-11-09: Made colour jitter and blur configurable (`--color-jitter`,
  `--blur-probability`). Ablation: the paper's augmentation lowers the score
  by 6.1 points, so the defaults stay.

## Decisions
| Setting | Value | Why |
|---|---|---|
| Encoder | ResNet9, 1024 dims | 3.49M parameters, 13.3 MiB ONNX, under the cap |
| Global crops | 2 at 96 px, scale 0.14-1 | SwAV paper's scales |
| Local crops | 4 at 48 px, scale 0.05-0.14 | Paper's scales, size scaled down |
| Colour jitter | 0.4, 0.4, 0.2, 0.1, p=0.8 | stable-pretraining benchmark; the paper's 0.8, 0.8, 0.8, 0.2 plus blur cost 6.1 points |
| Blur | Off (`--blur-probability 0`) | Tested only together with stronger jitter (see above) |
| Solarize | Off | Not tested |
| Prototypes | 300 | Starting choice for 30 classes, 7,500 images |
| Epsilon, iterations, temperature | 0.05, 3, 0.1 | SwAV paper |
| Batch size | 32 | Fits laptop training (MPS), 86 s per epoch |
| Optimizer | AdamW, 5e-4, weight decay 0.05, no schedule | stable-pretraining benchmark |
| Prototype freezing | First epoch | SwAV paper |
| Queue | Off (`--queue-length 0`) | A queue of 480 cost 3.6 points over 3 seeds |
| Epochs | 30 | Score still rising at 30 |

## Run log
Local score = `evaluate.py` on `data/train.parquet` (leave-one-out on labelled
images). It does not cover held-out images or new classes.

| Date | Commit | Run | Seed | Local score | Upload score | Notes |
|---|---|---|---|---:|---:|---|
| 2026-11-06 | 02757d5 | Untrained ResNet9 | 0 | 0.3627 | - | Floor |
| 2026-11-06 | 02757d5 | Untrained ResNet9 | 1 | 0.3893 | - | Floor |
| 2026-11-06 | 02757d5 | Untrained ResNet9 | 2 | 0.3707 | - | Floor |
| 2026-11-08 | 347b73e | SwAV baseline, 10 epochs | 0 | 0.4493 | - | Matches epoch 10 of the 30-epoch run |
| 2026-11-08 | 347b73e | SwAV baseline, 30 epochs | 0 | 0.5173 | - | +15.5 over floor; epochs 21-27 timings distorted by battery sleep |
| 2026-11-08 | 347b73e | SwAV baseline, 30 epochs | 1 | 0.5213 | - | +13.2 over floor |
| 2026-11-08 | 347b73e | SwAV baseline, 30 epochs | 2 | 0.5160 | - | +14.5 over floor |
| 2026-11-08 | 3077304 | Queue 480 from epoch 2, 30 epochs | 0 | 0.4693 | - | -4.8 vs baseline |
| 2026-11-08 | 3077304 | Queue 480 from epoch 2, 30 epochs | 1 | 0.4840 | - | -3.7 vs baseline |
| 2026-11-08 | 3077304 | Queue 480 from epoch 2, 30 epochs | 2 | 0.4947 | - | -2.1 vs baseline |
| 2026-11-09 | 9d078db | Paper augmentation, 10 epochs (probe) | 0 | 0.4693 | - | +2.0 vs baseline at epoch 10 |
| 2026-11-10 | 9d078db | Paper augmentation, 30 epochs | 0 | 0.4920 | - | -2.5 vs baseline |
| 2026-11-10 | 9d078db | Paper augmentation, 30 epochs | 1 | 0.4307 | - | -9.1 vs baseline |
| 2026-11-10 | 9d078db | Paper augmentation, 30 epochs | 2 | 0.4480 | - | -6.8 vs baseline |

Floor mean: 0.374. Chance: 3.4%.
Baseline mean (30 epochs): 0.518 +/- 0.003; mean gain over the paired floor
14.4 +/- 1.1 points. Means at 10 and 20 epochs: 0.451 and 0.478.
Baseline settings: `uv run main.py --seed S --epochs 30` with all other
defaults (batch 32, 300 prototypes). The model is initialised from the same
weights as the floor of the same seed, so each gain is a paired comparison.

Queue ablation: `--queue-length 480` (15 batches, the paper's 3840 / 256),
started in epoch 2, all else as the baseline. Mean at 30 epochs 0.483 +/- 0.013,
3.6 points below the baseline; lower for every seed. Mean over epochs 26-30:
2.3 points below. Not tested: why. One guess is that with AdamW at batch 32 the
network changes fast, so 15-batch-old projections give stale targets.

Augmentation ablation: `--color-jitter 0.8 0.8 0.8 0.2 --blur-probability 0.5`
(the SwAV paper's recipe), all else as the baseline. Mean at 30 epochs
0.457 +/- 0.032, 6.1 points below the baseline; lower for every seed. Mean over
epochs 26-30: 3.9 points below. Seeds 1 and 2 stayed at 0.30-0.35 for the first
7 epochs, below their untrained floors, before rising. Jitter and blur changed
together, so we cannot say which one hurts. Not tested: why. One guess is that
colour carries much of the class information in the galaxy and satellite
images, and strong saturation and hue jitter removes it.

## Observations
- Only 29 of the 30 classes appear among the labelled images.
- Three untrained models span 2.7 points, so small gaps need several seeds.
- The score jumps 2 to 6 points between neighbouring epochs, so the spread
  at epoch 30 (0.003) is luck. Treat gains below about 2 points as noise.
- We report the last epoch, never the best one: picking an epoch by score
  would use labels for model selection.
- Runs are exactly repeatable on MPS: three runs with the same seed gave
  identical ONNX files, even with a sleep interruption.
- The loss still falls at epoch 30 (4.29 for seed 0).
- A single-seed probe can mislead: the 10-epoch augmentation probe on seed 0
  was 2 points above the baseline, but seeds 1 and 2 were 7 to 9 points below.
- Losses from different augmentations are not comparable: stronger
  augmentation makes the task harder and the loss higher.

## Next
1. Learning-rate schedule (warmup plus cosine, as in the paper), 3 seeds
   against the baseline.
2. If time remains: blur alone and stronger jitter alone, prototypes, epochs.

## Open questions
- Exact presentation and paper dates
- `main.py` format and retrain time limit
- How uploads map to groups
