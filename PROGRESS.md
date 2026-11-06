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
- Use labels for monitoring only, never in the training objective.
- The final `main.py` must run offline from the data alone.
- Change one setting at a time and use at least 3 seeds per setting.

## Done
- 2026-11-06: Forked the repo and set up the environment with `uv sync`.
- 2026-11-06: Added `make_random_init.py` and measured the score floor.
- 2026-11-06: Added `swav_data.py` with 2 global crops (96 px) and 4 local
  crops (48 px).

## Decisions
| Setting | Value | Why |
|---|---|---|
| Encoder | ResNet9, 1024 dims | 3.49M parameters, 13.3 MiB ONNX, under the cap |
| Global crops | 2 at 96 px, scale 0.14-1 | SwAV paper's scales |
| Local crops | 4 at 48 px, scale 0.05-0.14 | Paper's scales, size scaled down |
| Colour jitter | 0.4, 0.4, 0.2, 0.1, p=0.8 | stable-pretraining benchmark |
| Blur, solarize | Off for now | Test later as ablations |
| Prototypes | 300 | Starting choice for 30 classes, 7,500 images |
| Epsilon, iterations, temperature | 0.05, 3, 0.1 | SwAV paper |
| Batch size, optimizer | 128, AdamW at 5e-4 | stable-pretraining benchmark |

## Run log
Local score = `evaluate.py` on `data/train.parquet` (leave-one-out on labelled
images). It does not cover held-out images or new classes.

| Date | Commit | Run | Seed | Local score | Upload score | Notes |
|---|---|---|---|---:|---:|---|
| 2026-11-06 | 02757d5 | Untrained ResNet9 | 0 | 0.3627 | - | Floor |
| 2026-11-06 | 02757d5 | Untrained ResNet9 | 1 | 0.3893 | - | Floor |
| 2026-11-06 | 02757d5 | Untrained ResNet9 | 2 | 0.3707 | - | Floor |

Floor mean: 0.374. Chance: 3.4%.

## Observations
- Only 29 of the 30 classes appear among the labelled images.
- Three untrained models span 2.7 points, so small gaps need several seeds.

## Next
1. Build the SwAV model: targets from global crops only, prototypes frozen for
   the first epoch.
2. Smoke test about 10 steps on the laptop.
3. Export the encoder only, then run `test_submission.py` and `evaluate.py`.
4. Do a first real run on a GPU and compare it with the floor.
5. Run ablations: prototypes, local crops, target crops, epsilon, augmentations.

## Open questions
- Exact presentation and paper dates
- `main.py` format and retrain time limit
- How uploads map to groups
