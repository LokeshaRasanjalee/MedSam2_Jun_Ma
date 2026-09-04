# CMIG VTUS Dataset Specification

## Output

The generated dataset is stored at:

```text
CMIG_clips/VTUS/vtus_clips_train_stride_15_val_test_stride_30/
├── Images/
├── Masks/
└── split_dict_vtus.txt
```

The existing subject-level split is retained without reshuffling.

| Split | Videos | Clip length | Frame interval | Inter-clip stride | Expected clips |
|---|---:|---:|---:|---:|---:|
| Training | 60 | 30 | 1 | 15 | 285 |
| Validation | 10 | 30 | 1 | 30 | 30 |
| Testing | 30 | 30 | 1 | 30 | 75 |
| **Total** | **100** | — | — | — | **390** |

Images and masks are saved at 256 x 256 pixels. Images use area interpolation and masks use nearest-neighbour interpolation. Only frames with a non-zero mask are collected. An incomplete tail containing fewer than 30 eligible frames is discarded without padding.

## Parallel batching

All 100 original video IDs are sorted lexicographically, then divided into 10-video batches. This creates exactly 10 deterministic, non-overlapping batches numbered 0 through 9. A batch can include different dataset splits; the script looks up each video's split and applies its corresponding stride and source directory.

Inspect assignments without processing data:

```bash
python CMIG_data_processing/create_vtus_clips.py --list-batches
```

Dry-run one batch:

```bash
python CMIG_data_processing/create_vtus_clips.py --batch-index 0 --batch-size 10 --dry-run
```

Dry-run the complete dataset and verify the expected counts:

```bash
python CMIG_data_processing/create_vtus_clips.py --dry-run
```

Submit all 10 CPU batches in parallel:

```bash
sbatch hpc/create_vtus_clips_cpu.slurm
```

Each source video belongs to exactly one batch, so parallel tasks never generate the same clip folder. Existing clip folders are protected by default. Use `--overwrite` only when intentionally rerunning batches whose outputs should be replaced.

## Acceptance checks

1. The completed dataset contains 285 training, 30 validation, and 75 testing clips.
2. Every clip contains exactly 30 paired images and masks.
3. Every output image and mask is 256 x 256.
4. Resizing masks introduces no new label values.
5. `split_dict_vtus.txt` matches the existing fixed VTUS split.
6. Every original video is assigned to exactly one 10-video batch.

## Current status

- Processing script: written and syntax-checked.
- CPU SLURM array script: written and syntax-checked.
- Clip generation: completed with 285 training, 30 validation, and 75 testing clips.
- Output pairing and frame counts: verified for all 390 clips.
- Duplicate review: no repeated first-frame convergence or exact duplicate clips was found with the finalized strides.
