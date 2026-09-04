# CMIG MUP Dataset Specification

## Purpose

Create a Micro-Ultrasound Prostate Segmentation (MUP) video-clip dataset for CMIG experiments. Preserve the existing subject-level train, validation, and test assignments while applying split-specific inter-clip strides.

## Output location

```text
/hpcfs/users/a1917962/Medsam2_working/MedSam2_Jun_Ma/
└── CMIG_clips/
    └── MUP/
        └── mup_clips_train_stride_5_val_test_stride_10/
            ├── micro_ultrasound_scans/
            ├── expert_annotations/
            ├── non_expert_annotations/
            └── split_dict_mup.txt
```

The three data directories use identical clip-folder names and matching sequential frame names.

## Subject-level split

The existing split from `miccai_data_pkl_mup/mask_k10_mup_all/split_dict.txt` is retained without regeneration or shuffling.

| Split key | Split | Subjects |
|---:|---|---:|
| 0 | Training | 39 |
| 1 | Validation | 8 |
| 2 | Testing | 8 |
| — | **Total** | **55** |

Every clip inherits the split of its source subject. A subject must never appear in more than one split.

## Clip-generation parameters

| Parameter | Training | Validation | Testing |
|---|---:|---:|---:|
| Clip length | 10 frames | 10 frames | 10 frames |
| Frame interval | 1 | 1 | 1 |
| Inter-clip stride | 5 | 10 | 10 |
| Output size | 256 x 256 | 256 x 256 | 256 x 256 |
| Unique clips | 232 | 25 | 25 |

The finalized total is **282 unique clips**. The source-window scan finds 248 training candidates, but 16 candidates converge onto sequences already selected from the same subject after ineligible frames are skipped. These exact repeated sequences are retained only once, leaving 232 unique training clips. Validation and testing contain 25 unique clips each.

## Frame eligibility and tails

A frame is eligible only when both its expert mask and corresponding non-expert mask contain at least one non-zero pixel. Starting at each stride position, the generator advances through the ordered source frames at interval 1 and collects eligible frames until it obtains 10 frames.

A clip is created only when all 10 eligible frames can be collected. An incomplete tail is discarded without padding or creating a partial clip.

If two different stride start positions skip the same ineligible region and then collect the same ordered set of 10 eligible frames, they represent the same clip content. Only one copy is retained in the finalized dataset. Ordinary partial overlap between different clips remains allowed.

## Resizing

- Save ultrasound images at exactly 256 x 256 pixels.
- Save expert masks at exactly 256 x 256 pixels.
- Save non-expert masks at exactly 256 x 256 pixels.
- Resize images with area interpolation.
- Resize both mask types with nearest-neighbour interpolation to preserve their label values.

## Naming

Each clip directory uses this convention:

```text
<subject_id>_<clip_length>_<frame_interval>_<stride>_<first_source_frame>
```

Example:

```text
01_10_1_5_0009
```

Frames inside each clip are renamed sequentially:

```text
micro_ultrasound_scans/<clip_name>/0000.jpg ... 0009.jpg
expert_annotations/<clip_name>/0000.png      ... 0009.png
non_expert_annotations/<clip_name>/0000.png  ... 0009.png
```

Every output image must have matching expert and non-expert masks with the same frame stem.

## Source layout

The generator reads the converted source frames from:

```text
datasets/Micro_Ultrasound_Prostate_Segmentation_Dataset/MUP_created_datasets/
├── micro_ultrasound_scans/microUS_train_<subject_id>/
├── expert_annotations/expert_annotation_train_<subject_id>/
└── non_expert_annotations/nonexpert_annotation_train_<subject_id>/
```

The source root, split path, and output root can be overridden using command-line arguments.

## Parallel batching

All 55 subject IDs are sorted lexicographically before batching. With the default batch size of 10, they form six deterministic and non-overlapping batches:

| Batch index | Subject IDs | Count |
|---:|---|---:|
| 0 | 01-10 | 10 |
| 1 | 11-20 | 10 |
| 2 | 21-30 | 10 |
| 3 | 31-40 | 10 |
| 4 | 41-50 | 10 |
| 5 | 51-55 | 5 |

A batch can contain subjects from different dataset splits. The generator looks up each subject's split and applies the appropriate stride. Because each subject belongs to exactly one batch, parallel jobs do not generate the same clip directories.

## Processing commands

The implementation is `CMIG_data_processing/create_mup_clips.py`.

Show batch assignments:

```bash
python CMIG_data_processing/create_mup_clips.py --list-batches
```

Dry-run the complete dataset and verify expected counts:

```bash
python CMIG_data_processing/create_mup_clips.py --dry-run
```

Dry-run one batch:

```bash
python CMIG_data_processing/create_mup_clips.py \
    --batch-index 0 \
    --batch-size 10 \
    --dry-run
```

Process one batch directly:

```bash
python CMIG_data_processing/create_mup_clips.py \
    --batch-index 0 \
    --batch-size 10
```

Existing clip directories are protected by default. Add `--overwrite` only when intentionally replacing the selected batch's existing output.

## SLURM execution

The CPU SLURM array script is `hpc/create_mup_clips_cpu.slurm`. It launches array indices 0 through 5, one task per batch.

```bash
cd /hpcfs/users/a1917962/Medsam2_working/MedSam2_Jun_Ma
sbatch hpc/create_mup_clips_cpu.slurm
```

Logs are written as:

```text
hpc/mup_clips_<job_id>_<array_index>.out
hpc/mup_clips_<job_id>_<array_index>.err
```

## Acceptance checks

Generation is complete only when all the following hold:

1. The dataset contains 232 unique training, 25 validation, and 25 testing clips.
2. The total number of clips is 282 in each of the three output collections.
3. Every clip contains exactly 10 images, 10 expert masks, and 10 non-expert masks.
4. Every saved image and mask is 256 x 256.
5. Image, expert-mask, and non-expert-mask paths match one-to-one.
6. Mask resizing introduces no new label values.
7. The output split file matches the existing fixed MUP split.
8. Every source subject is assigned to exactly one batch.
9. Existing output clip directories are not replaced unless `--overwrite` is supplied.

## Current status

- Processing script: written and syntax-checked.
- CPU SLURM array script: written and syntax-checked.
- Raw candidate-window scan: 248 training, 25 validation, and 25 testing windows.
- Clip generation: completed with 232 unique training, 25 validation, and 25 testing clips.
- Output pairing: verified across all 282 ultrasound, expert-mask, and non-expert-mask clip folders.
- Duplicate policy: exact repeated sequences are stored only once; ordinary overlapping clips are retained.
