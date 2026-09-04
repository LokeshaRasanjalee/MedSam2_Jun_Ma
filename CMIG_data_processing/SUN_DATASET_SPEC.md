# CMIG SUN Dataset Specification

## Purpose

Create a SUN video-clip dataset for SAM2 prompting and R(2+1)D rejector experiments. The generated dataset preserves the existing subject-level train, validation, and test assignments while using split-specific inter-clip strides.

## Output location

```text
/hpcfs/users/a1917962/Medsam2_working/MedSam2_Jun_Ma/
└── CMIG_clips/
    └── SUN/
        └── sun_clips_train_stride_30_test_stride_100/
            ├── Images/
            ├── Masks/
            └── split_dict_sun.txt
```

The `Images` and `Masks` directories use matching clip-folder names. The split file remains at the dataset root.

## Subject-level split

The split assignments come from `split_dict_sun.txt` and must not be regenerated or shuffled.

| Split key | Split | Video IDs | Videos expected to produce clips |
|---:|---|---:|---:|
| 0 | Training | 95 | 92 |
| 1 | Validation | 17 | 16 |
| 2 | Testing | 37 | 37 |

Clips inherit the split of their source video. Clips from one source video must never occur in more than one split.

The validation video `case4` is expected to produce no clip because it has fewer than 100 eligible frames.

## Clip-generation parameters

| Parameter | Training | Validation | Testing |
|---|---:|---:|---:|
| Clip length | 100 frames | 100 frames | 100 frames |
| Frame interval | 1 | 1 | 1 |
| Inter-clip stride | 30 | 100 | 100 |
| Output frame size | 256 × 256 | 256 × 256 | 256 × 256 |
| Expected clips | 278 | 24 | 73 |

The expected total is **375 clips**. These counts were measured directly from the extracted raw masks with the processing script's foreground eligibility rule. The earlier training estimate of 276, inferred from previously windowed datasets before the raw archives were extracted, was superseded by the authoritative dry-run result of 278.

A frame is eligible only when its corresponding mask contains at least one non-zero pixel. Starting at each stride position, the generator advances through source masks at frame interval 1 and collects eligible frames until it obtains 100 frames. A clip is emitted only if all 100 frames can be collected.

Stride 100 makes validation and test clips non-overlapping with respect to their starting windows. Training stride 30 intentionally creates overlapping clips.

Overlap between neighbouring training clips is intentional and does not by itself mean that two clips are identical. A content comparison was performed after generation. Four pairs had repeated displayed first-frame numbers, but their image and mask contents differed because that abbreviated SUN frame number is not globally unique within a source video. No exact duplicate SUN clips were confirmed, so all 278 training clips are retained.

## Resizing

- Save RGB images at exactly 256 × 256 pixels.
- Save segmentation masks at exactly 256 × 256 pixels.
- Resize images with area interpolation.
- Resize masks with nearest-neighbour interpolation to preserve label values.
- The downstream R(2+1)D preparation may resize these saved frames to 112 × 112; that model transformation is separate from dataset creation.

## Clip and frame naming

Each clip folder uses the existing SUN convention:

```text
<source_video_id>_<clip_number>_<first_source_frame_number>
```

Examples:

```text
case100_10_0022
case10_1_23_0045
```

The clip number starts at 1 independently for each source video. Frames inside every clip are renamed sequentially:

```text
Images/<clip_name>/0000.jpg ... 0099.jpg
Masks/<clip_name>/0000.png  ... 0099.png
```

Image and mask names must correspond one-to-one.

## Source layout

The generator expects these logical source locations:

```text
SUN RGB images (both roots are required):
datasets/SUN_data/home/lokesha/Downloads/sundatabase_positive_part1/
datasets/SUN_data/home/lokesha/Downloads/sundatabase_positive_part2/

Training and validation masks:
datasets/SUN_data/SUN-SEG-Annotation/SUN-SEG-Annotation/TrainDataset/GT/

Test masks:
datasets/SUN_data/SUN-SEG-Annotation/SUN-SEG-Annotation/TestHardDataset/Unseen/GT/
```

The paths can be overridden through command-line arguments. The generator searches both RGB roots for each base case and requires each case to occur in exactly one root. It must stop with a clear error if a required source directory, source image, source mask, or split file is missing.

## Processing script

The implementation is `CMIG_data_processing/create_sun_clips.py`.

Validate inputs and expected counts without writing clips:

```bash
python CMIG_data_processing/create_sun_clips.py --dry-run
```

Create the dataset:

```bash
python CMIG_data_processing/create_sun_clips.py
```

Replace clip directories from a previous run explicitly:

```bash
python CMIG_data_processing/create_sun_clips.py --overwrite
```

The default behavior must not silently replace an existing clip directory.

## Acceptance checks

Generation is complete only when all of the following hold:

1. Training, validation, and test contain 278, 24, and 73 clips respectively when classified using `split_dict_sun.txt`.
2. The dataset contains 375 clips in total.
3. Every clip has exactly 100 images and 100 masks.
4. Every saved image and mask is 256 × 256.
5. Every image has a matching mask with the same relative clip path and frame stem.
6. Mask resizing has not introduced new label values.
7. No source video ID occurs in multiple splits.
8. Existing output clip folders are not overwritten unless `--overwrite` is supplied.

## Current status

- Dataset directory structure: created.
- Subject split file: copied and verified against the existing SUN split.
- Processing script: written and syntax-checked.
- Clip generation: completed with 278 training, 24 validation, and 73 testing clips.
- Raw source archives: extracted successfully into their default locations.
- Dry-run validation: completed against the extracted raw data; measured 278 training, 24 validation, and 73 test clips.
- Output pairing and frame counts: verified for all 375 clips.
- Exact-duplicate review: no exact duplicate clips confirmed; repeated abbreviated first-frame labels were not content duplicates.
