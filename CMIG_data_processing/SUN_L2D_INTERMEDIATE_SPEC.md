# SUN L2D intermediate-data specification

## Purpose

Convert the merged four-round SUN SAM2 propagation results into compact,
resumable inputs for the new R(2+1)D learning-to-defer trainer. Each
`(prompt-scale combination, clip, round)` is an independent model sample.

This stage does **not** calculate action costs. It stores IoUs so that `beta`
and the loss temperature can be changed during training without rebuilding the
dataset.

## Inputs

- SUN clips: `CMIG_clips/SUN/sun_clips_train_stride_30_test_stride_100`
- Merged propagation results: `CMIG_npz_data/sunseg/sunseg_<initial>_<correction>`
- Prompt-scale tags: `10`, `12`, `14`, and `18` mean scales 1.0, 1.2, 1.4,
  and 1.8 respectively.

The generator reads the 10 candidate locations from each propagation log. For
SUN these are normally clip-frame indices `[0, 11, 22, 33, 44, 55, 66, 77,
88, 99]`. Action IoUs are the mean over all 100 propagated frames, not only
these 10 locations.

## Output layout

```text
CMIG_l2d_data/sunseg/
  shared_videos/
    <clip>.npz
  sunseg_<initial>_<correction>/
    samples/
      <clip>_round_1.npz
      ...
      <clip>_round_4.npz
    manifests/
      <invocation>.csv
```

RGB data are shared across all 16 prompt-scale combinations. Propagated masks
and action targets remain combination-specific.

## Shared-video NPZ

- `rgb_frames`: `uint8 [3, 10, 112, 112]`, RGB channel order.
- `candidate_frame_indices`: `int16 [10]`, locations within the 100-frame clip.
- `clip_name`: scalar string.
- `source_frame_count`: scalar `int16` (expected 100).

## Round-sample NPZ

- `propagated_masks`: `uint8 [1, 10, 112, 112]`, binary current-round SAM2
  masks sampled at the same 10 locations.
- `action_ious`: `float32 [11]` in decision order
  `[stop, frame_slot_0, ..., frame_slot_9]`. Stop is the baseline propagation
  mean IoU. A valid correction contains its corrected propagation mean IoU.
  Already-prompted correction slots contain `NaN`.
- `already_prompted_mask`: `bool [10]`. The loader derives the full mask as
  `valid_action_mask = [True] + ~already_prompted_mask`.
- `candidate_frame_indices`: `int16 [10]` (duplicated intentionally so a target
  sample is self-describing).
- Scalar metadata: `clip_name`, `source_video_id`, `split`, `round`,
  `initial_box_scale`, and `correction_box_scale`.

The trainer concatenates normalized RGB with the propagated mask to obtain
`[4, 10, 112, 112]` for one SUN sample. Candidate IoUs are targets only and
must never be included in the network input.

## Split safety

`split_dict_sun.txt` is the authority: key 0 is train, key 1 validation, and
key 2 test. All four rounds and all scale combinations of a clip retain its
source video's split. Splitting individual rounds would leak the same clip
between sets and is forbidden.

## Resuming and batching

Writes use a temporary file followed by an atomic rename. Existing valid NPZ
files are skipped unless `--overwrite` is supplied. Optional `--batch-size`
and `--batch-index` select a deterministic slice of sorted clips. Each
invocation writes its own manifest, so interrupted runs can be repeated.

## Trainer and TensorBoard contract

The future trainer must require CUDA and log separate `train/`, `val/`, and
`test/` TensorBoard series. Validation and test run every `--eval-every`
epochs (default 10) and on the final epoch. Log at least: surrogate loss,
exact action accuracy, corrected top-1/top-3/top-5 accuracy, chosen and best
available mean IoU, chosen and oracle cost, cost regret, IoU regret,
chosen-action cost rank, model/oracle deferral rates, stop/defer disagreement,
and index/temporal distance with their eligible sample count. Also log epoch
time, evaluation time, learning rate, and peak CUDA memory.

Checkpoints are retained for lowest validation chosen cost, highest validation
chosen mean IoU, lowest test chosen cost, highest test chosen mean IoU, and the
latest resumable state. Test-selected checkpoints are diagnostic and are not
an unbiased final model selection.
