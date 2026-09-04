# VTUS and MUP L2D intermediate-data specification

The generator `create_grayscale_l2d_intermediate.py` creates R(2+1)D inputs
for both grayscale datasets. Use `--dataset vtus` or `--dataset mup`.

Each clip/round is an independent sample. A shared video NPZ contains
`grayscale_frames uint8 [1,10,112,112]`. A round NPZ contains
`propagated_masks uint8 [1,10,112,112]`, `action_ious float32 [11]`, and
`already_prompted_mask bool [10]`. The dataloader concatenates image and mask
to produce `[2,10,112,112]`.

The 11 actions are `[stop, slot_0, ..., slot_9]`. Already-prompted slots have
`NaN` targets and are invalid. Costs are intentionally not stored; training
derives them from IoU and the selected `beta`.

VTUS has 30-frame clips and candidate locations `[0,3,6,9,12,16,19,22,25,29]`.
Each of its 16 box-scale combinations is stored independently under
`CMIG_l2d_data/vtus/vtus_<initial>_<correction>`.

MUP clips have 10 frames, so candidate slots map directly to frames 0 through
9. Frame 0 was prompted using a non-expert mask and corrections used expert
masks. The propagation program calculated every baseline and candidate IoU
against `expert_annotations`; the generator requires and records that GT
provenance. Output is `CMIG_l2d_data/mup/mup_mask_prompts`.

Split dictionaries remain authoritative. Every round of a clip stays in the
same train/validation/test split. Atomic NPZ writes, batch-specific manifests,
`--batch-size`, `--batch-index`, `--dry-run`, and `--overwrite` support safe
parallel and resumable processing.
