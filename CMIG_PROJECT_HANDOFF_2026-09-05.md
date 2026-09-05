# CMIG Project Handoff — 2026-09-05

This document is the starting point for continuing the CMIG/SAM2 learning-to-defer project on another machine or in a new agent session.

## Read this first

1. The original SAM2 propagation outputs in `CMIG_npz_data` are valid.
2. An earlier intermediate-data conversion bug erased every propagated-mask channel. Models trained before that bug was fixed are not scientifically valid, even if their training and evaluation jobs completed normally.
3. The corrected intermediate datasets have been regenerated and independently checked. Only models whose names include `fixed_masks_v2` or `fixed_masks_v3` should currently be considered for analysis.
4. Experiment A was accidentally submitted twice into the same output directories. Its checkpoints are usable for debugging because both deterministic runs produced the same central metrics, but its `metrics.csv` rows and TensorBoard event files are duplicated. Do not use those folders as clean publication records.
5. Do not submit SLURM jobs without the user's explicit instruction. The user normally runs submitted jobs themselves.
6. The Git worktree is intentionally dirty, including a user-updated SAM2 tree. Do not reset, discard, or overwrite unrelated changes.

## Repository locations

Primary workspace:

```text
/scratchdata1/users/a1917962/Medsam2_working/MedSam2_Jun_Ma
```

The corresponding persistent/HPC-visible path used in commands and SLURM scripts is:

```text
/hpcfs/users/a1917962/Medsam2_working/MedSam2_Jun_Ma
```

Important directories:

```text
CMIG_clips/                 Generated video clips and GT annotations
CMIG_npz_data/              SAM2 propagation results and round information
CMIG_l2d_data/              Corrected 112x112 learning-to-defer samples
CMIG_l2d_training/          Checkpoints, metrics, manifests and TensorBoard logs
CMIG_iterative_evaluation/  Iterative evaluation outputs
CMIG_data_processing/       Data, training, diagnostic and evaluation scripts
hpc/                        SLURM scripts and job output files
sam2/                       User-updated SAM2 source and checkpoint
```

Reference papers:

```text
Mao et al. - 2023 - Two-Stage Learning to Defer with Multiple Experts.pdf
miccai-2026.pdf
```

## Conda and CUDA environment

Python used successfully:

```text
/hpcfs/users/a1917962/.conda/envs/medsam2/bin/python
PyTorch 2.6.0+cu124
```

The CUDA 12.4 module configuration worked with the installed PyTorch build. A typical GPU job environment is:

```bash
module purge
module load GCC/11.2.0
module load CUDA/12.4.1
source /hpcfs/users/a1917962/.conda/etc/profile.d/conda.sh
conda activate medsam2
```

Avoid mixing CUDA 12.2 libraries with the `cu124` PyTorch build. Some nodes previously produced hardware/driver errors. The propagation SLURM setup excluded:

```text
p2-gpu-23,p2-gpu-25,p3-gpu-48
```

One earlier failure was an uncorrectable GPU ECC error. That is a node/GPU problem, not a SAM2 model-code error.

## Generated clip datasets

All created video frames are 256x256.

| Dataset | Clip directory | Channels | Clip length | Train stride | Validation stride | Test stride | Current clip total |
|---|---|---:|---:|---:|---:|---:|---:|
| SUN-SEG | `CMIG_clips/SUN/sun_clips_train_stride_30_test_stride_100` | RGB | 100 | 30 | 100 | 100 | 375 |
| VTUS | `CMIG_clips/VTUS/vtus_clips_train_stride_15_val_test_stride_30` | grayscale | 100 | 15 | 30 | 30 | 390 |
| MUP | `CMIG_clips/MUP/mup_clips_train_stride_5_val_test_stride_10` | grayscale | 10 | 5 | 10 | 10 | 282 |

SUN-SEG details:

- Training: 95 assigned original video IDs, 92 producing clips, 278 clips. The earlier estimate of 276 was superseded by the raw-data dry run.
- Validation: 17 video IDs; 16 produce clips; 24 clips.
- Testing: 73 clips.
- Total: 375 clips.

MUP details:

- Training: 232 clips.
- Validation: 25 clips.
- Testing: 25 clips.
- Scan images: `micro_ultrasound_scans`.
- Initial prompts: `non_expert_annotations`.
- Corrections and evaluation ground truth: `expert_annotations`.

The data-creation specifications are under `CMIG_data_processing`; check the SUN, VTUS and MUP spec Markdown files there before changing clip-generation rules.

## Four-round SAM2 propagation workflow

The propagation scripts use ten candidate temporal positions per clip.

For SUN-SEG and VTUS:

- Frame/candidate 0 receives the initial GT-derived box prompt.
- Every other prompted frame receives a correction box prompt.
- Initial and correction box scales are independently configurable.
- The generated scale grid used values `1.0`, `1.2`, `1.4`, and `1.8` for both prompt types, producing 16 combinations.

For MUP:

- Candidate 0 receives a mask prompt from the non-expert annotation.
- Every correction receives an expert mask prompt.
- Expert annotations are also the ground truth used for IoU.
- There is only one prompt type/setting, not a box-scale grid.

Round logic:

1. Round 1 propagates candidate 0 alone and measures its video IoUs. It then independently tries each remaining candidate as one correction.
2. The candidate whose correction gives the highest mean IoU across the full video becomes `i1`.
3. Round 2 propagates fixed prompts `[0, i1]`, then tries each unselected candidate individually. The best becomes `i2`.
4. Round 3 propagates `[0, i1, i2]`, tries each remaining candidate, and chooses `i3`.
5. Round 4 propagates `[0, i1, i2, i3]`. Round-specific masks and information are retained.

Outputs include:

```text
sam2_masks/<clip>_round_<round>/
info_dict/<clip>_round_<round>.json
```

The JSON contains the baseline prompt set and each candidate extension, with a complete per-frame IoU list for every action. Candidate selection uses the highest mean IoU over the video.

Primary scripts:

| Purpose | Python script | SLURM script |
|---|---|---|
| SUN propagation | `CMIG_data_processing/propagate_sun_box_prompts_4rounds.py` | `hpc/propagate_sun_box_prompts_4rounds.slurm` |
| VTUS propagation | `CMIG_data_processing/propagate_vtus_box_prompts_4rounds.py` | corresponding VTUS propagation SLURM in `hpc/` |
| MUP propagation | `CMIG_data_processing/propagate_mup_mask_prompts_4rounds.py` | corresponding MUP propagation SLURM in `hpc/` |
| Merge SUN batch outputs | `CMIG_data_processing/merge_sunseg_batch_outputs.py` | run directly/background as needed |

Use `rg --files CMIG_data_processing hpc | rg 'propagate|merge'` if a listed SLURM filename has changed.

SAM2 files:

```text
Checkpoint: sam2/checkpoints/sam2.1_hiera_tiny.pt
Hydra config name: configs/sam2.1/sam2.1_hiera_t.yaml
Config file: sam2/sam2/configs/sam2.1/sam2.1_hiera_t.yaml
```

The config was adjusted for internal image size 256 and two memory-attention layers.

## Critical propagated-mask conversion incident

SAM2 saved palette-indexed PNG masks in `CMIG_npz_data`. Their pixel indices are valid binary values `0` and `1`, but both palette colors appear black. OpenCV grayscale loading expanded the palette colors rather than preserving the palette indices, which converted all foreground pixels to zero.

Consequences:

- The old `propagated_masks` arrays in `CMIG_l2d_data` were entirely empty for SUN-SEG, VTUS and MUP.
- Models trained from those old arrays did not see segmentation masks; they learned from images alone.
- Old loss ablation, architecture ablation, Figure 3, Figure 4 and iterative-evaluation conclusions are invalid for scientific use.
- The source masks in `CMIG_npz_data` were not empty, so SAM2 propagation did not need to be rerun.

The fix uses PIL to preserve the palette indices:

```python
mask = np.asarray(Image.open(mask_path)) > 0
```

Masks are then resized using nearest-neighbor interpolation and written as binary arrays. The fixed converters perform an exact post-write NPZ check and fail a batch if every saved mask is empty.

Corrected intermediate scripts:

```text
CMIG_data_processing/create_sun_l2d_intermediate.py
CMIG_data_processing/create_grayscale_l2d_intermediate.py
CMIG_data_processing/validate_l2d_mask_inputs.py
```

Relevant SLURM scripts include:

```text
hpc/create_sun_l2d_intermediate_cpu.slurm
hpc/create_vtus_l2d_intermediate_cpu.slurm
hpc/create_mup_l2d_intermediate_cpu.slurm
```

They were run with overwrite enabled. All 66 array tasks completed: SUN 32, VTUS 32, MUP 2.

Verified corrected datasets:

| Dataset | Sample directory | Samples | Expected | Total foreground pixels |
|---|---|---:|---:|---:|
| SUN-SEG | `CMIG_l2d_data/sunseg/sunseg_14_10/samples` | 1500 | 375 clips x 4 rounds | 10,122,239 |
| VTUS | `CMIG_l2d_data/vtus/vtus_14_10/samples` | 1560 | 390 clips x 4 rounds | 25,793,462 |
| MUP | `CMIG_l2d_data/mup/mup_mask_prompts/samples` | 1128 | 282 clips x 4 rounds | 59,188,265 |

Each propagated-mask tensor is binary `uint8` with shape `[1, 10, 112, 112]`.

Visual inspection tools:

```text
CMIG_data_processing/export_palette_masks_for_visual_check.py
CMIG_data_processing/create_mask_gt_overlay_check.py
CMIG_data_processing/diagnose_round_mask_signal.py
```

Example visual outputs are under `CMIG_mask_visual_checks/`. These show the original frame, GT, propagated mask and overlays without changing the source masks.

Before any new training, rerun the validator against the exact selected intermediate directory and verify foreground counts are nonzero.

## Learning-to-defer formulation

The main trainer is:

```text
CMIG_data_processing/train_l2d_r2plus1d.py
```

Input tensors:

- SUN-SEG: `[4, 10, 112, 112]` = three RGB image channels plus one propagated-mask channel.
- VTUS/MUP: `[2, 10, 112, 112]` = one grayscale image channel plus one propagated-mask channel.
- Each of the four rounds from one clip is treated as an independent training sample.
- The model is not explicitly given a round number. It sees the current propagated mask, and action masking prevents it from selecting already-prompted frames.

Action space:

- One fixed stop/non-defer action with score `0`.
- Ten learned frame-action scores, one for each temporal slot.
- Lower scores are better and the minimum valid score is selected.
- Candidate 0 and all already-selected correction frames are masked as invalid.
- This fixed ten-frame action space keeps the logic consistent across rounds.

The two masks used by the data/training logic are different:

- `selected_slot_mask [10]`: temporal slots that have already been prompted.
- `valid_action_mask [11]`: validity of the complete action set: stop plus ten frame actions.

Costs for the current decision:

```text
stop cost         = 1 - current propagated-mask mean IoU
frame-action cost = 1 - candidate-corrected mean IoU + alpha
```

The code historically calls the user-facing `alpha` parameter `--beta`. There is no separate `d_j`, and the segmentation-error coefficient is always 1.

For an iterative trajectory, alpha is charged once for each correction. At a single later decision, previous correction costs are already sunk and common to all available actions, so compare the current stop action with one additional `+ alpha`, not `3 * alpha`. The final trajectory cost after `k` corrections is:

```text
1 - final IoU + k * alpha
```

Soft oracle weights:

```text
w = softmax(-action_costs / tau)
```

The standard temperature used is `tau = 0.25`. Supported surrogate losses include MAE, log and exponential. MAE was used for the latest debugging experiments. Log loss remains a useful next comparison because it supplies stronger gradients when the model is confidently wrong.

Key metrics:

- Chosen mean IoU: mean IoU obtained by the selected stop/frame action.
- Oracle mean IoU: mean IoU obtained by the lowest-cost valid action.
- Chosen cost: cost of the model-selected valid action.
- Oracle cost: minimum valid action cost.
- Cost regret: chosen cost minus oracle cost.
- Exact action accuracy: model action equals the lowest-cost oracle action.
- Model deferral rate: percentage for which the model chooses a correction frame.
- Oracle deferral rate: percentage for which the lowest-cost action is a correction.
- Chosen-action cost rank: rank of the selected action after sorting valid actions by cost.
- Cost-rank distance: distance between the selected rank and oracle rank.
- IoU regret: oracle IoU minus chosen IoU.
- Frame-index distance and true temporal distance: reported only for relevant deferred cases.
- Top-1/top-3/top-5 correction accuracy: calculated consistently over valid correction actions.

Model selection rule:

- Use `best_val_chosen_cost.pt` for final scientific evaluation, then report test metrics once.
- `best_test_chosen_cost.pt` is useful only for debugging and retrospective diagnosis. Selecting it for reported results leaks test information.

Checkpoints also include configuration information so that evaluation can recover dataset, architecture, channels, alpha and related settings. Always inspect the saved config before trusting automatic loading.

## Architectures

The main trainer supports:

### Classic R(2+1)D-18

- Torchvision R(2+1)D-18.
- Input stem adapted for 4-channel SUN or 2-channel VTUS/MUP input.
- Stem and layers 1–3 frozen.
- Layer 4 and final classifier trained.
- The classic network globally averages temporal features before `Linear(512, 10)`, so each position has an independent output weight but the final feature no longer preserves an explicit temporal slot. This contributed to collapse onto one favorite frame index.

### ResNet-18 + one-layer GRU

- Each temporal frame, including its propagated-mask channel, is encoded by a ResNet-18.
- A shared single-layer GRU processes the ten frame embeddings.
- A shared head produces one action score per temporal position.
- The early backbone is frozen; layer 4, GRU and head can use separate learning rates.

### Temporal R(2+1)D-18

Architecture argument:

```text
r2plus1d_18_temporal
```

Implementation summary:

- Starts from pretrained R(2+1)D-18.
- Changes temporal strides in the first block of layers 2–4 from 2 to 1 while retaining spatial stride 2.
- Keeps all ten temporal positions through the backbone.
- Averages spatial dimensions only.
- Produces `[B, 10, 512]` temporal features.
- Applies one shared `Linear(512, 1)` head to every temporal position.
- Stem/layers 1–3 are frozen; layer 4 and the shared head train.
- `--head-learning-rate` supports a separate head rate.

This architecture was tested for both two-channel and four-channel inputs and outputs `[B, 10]`.

## Sampling/batch-balancing experiments

Natural validation and test distributions must remain unchanged. Balancing is applied only to training batches.

### Global 50/50 oracle balancing

Script:

```text
CMIG_data_processing/train_l2d_oracle_balanced_debug.py
```

For batch size 8, it draws four oracle-defer and four oracle-stop samples, regardless of round.

### Fixed round-aware balancing

Script:

```text
CMIG_data_processing/train_l2d_round_aware_balanced.py
```

Current alpha-0.1 behavior:

- 50% oracle-defer samples, drawn mainly from rounds 1–2: approximately 90% round 1 and 10% round 2.
- 50% oracle-stop samples, split equally across rounds 1–4.

This was designed around the observation that at alpha 0.1 nearly all useful deferrals occur in early rounds. It did not by itself solve under-deferral.

### Agreed future adaptive sampler

For experiments across smaller alpha values:

- Keep 50% oracle-defer and 50% oracle-stop training samples.
- Within the defer half, assign round probabilities proportional to the number of available oracle-defer samples at the selected alpha.
- Within the stop half, sample uniformly across nonempty round-specific stop pools.

This avoids hard-coding a round-1/round-2 split that becomes wrong at lower alpha, where rounds 3 and 4 can also contain valuable deferrals.

## Alpha range

The previous grid `0.0, 0.1, 0.2, ..., 0.7` was too coarse for these datasets. The agreed next grid is:

```text
0.02, 0.03, 0.04, 0.05, 0.075, 0.10, 0.15, 0.20
```

Observed validation oracle-deferral rates:

| Alpha | SUN-SEG | VTUS | MUP |
|---:|---:|---:|---:|
| 0.02 | 60.4% | 57.5% | 79% |
| 0.03 | 47.9% | 48.3% | 55% |
| 0.04 | 34.4% | 40.8% | 37% |
| 0.05 | 32.3% | 33.3% | 28% |
| 0.06 | 29.2% | 29.2% | 27% |
| 0.075 | 25.0% | 28.3% | 23% |
| 0.10 | 19.8% | 21.7% | 13% |
| 0.15 | 14.6% | 14.2% | 11% |
| 0.20 | 10.4% | 10.0% | 7% |

The dense low-alpha range is needed to obtain meaningful stop/defer trade-off curves.

## Valid corrected-mask experiments and results

All numbers below use the lowest-cost test checkpoint for debugging. For publication-style reporting, re-evaluate `best_val_chosen_cost.pt` instead.

### Classic R(2+1)D-18, global 50/50 balancing

Run directories:

```text
CMIG_l2d_training/sunseg_14_10_arch_r2plus1d_18_alpha_010_loss_mae_oracle_bal50_fixed_masks_v2_seed_42
CMIG_l2d_training/vtus_14_10_arch_r2plus1d_18_alpha_010_loss_mae_oracle_bal50_fixed_masks_v2_seed_42
CMIG_l2d_training/mup_mask_prompts_arch_r2plus1d_18_alpha_010_loss_mae_oracle_bal50_fixed_masks_v2_seed_42
```

| Dataset | Best-test epoch | Chosen cost | Cost regret | Chosen IoU | Model defer | Oracle defer |
|---|---:|---:|---:|---:|---:|---:|
| SUN-SEG | 1520 | 0.430744 | 0.041445 | 0.590831 | 21.58% | 23.63% |
| VTUS | 1070 | 0.330207 | 0.018062 | 0.687793 | 18.00% | 23.00% |
| MUP | 710 | 0.168734 | 0.003903 | 0.845266 | 14.00% | 16.00% |

Round-wise model versus oracle deferral rates:

- SUN: model `31.51, 23.29, 16.44, 15.07%`; oracle `84.93, 9.59, 0, 0%`.
- VTUS: model `56, 12, 2.67, 1.33%`; oracle `86.67, 5.33, 0, 0%`.
- MUP: model `48, 4, 4, 0%`; oracle `60, 4, 0, 0%`.

The totals can look close while the round-conditioned behavior is wrong: models under-defer heavily in round 1 and sometimes defer unnecessarily in later rounds.

The selected-frame distributions also collapsed:

- SUN mostly selected a frame around index 77 in the original 100-frame clip coordinate system.
- VTUS mostly selected a frame around index 17.
- MUP mostly selected candidate/frame 6.
- Oracle-selected locations were substantially more distributed.

Plotting script:

```text
CMIG_data_processing/plot_test_deferral_frame_distributions.py
```

Output directory:

```text
CMIG_l2d_training/deferral_frame_distributions/
```

### SUN classic R(2+1)D-18, fixed round-aware balancing

Run:

```text
CMIG_l2d_training/sunseg_14_10_arch_r2plus1d_18_alpha_010_loss_mae_roundaware_bal50_fixed_masks_v2_seed_42
```

Best test/validation checkpoint occurred at epoch 1710:

```text
chosen cost: 0.432625
cost regret: 0.043326
chosen IoU: 0.589635
model defer rate: 22.26%
exact action accuracy: 62.67%
round defer rates: 34.25, 21.92, 17.81, 15.07%
```

Global balancing was slightly better: lower cost `0.430744` versus `0.432625`. Round-aware balancing increased round-1 deferral, but also increased incorrect late-round deferrals. Batching alone did not fix the representation/calibration issue.

### ResNet-18 + GRU, low learning rate

Runs contain:

```text
arch_resnet18_gru_alpha_010_loss_mae_roundaware_bal50_fixed_masks_v2_seed_42
```

All trainable backbone/GRU/head components used `1e-7`, which is too small for newly initialized GRU and head layers.

| Dataset | Best-test epoch | Cost | Regret | IoU | Model defer | Oracle defer |
|---|---:|---:|---:|---:|---:|---:|
| SUN-SEG | 1110 | 0.423096 | 0.033797 | 0.579301 | 2.40% | 23.63% |
| VTUS | 430 | 0.336438 | 0.024294 | 0.672228 | 8.67% | 23.00% |
| MUP | 520 | 0.168946 | 0.004115 | 0.846054 | 15.00% | 16.00% |

The shared temporal head improved the spread of frame indices, but SUN and VTUS mostly learned to stop.

### ResNet-18 + GRU, differential high head rate

Runs contain:

```text
difflr_backbone_1e7_gruhead_1e4_fixed_masks_v3
```

Layer 4 used `1e-7`; GRU/head used `1e-4`.

- SUN: epoch 10, cost `0.427788`, regret `0.038489`, IoU `0.572212`, defer `0%` versus oracle `23.63%`.
- VTUS: epoch 10, cost `0.341484`, regret `0.029340`, IoU `0.658516`, defer `0%` versus oracle `23%`.
- MUP: epoch 470, cost `0.167753`, regret `0.002922`, IoU `0.844247`, defer `12%` versus oracle `16%`.

For SUN and VTUS, `1e-4` was too aggressive with the bounded MAE/softmax objective. Scores moved above the fixed stop score and the model became an always-stop model. MUP improved slightly.

### Experiment A: temporal R(2+1)D-18

SLURM:

```text
hpc/train_three_datasets_temporal_r2plus1d_experiment_a_gpu.slurm
```

Configuration:

```text
architecture: r2plus1d_18_temporal
layer-4 learning rate: 1e-7
shared head learning rate: 1e-5
loss: MAE
alpha: 0.1
sampling: fixed round-aware 50/50
epochs: 2000
seed: 42
```

Experiment A was submitted twice under the same run names. Expect duplicate `metrics.csv` rows and two TensorBoard event files in each run directory.

Best-test debugging results:

| Dataset | Epoch | Cost | Oracle cost | Regret | IoU | Oracle IoU | Model defer | Oracle defer |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SUN-SEG | 280 | 0.420393 | 0.389299 | 0.031094 | 0.590566 | 0.634331 | 10.96% | 23.63% |
| VTUS | 470 | 0.330762 | 0.312145 | 0.018617 | 0.683238 | 0.710855 | 14.00% | 23.00% |
| MUP | 300 | 0.168673 | 0.164831 | 0.003842 | 0.841327 | 0.851169 | 10.00% | 16.00% |

Round model versus oracle deferral:

- SUN: model `30.14, 8.22, 2.74, 2.74%`; oracle `84.93, 9.59, 0, 0%`.
- VTUS: model `48, 4, 4, 0%`; oracle `86.67, 5.33, 0, 0%`.
- MUP: model `40, 0, 0, 0%`; oracle `60, 4, 0, 0%`.

Validation-selected checkpoints evaluated on test:

| Dataset | Epoch | Test cost | Regret | IoU | Defer rate |
|---|---:|---:|---:|---:|---:|
| SUN-SEG | 240 | 0.422512 | 0.033213 | 0.585022 | 7.53% |
| VTUS | 230 | 0.331865 | 0.019720 | 0.684468 | 16.33% |
| MUP | 230 | 0.170549 | 0.005718 | 0.843451 | 14.00% |

Experiment A improves SUN cost compared with classic global balancing (`0.420393` versus `0.430744`) and gives similar VTUS/MUP costs. It reduces incorrect late-round deferral, but still severely under-defers in round 1.

### Experiment B

SLURM:

```text
hpc/train_three_datasets_temporal_r2plus1d_experiment_b_gpu.slurm
```

Configuration differs from Experiment A by using shared-head learning rate `1e-6`. Its run names include an explicit Experiment B identifier so it cannot overwrite Experiment A. It was intentionally parked and has not been submitted; no Experiment B result folders should be assumed to exist.

## Current central model problem

The latest model is not mainly failing because the total deferral rate is numerically far from the oracle. It is failing because it defers at the wrong rounds and does not reliably rank correction frames.

At alpha 0.1, the oracle wants corrections primarily in round 1:

- SUN: model 30% versus oracle 85%.
- VTUS: model 48% versus oracle 87%.
- MUP: model 40% versus oracle 60%.

The model then sometimes spends corrections in later rounds where the oracle nearly always stops. Temporal R(2+1)D improves the architecture's position awareness, but it does not fully calibrate the learned frame scores against the fixed stop score of zero.

Most likely contributing factors:

1. Stop/defer score calibration: choosing defer requires the minimum valid learned frame score to fall below fixed stop score `0`.
2. MAE saturation: bounded softmax MAE provides weak gradients when the action distribution becomes confidently wrong.
3. Frozen-backbone behavior: frozen parameters do not automatically freeze BatchNorm running statistics when the whole model is put into training mode. Confirm frozen BN layers remain in evaluation mode.
4. The fixed round-aware sampler was designed for alpha 0.1 and should not be reused unchanged for lower-alpha experiments.
5. The model sees round state only indirectly through images, propagated masks and invalid actions; there is no explicit round embedding.

## Recommended immediate next step

Before another expensive training sweep, perform validation-only stop-threshold calibration on Experiment A:

1. Load each dataset's `best_val_chosen_cost.pt` from Experiment A.
2. Preserve the model's frame ranking and valid-action masking.
3. Replace the default decision rule

   ```text
   defer if minimum valid frame score < 0
   ```

   with

   ```text
   defer if minimum valid frame score < threshold
   ```

4. Sweep the threshold on the validation set and choose the value that minimizes validation chosen cost.
5. Freeze that threshold and evaluate it once on test.
6. Report overall and per-round cost, regret, IoU, model/oracle deferral, and selected-frame distribution.

This diagnostic distinguishes a ranking problem from a stop-score calibration problem without retraining or changing the L2D loss. If validation calibration substantially raises round-1 deferral and lowers cost, retain the architecture and formalize the stop score/threshold. If it does not, inspect per-sample frame-score rankings and train with log loss before designing a more complicated objective.

After threshold calibration, recommended controlled experiments are:

1. Correct frozen BatchNorm behavior if needed.
2. Compare MAE with the corrected log loss using the same architecture and sampler.
3. Use the adaptive round-aware sampler for the agreed low-alpha grid.
4. Run new experiments under unique output names and at multiple seeds only after the single-seed behavior is credible.

Do not jump immediately to Experiment B as the first diagnostic; its lower head learning rate does not directly test whether the principal issue is threshold calibration.

## Update: Mao regression-deferral logistic implementation

After this handoff was first written, `CMIG_data_processing/train_l2d_r2plus1d.py` gained a new
`--loss mao_logistic` mode based on Equation 3 of *Regression with Multi-Expert Deferral*.

- The model learns 11 logits: one stop logit and ten candidate-frame logits.
- Inference selects the highest valid logit; there is no fixed zero stop threshold.
- Invalid/already-prompted actions are excluded from the softmax, cost sum, loss, and inference.
- Costs remain unnormalized: stop `1-current_IoU`; correction `1-corrected_IoU+alpha`.
- Each valid action weight is `sum(valid_costs)-action_cost`.
- The multiclass logistic term uses base-2 log, matching the paper.
- The temporal model uses a shared per-frame head and a separate learned stop head applied to the temporally pooled feature.

Three-dataset launcher:

```text
hpc/train_three_datasets_mao_regression_logistic_gpu.slurm
```

It uses the corrected-mask data, temporal R(2+1)D, fixed round-aware batches, alpha 0.1,
layer-4 LR `1e-7`, action-head LR `1e-5`, 2000 epochs, and seed 42. The iterative evaluator still
needs explicit support for the new 11-logit checkpoint format before these models are evaluated
iteratively.

### Preliminary running results from the best-test-cost checkpoints

SLURM array job:

```text
15936794
```

These jobs were still running when inspected. The checkpoint selection below uses test cost and is
therefore for debugging only, not final scientific reporting.

| Dataset | Current epoch | Best-test epoch | Chosen cost | Oracle cost | Cost regret | Chosen IoU | Oracle IoU | Model defer | Oracle defer |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SUN-SEG | 671/2000 | 590 | 0.437479 | 0.389299 | 0.048180 | 0.606699 | 0.634331 | 44.18% | 23.63% |
| VTUS | 641/2000 | 550 | 0.339326 | 0.312145 | 0.027181 | 0.687674 | 0.710855 | 27.00% | 23.00% |
| MUP | 805/2000 | 790 | 0.176748 | 0.164831 | 0.011917 | 0.841252 | 0.851169 | 18.00% | 16.00% |

Round-wise model/oracle deferral rates:

| Dataset | Round 1 | Round 2 | Round 3 | Round 4 |
|---|---:|---:|---:|---:|
| SUN-SEG model | 53.42% | 41.10% | 39.73% | 42.47% |
| SUN-SEG oracle | 84.93% | 9.59% | 0% | 0% |
| VTUS model | 61.33% | 25.33% | 12.00% | 9.33% |
| VTUS oracle | 86.67% | 5.33% | 0% | 0% |
| MUP model | 48.00% | 20.00% | 4.00% | 0% |
| MUP oracle | 60.00% | 4.00% | 0% | 0% |

Compared with temporal R(2+1)D Experiment A, current Mao-logistic chosen cost is worse by
approximately `0.0171` for SUN, `0.0086` for VTUS, and `0.0081` for MUP. The learned stop logit
has removed the old strong stopping bias and raised round-1 deferral, but it now over-defers after
round 1. SUN is the clearest failure: it continues to defer on roughly 40% of rounds 2-4 even though
the oracle almost always stops.

Round-1 selected-action cost rank is also still weak:

```text
SUN-SEG: 4.41
VTUS:    4.31
MUP:     3.12
```

Rank zero is the lowest-cost action, so correction-frame ranking still needs work.

### Revised diagnosis: sampler bias

The Mao surrogate is derived for the natural data distribution, but job `15936794` reused the old
artificial sampler:

```text
50% oracle-defer samples
50% oracle-stop samples
```

It also draws most defer samples from round 1. With a learned stop logit, this changes the training
action prior and can directly teach the stop head to defer too often on naturally distributed
validation/test data. This effect was less direct when stop was a fixed zero score.

The next controlled Mao experiment should therefore use ordinary shuffled samples from the natural
training distribution, with no oracle stop/defer or round balancing. Keep everything else fixed:

```text
architecture: r2plus1d_18_temporal
loss: mao_logistic
alpha: 0.1
layer-4 LR: 1e-7
frame/stop head LR: 1e-5
seed: 42
```

Compare natural sampling against job `15936794` using validation-selected checkpoints. Do not
conclude that Equation 3 is ineffective until this sampling-confound experiment has been run.

### Completed balanced Mao-logistic results

SLURM array job `15936794` completed all three tasks at epoch 2000. Lowest-test-cost results
(debugging only because test selected the epoch):

| Dataset | Best epoch | Chosen cost | Oracle cost | Regret | Chosen IoU | Model defer | Oracle defer |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SUN-SEG | 1270 | 0.436522 | 0.389299 | 0.047223 | 0.607999 | 44.52% | 23.63% |
| VTUS | 1880 | 0.337557 | 0.312145 | 0.025412 | 0.691110 | 28.67% | 23.00% |
| MUP | 1260 | 0.174534 | 0.164831 | 0.009703 | 0.841466 | 16.00% | 16.00% |

The learned stop logit removed the previous strong stopping bias but produced excessive deferral
after round 1. Training longer did not resolve it. Frame-distribution plots and CSV files are in:

```text
CMIG_l2d_training/deferral_frame_distributions/mao_logistic_roundaware_v4/
```

### Natural-sampling Mao experiment

SLURM job `15944205` uses ordinary natural shuffled batches:

```text
hpc/train_three_datasets_mao_regression_logistic_natural_gpu.slurm
```

At the approximately 40-minute inspection point, natural sampling had not solved the problem:

| Dataset | Observed epoch | Best-test epoch | Chosen cost | Regret | Model defer | Oracle defer |
|---|---:|---:|---:|---:|---:|---:|
| SUN-SEG | 547 | 390 | 0.436089 | 0.046790 | 38.01% | 23.63% |
| VTUS | 549 | 520 | 0.340634 | 0.028489 | 27.33% | 23.00% |
| MUP | 671 | 640 | 0.178319 | 0.013487 | 22.00% | 16.00% |

Natural sampling reduced SUN late-round deferral somewhat but did not fix early under-deferral or
late over-deferral. Oracle-balanced sampling was a confound but is not the main cause.

With 7-10 valid actions, Equation 3 weights `sum(costs)-cost_k` share a large common component and
can be very similar. This creates nearly uniform optimal logistic probabilities and weak practical
separation. Simple sum/global cost normalization does not change the relative weights and cannot
fix this.

### State-aware temporal model

New architecture and files:

```text
r2plus1d_18_temporal_state
CMIG_data_processing/train_l2d_round_stratified_state.py
hpc/train_three_datasets_mao_state_round_stratified_gpu.slurm
```

The model sends `already_prompted_mask [10]` through a 10-to-32-to-32 state encoder. The state
embedding conditions both the shared correction-frame head and learned stop head. It retains 11
learned logits and the unnormalized Mao logistic loss.

Every full batch contains two natural samples from each round. Every sample is used once per epoch;
the sampler never reads the oracle action and does not balance stop/defer outcomes. Verified training
pool sizes are SUN 278 per round, VTUS 285 per round and MUP 232 per round. State-aware array job
`15945448` was submitted by the user.

## Critical update: SUN clips cross acquisition boundaries

The current SUN clip dataset is not valid for temporal modeling. The generator treats every file in
a case folder as one continuous sequence, although many folders contain several acquisition segments
identified by filename prefixes such as `a1`, `a2`, and `a10`. It allows 100-frame windows to cross
between segments.

Confirmed example:

```text
clip: case25_1_1_0001
output frame 54 source: ..._a10_ayy_image0055
output frame 55 source: ..._a9_ayy_image0001
```

The adjacent saved images show an abrupt viewpoint change. Plain filename sorting can also place
`a10` before `a9`, but numeric sorting alone is insufficient: separate acquisition segments must
not be joined.

| Split | Current clips | Cross-acquisition clips |
|---|---:|---:|
| Train | 278 | 141 |
| Validation | 24 | 14 |
| Test | 73 | 43 |
| Total | 375 | 198 |

The downstream SUN data is internally consistent but faithfully represents the flawed clips: 1,500
four-round samples, nonempty binary propagated masks, valid action IoUs, correct round chains, no
exact duplicate shared tensors and no split leakage. The defect starts in `create_sun_clips.py`.

Treating each filename prefix before `_imageNNNN` as an independent acquisition gives a provisional
boundary-safe estimate of 143 training, 11 validation and 35 test clips (189 total) under the same
clip length and strides. Review the corrected dry run before accepting these counts.

Consequences:

- Existing SUN propagation, intermediate data, training and evaluation are invalid as final evidence.
- State-aware task `15945448_0` is also using affected SUN data and is not scientifically valid.
- VTUS and MUP tasks can continue.
- Generate corrected SUN data in a new folder; do not overwrite the current data before verification.
- SUN must be rerun from clips through SAM2 propagation, intermediate conversion and training.
- VTUS passed the audit: 390 unique clips, correct splits and round chains, nonempty masks, and no detected SUN-style boundary defect.

## Iterative evaluation

Main script:

```text
CMIG_data_processing/evaluate_iterative_l2d.py
```

Workflow for each test clip:

1. Apply the dataset-specific initial prompt and propagate through SAM2.
2. Give the current video and propagated mask to the rejector.
3. Either accept/stop or select one valid correction frame.
4. If deferring, apply the correct prompt type, propagate again, mask all previously selected positions, and make the next decision.
5. Continue for at most four corrections.
6. Record results at every iteration. Masks do not need to be saved for routine evaluation.

The script auto-loads the common SAM2 checkpoint/config and recovers training configuration from the rejector checkpoint. Treat automatic configuration as convenient but verify prompt type, prompt sizes, channels, architecture and alpha in the saved configuration before launching.

Utilities:

```text
CMIG_data_processing/summarize_iterative_evaluations.py
CMIG_data_processing/plot_figure3_figure4_iterative.py
```

There are SLURM scripts in `hpc/` for loss-ablation, architecture-ablation, Figure 3 and Figure 4 evaluation. Existing results under `CMIG_iterative_evaluation` were produced from the pre-fix models unless their corresponding training run explicitly says `fixed_masks_v2` or later. Do not use the old plots as final evidence. Iterative evaluations must be rerun after final corrected-mask models are selected.

## Experiment records

Existing manifests:

```text
CMIG_l2d_training/experiment_manifests/surrogate_loss_ablation.md
CMIG_l2d_training/experiment_manifests/model_architecture_ablation_mae.md
CMIG_l2d_training/experiment_manifests/figure3_prompt_scale_ablation_mae.md
CMIG_l2d_training/experiment_manifests/figure4_alpha_ablation_mae.md
CMIG_l2d_training/experiment_manifests/iterative_ablation_evaluation.md
```

These are path records, not proof that all experiments are valid. The original surrogate, architecture, Figure 3 and Figure 4 runs predate the propagated-mask fix and must be labeled legacy/invalid or replaced with corrected-mask runs.

For every future run, record at minimum:

- Dataset and exact intermediate-data path.
- Prompt type and box scales, if applicable.
- Architecture.
- Loss and temperature.
- Alpha.
- Batch sampler.
- Backbone and head learning rates.
- Frozen/trainable layers and BN behavior.
- Epoch count and evaluation interval.
- Seed.
- SLURM job ID.
- Output, TensorBoard and checkpoint paths.
- Best validation checkpoint and its final test result.
- Whether the input-mask validator passed.

## TensorBoard

TensorBoard logs are stored under:

```text
CMIG_l2d_training/tensorboard/<run_name>
```

To avoid loading every historical experiment, open one run at a time on the HPC node:

```bash
/hpcfs/users/a1917962/.conda/envs/medsam2/bin/python -m tensorboard.main \
  --logdir /hpcfs/users/a1917962/Medsam2_working/MedSam2_Jun_Ma/CMIG_l2d_training/tensorboard/<run_name> \
  --host 0.0.0.0 \
  --port 6006
```

From the local machine, forward the port through the relevant host, for example:

```bash
ssh -N -L 16006:p2-log-2:6006 a1917962@phoenix-login.adelaide.edu.au
```

Then open:

```text
http://localhost:16006
```

TensorBoard does not start automatically. Ensure the selected run directory actually contains an `events.out.tfevents...` file. Experiment A contains two event files because it was submitted twice.

## Safe continuation checklist for a new agent

Before modifying or running anything:

1. Read this handoff completely.
2. Run `git status --short` and preserve all existing user changes.
3. Do not restore the old tracked SAM2 directory over the user's updated SAM2 tree.
4. Confirm the exact scripts currently present with `rg --files CMIG_data_processing hpc`.
5. Validate the chosen intermediate sample directory and confirm nonzero mask foreground.
6. Inspect the saved checkpoint configuration rather than inferring it from only the directory name.
7. Use `best_val_chosen_cost.pt` for scientific evaluation.
8. Use a new unique experiment name for every new run; never write two jobs into one run folder.
9. Keep validation/test naturally distributed.
10. Ask the user before submitting SLURM jobs.

## Suggested opening instruction on the other machine

Give the new agent this repository and say:

```text
Read CMIG_PROJECT_HANDOFF_2026-09-05.md completely, especially "Critical update: SUN clips cross acquisition boundaries." Inspect the current files and Git status without changing anything. First design a corrected SUN clip dry run that treats each acquisition prefix before `_imageNNNN` as a separate sequence and writes to a new folder. Do not overwrite existing data or submit SLURM jobs. Report the measured counts and exact proposed changes before implementation. VTUS and MUP state-aware job tasks may be analyzed independently.
```
