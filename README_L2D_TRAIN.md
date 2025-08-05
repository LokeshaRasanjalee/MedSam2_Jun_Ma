```bash
python L2D_data_creation_2.py \
    --base_video_dir ./dataset/Images/ \
    --input_mask_dir ./dataset/Masks/ \
    --data_pkl_dir ./dataset/dataset_pkl/ \
    -o ./dataset/out/ \
    --experiment_name 'npz_creation' \
    --array_id 0 \
    --num_groups 1 \
    --p1 0.390 \
    --p99 0.875 \
    --loss_type 'iou'
```

## Parameter Explanations

### Required Parameters

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `--base_video_dir` | str | Directory containing video frames (JPEG files) | 
| `--input_mask_dir` | str | Directory containing ground truth masks (PNG files) | 
| `--data_pkl_dir` | str | Directory containing pre-processed pickle files | 
| `-o, --output_mask_dir` | str | Directory to save output masks and logs | 
| `--experiment_name` | str | Name for the experiment (used in logging) | 
| `--p1` | float | 234.0 | P1 parameter for deferral loss (percentile 1) |
| `--p99` | float | 8384.0 | P99 parameter for deferral loss (percentile 99) |
| `--loss_type` | str | 'sam' | Loss type: 'sam', 'dice', or 'iou' |

### Processing Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--array_id` | int | 0 | Array ID for batch processing (0-based) |
| `--num_groups` | int | 0 | Number of groups for data splitting (0 = use all data) |
