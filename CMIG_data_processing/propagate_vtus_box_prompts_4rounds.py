#!/usr/bin/env python3
"""Process VTUS clips in batches with four greedy SAM2 box-prompt rounds and 10 equally spaced prompt locations."""

from pathlib import Path

import propagate_sun_box_prompts_4rounds as workflow


REPO_ROOT = Path(__file__).resolve().parents[1]

# Reuse the validated four-round workflow while supplying VTUS-specific paths,
# output naming, a 50-clip batch size, and mandatory GPU execution.
workflow.DEFAULT_DATASET_ROOT = (
    REPO_ROOT / "CMIG_clips/VTUS/vtus_clips_train_stride_15_val_test_stride_30"
)
workflow.DEFAULT_OUTPUT_ROOT = REPO_ROOT / "CMIG_npz_data/vtus"
workflow.DEFAULT_BATCH_SIZE = 50
workflow.DATASET_LABEL = "VTUS"
workflow.EXPERIMENT_PREFIX = "vtus"


if __name__ == "__main__":
    import sys

    if "--require-cuda" not in sys.argv:
        sys.argv.insert(1, "--require-cuda")
    workflow.main()
