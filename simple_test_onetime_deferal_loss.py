
import torch
import torch.nn.functional as F
from L2D_train import onetime_deferal_loss, onetime_deferal_loss_normalized_weights, onetime_deferal_loss_normalized_weights_0_1

def compute_adjusted_scores(acc_no_def_batch, acc_post_def_batch, beta, distance_loss):
    """Compute adjusted Dice scores for validation."""
    post_diff_adjust = acc_post_def_batch - distance_loss - beta
    adjusted_scores = torch.cat([acc_no_def_batch.unsqueeze(1), post_diff_adjust], dim=1)
    return adjusted_scores

def run_test_case(test_name, acc_no_def_batch, rejector_logits, acc_post_def_batch, beta, distance_loss, expected_behavior):
    """Run a test case and print results."""
    print(f"\n=== {test_name} ===")
    try:
        # loss = onetime_deferal_loss(acc_no_def_batch, rejector_logits, acc_post_def_batch, beta, distance_loss)
        loss = onetime_deferal_loss_normalized_weights_0_1(acc_no_def_batch, rejector_logits, acc_post_def_batch, beta, distance_loss)

        adjusted_scores = compute_adjusted_scores(acc_no_def_batch, acc_post_def_batch, beta, distance_loss)
        optimal_index = adjusted_scores.argmax().item()
        max_logit_index = rejector_logits[0].argmax().item()  # Access first batch element
        print(f"Loss: {loss.item():.6f}")
        print(f"Adjusted Scores: {adjusted_scores[0].tolist()}")
        print(f"Logits: {rejector_logits[0].tolist()}")
        print(f"Optimal Index (max adjusted score): {optimal_index}")
        print(f"Max Logit Index: {max_logit_index}")
        print(f"Expected Behavior: {expected_behavior}")
        return loss.item()
    
    except Exception as e:
        print(f"Error: {str(e)}")
        print("FAIL: Exception raised")
        return None

# Common inputs
acc_no_def_batch_1 = torch.tensor([0.8761])
acc_post_def_batch_1 = torch.tensor([[0.7081, 0.6744, 0.4704, 0.6037, 0.5873, 0.5262, 0.5365, 0.5580, 0.5796]])
beta_1 = 0.003
distance_loss_1 = torch.tensor([[0.1239, 0.2017, 0.4057, 0.2714, 0.3887, 0.4488, 0.4385, 0.4170, 0.3954]])

# Test Case 1_1: No deferral : Optimal prediction (max logit at i=0, lowest cost)
rejector_logits_1 = torch.tensor([[2.0, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0, -0.1, -0.2, -0.3]])
loss_1_1 = run_test_case(
    "Test Case 1_1: No deferral : Highest Logit Matches Optimal Option",
    acc_no_def_batch_1,
    rejector_logits_1,
    acc_post_def_batch_1,
    beta_1,
    distance_loss_1,
    "Optimal prediction (max logit at i=0, lowest cost)"
)

# Test Case 1_2: No deferral : Suboptimal prediction, moderate cost (max logit at i=4)
rejector_logits_2 = torch.tensor([[0.5, 0.4, 0.3, 1.2, 2.0, 0.1, 0.0, -0.1, -0.2, -0.3]])
loss_1_2 = run_test_case(
    "Test Case 1_2: No deferral : Suboptimal prediction, moderate cost (max logit at i=4)",
    acc_no_def_batch_1,
    rejector_logits_2,
    acc_post_def_batch_1,
    beta_1,
    distance_loss_1,
    "Medium loss"
)

# Test Case 1_3: No deferral : Suboptimal prediction, high cost (max logit at i=5)
rejector_logits_3 = torch.tensor([[0.5, 0.4, 0.3, 1.2, 0.5, 2.5, 0.0, -0.1, -0.2, -0.3]])
loss_1_3 = run_test_case(
    "Test Case 1_3: No deferral : Suboptimal prediction, high cost (max logit at i=5)",
    acc_no_def_batch_1,
    rejector_logits_3,
    acc_post_def_batch_1,
    beta_1,
    distance_loss_1,
    "Highest loss"
)

# Test Case 2_1: Fixed logits : Lowest loss
acc_post_def_batch_2_1 = torch.tensor([[0.7081, 0.6744, 0.4704, 0.6037, 0.5873, 0.5262, 0.5365, 0.5580, 0.5796]])
loss_2_1 = run_test_case(
    "Test Case 2_1: Fixed logits : Lowest loss",
    acc_no_def_batch_1,
    rejector_logits_1,
    acc_post_def_batch_2_1,
    beta_1,
    distance_loss_1,
    "Lowest loss"
)

# Test Case 2_2: Fixed logits : Medium loss
acc_post_def_batch_2_2 = torch.tensor([[0.9081, 0.8744, 0.6704, 0.8037, 0.7873, 0.7262, 0.7365, 0.7580, 0.7796]])
loss_2_2 = run_test_case(
    "Test Case 2_2: Fixed logits : Medium loss",
    acc_no_def_batch_1,
    rejector_logits_1,
    acc_post_def_batch_2_2,
    beta_1,
    distance_loss_1,
    "Medium loss"
)

# Test Case 2_3: Fixed logits : Highest loss
acc_post_def_batch_2_3 = torch.tensor([[1.1081, 1.0744, 0.8704, 1.0037, 0.9873, 0.9262, 0.9365, 0.9580, 0.9796]])
loss_2_3 = run_test_case(
    "Test Case 2_3: Fixed logits : Highest loss",
    acc_no_def_batch_1,
    rejector_logits_1,
    acc_post_def_batch_2_3,
    beta_1,
    distance_loss_1,
    "Highest loss"
)


# Compare losses for sensitivity tests
if all(x is not None for x in [loss_1_1, loss_1_2, loss_1_3, loss_2_1, loss_2_2, loss_2_3]):
    print("\n=== Sensitivity Test Comparison ===")
    print(f"PASS: Sensitivity check" if loss_1_1 < loss_1_2 < loss_1_3 else "FAIL: Sensitivity check (expected loss_1_1 < loss_1_2 < loss_1_3)")
    print(f"PASS: Sensitivity check" if loss_2_1 < loss_2_2 < loss_2_3 else "FAIL: Sensitivity check (expected loss_2_1 < loss_2_2 < loss_2_3)")


if __name__ == "__main__":
    print("Running all test cases for onetime_deferal_loss...")