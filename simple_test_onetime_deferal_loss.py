import torch
import sys
import os

# Add the current directory to the path to import from L2D_train
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from L2D_train import onetime_deferal_loss

def test_onetime_deferal_loss():
    
    # Test case 1: Rej_logits_1 is better than Rej_logits_2
    rej_logits_1 = torch.tensor([0.7335, 0.5381, -0.1235, -0.8803, 1.3576, 0.0166, -1.0322, -0.5578, 1.0214, -0.5228])
    no_df_dice_batch_1 = torch.tensor([0.8761])
    post_df_dice_batch_1 = torch.tensor([[0.7081, 0.6744, 0.4704, 0.6037, 0.5873, 0.5262, 0.5365, 0.5580, 0.5796]])
    distance_loss_1 = torch.tensor([0.3000, 0.2222, 0.1646, 0.1220, 0.0904, 0.0669, 0.0496, 0.0367, 0.0272])
    beta_1 = 0.003
    
    rej_logits_1 = rej_logits_1.unsqueeze(0)  # Shape: [1, 10]
    loss_1 = onetime_deferal_loss(no_df_dice_batch_1, rej_logits_1, post_df_dice_batch_1, beta_1, distance_loss_1)
    
    # Test case 2: Rej_logits_2 is worse than Rej_logits_1
    rej_logits_2 = torch.tensor([-0.5, -0.3, -0.8, -1.2, 0.1, -0.9, -1.5, -0.7, 0.2, -1.0])
    rej_logits_2 = rej_logits_2.unsqueeze(0)  # Shape: [1, 10]
    
    loss_2 = onetime_deferal_loss(no_df_dice_batch_1, rej_logits_2, post_df_dice_batch_1, beta_1, distance_loss_1)
    
    print(f"Loss 1: {loss_1.item()}")
    print(f"Loss 2: {loss_2.item()}")
    
    # Assert loss_2 is greater than loss_1
    assert loss_2.item() < loss_1.item(), f"Loss 2 ({loss_2.item()}) should be greater than Loss 1 ({loss_1.item()})"
    

if __name__ == "__main__":
    test_onetime_deferal_loss() 