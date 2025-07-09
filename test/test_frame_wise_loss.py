import torch
import pytest
from L2D_train import frame_wise_loss

# def test_frame_wise_loss_logistic():
#     # Setup dummy data
#     B, n_e = 4, 3
#     i = 2  # frame index (1-based)
#     no_df_dice_batch = torch.tensor([0.8, 0.6, 0.9, 0.7])  # [B]
#     rejector_logits = torch.randn(B, n_e)  # [B, n_e]
#     post_df_dice_batch = torch.tensor([
#         [0.85, 0.75, 0.65],
#         [0.55, 0.65, 0.75],
#         [0.95, 0.85, 0.80],
#         [0.60, 0.70, 0.80],
#     ])  # [B, n_e]
#     beta = 0.1
#     distance_loss = torch.tensor([0.05, 0.10, 0.15])  # [n_e]

#     # Call function
#     loss = frame_wise_loss(i, no_df_dice_batch, rejector_logits, post_df_dice_batch, beta, distance_loss, loss_type="logistic")

#     # Check output shape
#     assert loss.shape == (B, n_e), f"Expected shape ({B}, {n_e}), got {loss.shape}"
#     # Check values are finite
#     assert torch.isfinite(loss).all(), "Loss contains non-finite values"


# def test_frame_wise_loss_square():
#     # Setup dummy data
#     B, n_e = 2, 2
#     i = 1
#     no_df_dice_batch = torch.tensor([0.5, 0.9])
#     rejector_logits = torch.tensor([[0.2], [0.3]])
#     post_df_dice_batch = torch.tensor([[0.6, 0.7], [0.8, 0.9]])
#     beta = 0.2
#     distance_loss = torch.tensor([0.1, 0.2])

#     loss = frame_wise_loss(i, no_df_dice_batch, rejector_logits, post_df_dice_batch, beta, distance_loss, loss_type="square")
#     assert loss.shape == (B, n_e)
#     assert torch.isfinite(loss).all()
#     # Check that loss is non-negative
#     assert (loss >= 0).all()


# def test_frame_wise_loss_real_data():
#     # Provided real data example
#     rej_logits = torch.tensor([[-0.0983], [0.2770], [0.2200]])
#     no_df_dice_batch = torch.tensor([0.2761, 0.6705, 0.5594])
#     post_df_dice_batch = torch.tensor([
#         [0.7081, 0.6744, 0.4704, 0.6037, 0.5873, 0.5262, 0.5365, 0.5580, 0.5796],
#         [0.9073, 0.7706, 0.7968, 0.7752, 0.7668, 0.7280, 0.7020, 0.7073, 0.6749],
#         [0.7813, 0.4732, 0.6605, 0.6298, 0.5457, 0.5534, 0.5614, 0.5809, 0.5593],
#     ])
#     distance_loss = torch.tensor([0.3000, 0.2222, 0.1646, 0.1220, 0.0904, 0.0669, 0.0496, 0.0367, 0.0272])
#     beta = 0.003

#     i = 1  # Example for first frame/expert
#     loss = frame_wise_loss(i, no_df_dice_batch, rej_logits, post_df_dice_batch, beta, distance_loss, loss_type="logistic")
#     # assert loss.shape == (3, 1)
#     print(loss)
#     # assert torch.isfinite(loss).all()


def test_frame_wise_loss_real_data():
    # Provided real data example
    print("---------------Test 1----------------")
    rej_logits = torch.tensor([[0.0001]])
    no_df_dice_batch = torch.tensor([0.8761])
    post_df_dice_batch = torch.tensor([
        [0.7081, 0.6744, 0.4704, 0.6037, 0.5873, 0.5262, 0.5365, 0.5580, 0.5796]
    ])
    distance_loss = torch.tensor([0.3000, 0.2222, 0.1646, 0.1220, 0.0904, 0.0669, 0.0496, 0.0367, 0.0272])
    beta = 0.003

    i = 1  # Example for first frame/expert
    loss1 = frame_wise_loss(i, no_df_dice_batch, rej_logits, post_df_dice_batch, beta, distance_loss, loss_type="logistic")
    # assert loss.shape == (3, 1)
    print(f"Frame {i} loss: {loss1}")
    
    if no_df_dice_batch[0]< post_df_dice_batch[0][i-1]-beta-distance_loss[i-1]:
        print("Defer to expert")
    #     assert rej_logits[0] >= 0
    else:
        print("Use machine decision")
    #     assert rej_logits[0] < 0
    
    print("---------------Test 2----------------")
    rej_logits = torch.tensor([[-0.0001]])
    no_df_dice_batch = torch.tensor([0.8761])
    post_df_dice_batch = torch.tensor([
        [0.7081, 0.6744, 0.4704, 0.6037, 0.5873, 0.5262, 0.5365, 0.5580, 0.5796]
    ])
    distance_loss = torch.tensor([0.3000, 0.2222, 0.1646, 0.1220, 0.0904, 0.0669, 0.0496, 0.0367, 0.0272])
    beta = 0.003

    i = 1  # Example for first frame/expert
    loss2 = frame_wise_loss(i, no_df_dice_batch, rej_logits, post_df_dice_batch, beta, distance_loss, loss_type="logistic")
    print(f"Frame {i} loss: {loss2}")
    
    if no_df_dice_batch[0]< post_df_dice_batch[0][i-1]-beta-distance_loss[i-1]:
        print("Defer to expert")
    else:
        print("Use machine decision")

    assert loss2 <  loss1
    
    print("---------------Test 3----------------")
    rej_logits = torch.tensor([[0.7651]])
    no_df_dice_batch = torch.tensor([0.8761])
    post_df_dice_batch = torch.tensor([
        [0.7081, 0.6744, 0.4704, 0.6037, 0.5873, 0.5262, 0.5365, 0.5580, 0.5796]
    ])
    distance_loss = torch.tensor([0.3000, 0.2222, 0.1646, 0.1220, 0.0904, 0.0669, 0.0496, 0.0367, 0.0272])
    beta = 0.003

    i = 1  # Example for first frame/expert
    loss3 = frame_wise_loss(i, no_df_dice_batch, rej_logits, post_df_dice_batch, beta, distance_loss, loss_type="logistic")
    # assert loss.shape == (3, 1)
    print(f"Frame {i} loss: {loss3}")

    assert loss3 >  loss1
    
    print("---------------Test 4----------------")
    rej_logits = torch.tensor([[0.7651]])
    no_df_dice_batch = torch.tensor([0.2761])
    post_df_dice_batch = torch.tensor([
        [0.7081, 0.6744, 0.4704, 0.6037, 0.5873, 0.5262, 0.5365, 0.5580, 0.5796]
    ])
    distance_loss = torch.tensor([0.3000, 0.2222, 0.1646, 0.1220, 0.0904, 0.0669, 0.0496, 0.0367, 0.0272])
    beta = 0.003

    i = 1  # Example for first frame/expert
    loss4 = frame_wise_loss(i, no_df_dice_batch, rej_logits, post_df_dice_batch, beta, distance_loss, loss_type="logistic")
    # assert loss.shape == (3, 1)
    print(f"Frame {i} loss: {loss4}")
    
    rej_logits = torch.tensor([[-0.7651]])
    no_df_dice_batch = torch.tensor([0.2761])
    post_df_dice_batch = torch.tensor([
        [0.7081, 0.6744, 0.4704, 0.6037, 0.5873, 0.5262, 0.5365, 0.5580, 0.5796]
    ])
    distance_loss = torch.tensor([0.3000, 0.2222, 0.1646, 0.1220, 0.0904, 0.0669, 0.0496, 0.0367, 0.0272])
    beta = 0.003

    i = 1  # Example for first frame/expert
    loss5 = frame_wise_loss(i, no_df_dice_batch, rej_logits, post_df_dice_batch, beta, distance_loss, loss_type="logistic")
    # assert loss.shape == (3, 1)
    print(f"Frame {i} loss: {loss5}")

    assert loss5 >  loss4
    
    print("---------------Test 5----------------")
    
    rej_logits = torch.tensor([[0.00]])
    no_df_dice_batch = torch.tensor([0.2761])
    post_df_dice_batch = torch.tensor([
        [0.7081, 0.6744, 0.4704, 0.6037, 0.5873, 0.5262, 0.5365, 0.5580, 0.5796]
    ])
    distance_loss = torch.tensor([0.3000, 0.2222, 0.1646, 0.1220, 0.0904, 0.0669, 0.0496, 0.0367, 0.0272])
    beta = 0.003

    i = 1  # Example for first frame/expert
    loss6 = frame_wise_loss(i, no_df_dice_batch, rej_logits, post_df_dice_batch, beta, distance_loss, loss_type="logistic")
    # assert loss.shape == (3, 1)
    print(f"Frame {i} loss: {loss5}")

    assert loss5 >  loss6
    




# def test_frame_wise_loss_penalizes_wrong_predictions():
#     # Simulate a batch of 2
#     # Case 1: Correct - high logit for good base, low for bad base
#     # Case 2: Wrong  - high logit for bad base, low for good base
#     no_df_dice_batch = torch.tensor([0.9, 0.2])  # [good, bad]
#     post_df_dice_batch = torch.tensor([[0.5, 0.8], [0.7, 0.6]])  # [B, n_e]
#     distance_loss = torch.tensor([0.1, 0.2])
#     beta = 0.05
#     i = 1

#     # Correct: high logit for good base, low for bad base
#     rej_logits_correct = torch.tensor([[2.0], [-2.0]])  # [B, 1]
#     # Wrong: high logit for bad base, low for good base
#     rej_logits_wrong = torch.tensor([[-2.0], [2.0]])    # [B, 1]

#     loss_correct = frame_wise_loss(i, no_df_dice_batch, rej_logits_correct, post_df_dice_batch, beta, distance_loss, loss_type="logistic")
#     loss_wrong = frame_wise_loss(i, no_df_dice_batch, rej_logits_wrong, post_df_dice_batch, beta, distance_loss, loss_type="logistic")

#     # The loss for the wrong scenario should be higher
#     assert loss_wrong.mean() > loss_correct.mean(), f"Expected higher loss for wrong predictions, got {loss_wrong.mean()} <= {loss_correct.mean()}" 