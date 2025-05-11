# Re-run after kernel reset: Training with alpha/beta-based deferral costs and evaluation

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# Cost function: cj(x, y) = alpha_j * 1[h_j(x) ≠ y] + beta_j
def compute_costs(acc_post_def_batch, alpha, beta):
    """
    batch_expert_preds: [batch, n_experts]
    y_true: [batch]
    Returns: costs [batch, n_experts]
    """
    incorrect = (1-acc_post_def_batch)
    cost = alpha * incorrect + beta  # shape [batch, n_experts]
    return cost

# Second-stage loss
def deferral_loss(acc_no_def_batch, rejector_logits, acc_post_def_batch, alpha=1.0, beta=0.1):
    """
    predictor_logits: [batch, n_classes]
    rejector_logits: [batch, n_experts]
    """
    batch_size, n_experts = rejector_logits.shape

    # Predictor prediction
    
    correct = acc_no_def_batch  # [batch]

    # r(x, 0) = 0, r(x, j) = -rj(x)
    # Create full logits over [0, ..., n_experts]
    r_scores = torch.zeros(batch_size, n_experts + 1, device=rejector_logits.device)
    r_scores[:, 1:] = -rejector_logits  # negative because paper defines r(x, j) = -rj(x)

    # Softmax over [0 (predict), 1...n_experts]
    r_probs = F.log_softmax(r_scores, dim=1)

    # Compute costs
    cost = compute_costs(acc_post_def_batch, alpha, beta)  # [B, n_experts]

    # Loss term 1: when predictor is correct
    loss_predict = -r_probs[range(batch_size), 0] * correct  # [batch]

    # Loss term 2: for deferrals (cost-weighted)
    loss_defer = 0
    for j in range(n_experts):
        cj = cost[:, j]
        pj = r_probs[range(batch_size), j + 1]  # expert j has index j+1 in r_probs
        loss_defer += -cj * pj  # [batch]

    # Total loss
    total_loss = (loss_predict + loss_defer).mean()
    return total_loss

# Set seed for reproducibility
torch.manual_seed(42)

# ----- Synthetic Data Parameters -----
num_videos = 2000  # Number of chunks
num_frames = 5
feature_dim = 8

# Generate synthetic features
X = torch.randn(num_videos, feature_dim)

# Generate accuracy scores
acc_no_def = torch.rand(num_videos) * 0.5 + 0.5
noise = (torch.rand(num_videos, 4) - 0.5) * 0.2
acc_post_def = acc_no_def.unsqueeze(1) + noise
acc_post_def = torch.clamp(acc_post_def, min=0.0, max=1.0)

print(f"Generated synthetic data for {num_videos} chunks")
print(f"Feature dimension: {feature_dim}")
print(f"Number of experts: {acc_post_def.shape[1]}")

# ----- Dataset and DataLoader -----
class SyntheticDataset(torch.utils.data.Dataset):
    def __init__(self, X, acc_no_def, acc_post_def):
        self.X = X
        self.acc_no_def = acc_no_def
        self.acc_post_def = acc_post_def
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.acc_no_def[idx], self.acc_post_def[idx]

# Create train-test split
train_ratio = 0.8
num_train = int(train_ratio * num_videos)

# Create datasets
train_dataset = SyntheticDataset(
    X[:num_train],
    acc_no_def[:num_train],
    acc_post_def[:num_train]
)

test_dataset = SyntheticDataset(
    X[num_train:],
    acc_no_def[num_train:],
    acc_post_def[num_train:]
)

# Create dataloaders
batch_size = 8
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False
)

# Define alpha and beta for cost function
alpha = 1  # penalty for expert error
beta = 0.001   # fixed cost of deferral

incorrect = 1 - acc_post_def


# Cost = alpha * error + beta
cost_defer_all = alpha * incorrect + beta  # shape: [N, 4]
cost_base_all = 1.0 - acc_no_def           # shape: [N]


# ----- Define Model -----
class RejectorModel(nn.Module):
    def __init__(self, input_dim, n_experts):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, n_experts)
        )

    def forward(self, x):
        return self.net(x)  # logits for deferring to each expert

rejector = RejectorModel(input_dim=feature_dim, n_experts=4)
optimizer = optim.Adam(rejector.parameters(), lr=0.01)

# ----- Training Loop -----
epochs = 100
rejector.train()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
rejector = rejector.to(device)  # Move model to device

for epoch in range(epochs):
    total_loss = 0.0
    
    for features, acc_no_def_batch, acc_post_def_batch in train_loader:
        # Move data to device if using GPU
        features = features.to(device)
        acc_no_def_batch = acc_no_def_batch.to(device)
        acc_post_def_batch = acc_post_def_batch.to(device)
        
        # Forward pass
        rej_logits = rejector(features)  # This will now output shape [batch_size, n_experts]
        
        # Compute loss
        loss = deferral_loss(acc_no_def_batch, rej_logits, acc_post_def_batch, alpha=3.0, beta=0.001)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

# ----- Evaluation -----
def evaluate_model(model, test_loader, device):
    model.eval()  # Set model to evaluation mode
    correct = 0
    total_regret = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for features, acc_no_def_batch, acc_post_def_batch in test_loader:
            # Move data to device
            features = features.to(device)
            acc_no_def_batch = acc_no_def_batch.to(device)
            acc_post_def_batch = acc_post_def_batch.to(device)
            
            # Get model predictions
            rej_logits = model(features)
            scores = torch.zeros(features.shape[0], rej_logits.shape[1] + 1, device=device)
            scores[:, 1:] = -rej_logits  # negative because paper defines r(x, j) = -rj(x)
            chosen_actions = torch.argmax(scores, dim=1)
            
            # Get best actions
            all_accs = torch.cat([acc_no_def_batch.unsqueeze(1), acc_post_def_batch], dim=1)
            best_actions = torch.argmax(all_accs, dim=1)
            print("Best Actions:", best_actions)
            print("Chosen Actions:", chosen_actions)
            
            # Calculate metrics
            correct += (chosen_actions == best_actions).sum().item()
            
            # Calculate regret
            chosen_accs = torch.gather(all_accs, 1, chosen_actions.unsqueeze(1)).squeeze(1)
            best_accs = torch.gather(all_accs, 1, best_actions.unsqueeze(1)).squeeze(1)
            regret = best_accs - chosen_accs
            total_regret += regret.sum().item()
            
            total_samples += features.shape[0]
    
    selection_accuracy = correct / total_samples
    mean_regret = total_regret / total_samples
    
    return selection_accuracy, mean_regret

# Run evaluation
print("\nEvaluating model on test set...")
selection_accuracy, mean_regret = evaluate_model(rejector, test_loader, device)
print(f"Test Set Evaluation with alpha={alpha}, beta={beta}:")
print(f"  Best frame selection accuracy: {selection_accuracy:.2%}")
print(f"  Mean regret: {mean_regret:.4f}")


