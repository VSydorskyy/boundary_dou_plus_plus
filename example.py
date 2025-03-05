import torch

from src import BoundaryDoULoss


def run_test():
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define sample input tensors
    sample_2d = torch.randint(
        0, 2, (1, 10, 10), dtype=torch.long, device=device
    )  # Binary 2D tensor
    sample_3d = torch.randint(
        0, 2, (1, 10, 10, 10), dtype=torch.long, device=device
    )  # Binary 3D tensor

    # Define model predictions (logits before softmax)
    pred_2d = torch.randn(1, 2, 10, 10, device=device)
    pred_3d = torch.randn(1, 2, 10, 10, 10, device=device)

    # Initialize loss functions
    loss_basic = BoundaryDoULoss(n_classes=2, use_reformulated_version=True).to(device)
    loss_gamma_2 = BoundaryDoULoss(
        n_classes=2, use_reformulated_version=True, gamma=2.0
    ).to(device)

    # Run loss calculations
    loss_2d_basic = loss_basic(pred_2d, sample_2d)
    loss_2d_gamma = loss_gamma_2(pred_2d, sample_2d)
    loss_3d_basic = loss_basic(pred_3d, sample_3d)
    loss_3d_gamma = loss_gamma_2(pred_3d, sample_3d)

    # Print results
    print(f"Loss 2D (Basic): {loss_2d_basic.item():.6f}")
    print(f"Loss 2D (Gamma=2.0): {loss_2d_gamma.item():.6f}")
    print(f"Loss 3D (Basic): {loss_3d_basic.item():.6f}")
    print(f"Loss 3D (Gamma=2.0): {loss_3d_gamma.item():.6f}")


if __name__ == "__main__":
    run_test()
