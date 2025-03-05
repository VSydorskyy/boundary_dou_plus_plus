"""
Boundary DoU Loss ++ Implementation

This code implements an improved version of the Boundary Dice-over-Union (DoU) loss function, a metric designed for
boundary-aware segmentation performance. The implementation builds upon the original Boundary DoU loss from:

Original Source: https://github.com/sunfan-bvb/BoundaryDoULoss/blob/main/TransUNet/utils.py#L270

Modifications:
- Reformulated version with a gamma parameter for focal-like weighting
- Support for both 2D and 3D inputs
- Improved computational efficiency
- Code readability and modularization improvements

References:
[1] Sun, F., Luo, Z., Li, S.: "Boundary Difference over Union Loss for Medical Image
    Segmentation." In: International Conference on Medical Image Computing and
    Computer-Assisted Intervention, pp. 292–301 (2023). Springer.
[2] Yeung, M., Rundo, L., Nan, Y., Sala, E., Sch¨onlieb, C.-B., Yang, G.: "Calibrating the Dice Loss to Handle Neural
    Network Overconfidence for Biomedical Image Segmentation." Journal of Digital Imaging 36(2), 739–752 (2023).
"""

import torch
import torch.nn as nn


class BoundaryDoULoss(nn.Module):
    """
    Improved Implementation of Boundary DoU Loss for segmentation tasks.

    This enhanced loss function improves segmentation performance by focusing on boundary regions.
    It supports both the original and reformulated versions with an optional gamma parameter.

    Args:
        n_classes (int): Number of segmentation classes.
        use_reformulated_version (bool): Whether to use the reformulated version.
        gamma (float): Weighting factor for reformulated version (default: 1.0). Should be 1.0
                      for the original version.
    """

    def __init__(
        self, n_classes, use_reformulated_version: bool = False, gamma: float = 1.0
    ):
        if not use_reformulated_version and gamma != 1.0:
            raise ValueError(
                "gamma should be 1.0 for the original version of BoundaryDoULoss"
            )

        super(BoundaryDoULoss, self).__init__()
        self.n_classes = n_classes
        self.use_reformulated_version = use_reformulated_version
        self.gamma = gamma

    def _one_hot_encoder(self, input_tensor):
        """
        Converts a label tensor into a one-hot encoded tensor.
        """
        return torch.cat(
            [(input_tensor == i).unsqueeze(1) for i in range(self.n_classes)], dim=1
        ).float()

    def _compute_Y_2d(self, target):
        """
        Computes the boundary region for 2D segmentation maps using a 3x3 kernel.
        """
        kernel = (
            torch.Tensor([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
            .to(target)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        Y = torch.conv2d(target.unsqueeze(1), kernel, padding=1).squeeze(1) * target
        Y[Y == 5] = 0
        return Y

    def _compute_Y_3d(self, target):
        """
        Computes the boundary region for 3D segmentation maps using a 3x3x3 kernel.
        """
        kernel = (
            torch.Tensor(
                [
                    [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                    [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
                    [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                ]
            )
            .to(target)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        Y = torch.conv3d(target.unsqueeze(1), kernel, padding=1).squeeze(1) * target
        Y[Y == 7] = 0
        return Y

    def _adaptive_size(self, score, target):
        """
        Computes the Boundary DoU loss by dynamically adjusting the weight parameter alpha.
        """
        if target.dim() == 4:
            Y = self._compute_Y_3d(target)
        elif target.dim() == 3:
            Y = self._compute_Y_2d(target)
        else:
            raise ValueError("Input tensor should be 3D or 4D")
        C, S = torch.count_nonzero(Y), torch.count_nonzero(target)
        smooth = 1e-5
        alpha = min(2 * (1 - (C + smooth) / (S + smooth)) - 1, 0.8)

        if self.use_reformulated_version:
            s_i = torch.sum(score * target)
            # Refers to Eq. 4 in the original paper if gamma != 1.0
            s_d_fn = torch.sum((target * (1 - score)).pow(self.gamma))
            s_d_fp = torch.sum((score * (1 - target)).pow(self.gamma))
            s_d = s_d_fn + s_d_fp
            loss = (s_d + smooth) / (s_d + (1 - alpha) * s_i + smooth)
        else:
            intersect = torch.sum(score * target)
            y_sum = torch.sum(target * target)
            z_sum = torch.sum(score * score)
            loss = (z_sum + y_sum - 2 * intersect + smooth) / (
                z_sum + y_sum - (1 + alpha) * intersect + smooth
            )

        return loss

    def forward(self, inputs, target):
        """
        Forward pass to compute the Boundary DoU loss.

        Args:
            inputs (torch.Tensor): Model predictions (logits before softmax).
            target (torch.Tensor): Ground truth segmentation map.

        Returns:
            torch.Tensor: Computed loss value.
        """
        inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        assert (
            inputs.size() == target.size()
        ), f"Predict {inputs.size()} & Target {target.size()} shape do not match"

        return (
            sum(
                self._adaptive_size(inputs[:, i], target[:, i])
                for i in range(self.n_classes)
            )
            / self.n_classes
        )
