import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TunableLabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, total_smooth_capacity=0.1, uniform_smooth=False, ce=False):
        super().__init__()
        self.uniform_smooth = uniform_smooth
        self.ce = ce
        # assert total_smooth_capacity <= 0.5, "smooth is too large"
        self.total_smooth_capacity = total_smooth_capacity

    def forward(self, y_pred: torch.Tensor, target: torch.Tensor, per_cls_smooth: torch.Tensor) -> torch.Tensor:
        assert per_cls_smooth.shape == y_pred.shape, "smoothing values should match each cell in y_pred"
        confidence = 1.0 - self.total_smooth_capacity
        per_cls_smooth = per_cls_smooth.to(y_pred.device) * self.total_smooth_capacity

        logprobs = F.log_softmax(y_pred, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        if not self.uniform_smooth:
            print(per_cls_smooth[0].max())
            smooth_loss = -(per_cls_smooth * logprobs).mean(dim=-1)  # scale smooth with per-class values
        else:
            smooth_loss = -logprobs.mean(dim=-1)  # scale smooth with per-class values
        loss = confidence * nll_loss + smooth_loss

        if self.ce:
            loss = nll_loss  # cancel everything - normal ce

        return loss.mean()
