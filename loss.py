from torch import nn
import torch


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.smooth = 1.0

    def forward(self, output, ground_truth):
        output = output[:, 0].contiguous().view(-1)
        ground_truth = ground_truth[:, 0].contiguous().view(-1)

        intersection = torch.sum(torch.mul(output, ground_truth), dim=1)
        union = torch.sum(output + ground_truth, dim=1)

        loss = 1 - (intersection + self.smooth) / (union + self.smooth)
        return loss.sum()
