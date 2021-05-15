from torch import nn


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.smooth = 1.0

    def forward(self, output, ground_truth):
        output = output[:, 0].contiguous().view(-1)
        ground_truth = ground_truth[:, 0].contiguous.view(-1)
        intersection = (output * ground_truth).sum()
        dice_score = (2 * intersection + self.smooth) / (
                output.sum() + ground_truth.sum() + self.smooth
        )

        return 1 - dice_score
