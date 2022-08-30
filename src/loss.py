import torch
from torch import nn


class CrossEntropyLossSoft(nn.Module):
    def __init__(self, ignore_index=None, weight=None):
        super(CrossEntropyLossSoft, self).__init__()
        self.ignore_index = ignore_index
        self.weight = weight

    def unigram_loss(self, inp: torch.Tensor, target: torch.Tensor):
        """
        Calculate loss over the unigram subset

        Args:
            inp:
            target:

        Returns: loss (log softmax)

        """
        return self(inp, target, disable_weight=True)

    def forward(self, input, target, disable_weight=False):
        """
        Args:
            input: (batch, *)
            target: (batch, *) same shape as input, each item must be a valid distribution: target[i, :].sum() == 1.

        """
        if self.ignore_index:
            try:
                target[:, self.ignore_index] = 0
            except IndexError:
                pass

        # With softmax norm
        logprobs = nn.functional.log_softmax(input.view(input.shape[0], -1), dim=1)

        # Calculate logprobs for each class with weights
        if self.weight is not None:
            # Late move to device
            if self.weight.device != logprobs.device:
                self.weight = self.weight.to(logprobs.device)

            if not disable_weight:
                logprobs = logprobs * self.weight

        # Calculate loss
        batchloss = -torch.sum(target.view(target.shape[0], -1) * logprobs, dim=1)

        return torch.mean(batchloss)
