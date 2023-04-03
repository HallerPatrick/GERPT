import torch
from torch import nn

import functorch


def logsigsoftmax(logits):
    """
    Computes sigsoftmax from the paper - https://arxiv.org/pdf/1805.10829.pdf
    """
    max_values = torch.max(logits, 1, keepdim=True)[0]
    exp_logits_sigmoided = torch.exp(logits - max_values) * torch.sigmoid(logits)
    sum_exp_logits_sigmoided = exp_logits_sigmoided.sum(1, keepdim=True)
    log_probs = (
        logits
        - max_values
        + torch.log(torch.sigmoid(logits))
        - torch.log(sum_exp_logits_sigmoided)
    )
    return log_probs


class CrossEntropyLossSoft(nn.Module):
    def __init__(self, ignore_index=None, weight=None):
        super(CrossEntropyLossSoft, self).__init__()
        self.ignore_index = ignore_index
        self.weight = weight
        self.batched_sigmoid = functorch.vmap(nn.functional.logsigmoid)

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
        # logprobs = logsigsoftmax(input.view(input.shape[0], -1))
        # logprobs = nn.functional.logsigmoid(input.view(input.shape[0], -1))
        # logprobs = self.batched_sigmoid(input.view(input.shape[0], -1))

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
