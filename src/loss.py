from collections import namedtuple
from typing import Optional, Union

import torch
from torch import nn

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


_ASMoutput = namedtuple("_ASMoutput", ["output", "loss"])


class AdaptiveLogSoftmaxWithLossSoft(nn.AdaptiveLogSoftmaxWithLoss):
    def __init__(
        self,
        in_features,
        n_classes,
        cutoffs,
        div_value=4.0,
        head_bias=True,
        weight: Optional[torch.Tensor] = None,
    ):
        super(AdaptiveLogSoftmaxWithLossSoft, self).__init__(
            in_features, n_classes, cutoffs, div_value, head_bias
        )

        self.weight = weight
        
        # We cannot just simply apply the weights to a general output probability
        # Construct weight vectors for head and tails
        if self.weight is not None:
            self.head_weight = torch.concat((self.weight[0:self.cutoffs[0]], torch.ones((self.n_clusters))))
            # self.head_weight = self.weight[0:self.cutoffs[0]]
            self.tail_weights = []
            for i in range(0, len(cutoffs)):
                self.tail_weights.append(self.weight[self.cutoffs[i]:self.cutoffs[i+1]])


    def forward(self, input_, target_):
        targ_dim = target_.dim()

        use_ngme = False

        if targ_dim == 1:
            if input_.size(0) != target_.size(0):
                raise RuntimeError(
                    "Input and target should have the same size "
                    "in the batch dimension."
                )
            if input_.dim() != 2:
                raise RuntimeError(
                    "1D target tensor expects 2D input tensors, "
                    "but found inputs with size",
                    input_.size(),
                )
        elif targ_dim == 0:
            if input_.dim() != 1:
                raise RuntimeError(
                    "0D target tensor expects 1D input tensors, "
                    "but found inputs with size",
                    input_.size(),
                )
        else:
            if targ_dim == 2:
                use_ngme = True

        is_batched = targ_dim > 0
        input = input_ if is_batched else input_.unsqueeze(0)
        target = target_ if is_batched else target_.unsqueeze(0)

        used_rows = 0
        if use_ngme:
            batch_size = target.size(1)
        else:
            batch_size = target.size(0)

        output = input.new_zeros(batch_size)
        gather_inds = target.new_empty(batch_size)

        cutoff_values = [0] + self.cutoffs
        for i in range(len(cutoff_values) - 1):

            low_idx = cutoff_values[i]
            high_idx = cutoff_values[i + 1]

            # NOTE: Binary mask of the target values that are in the current cutoff
            target_mask = (target >= low_idx) & (target < high_idx)

            # If target mask is a 2d-tensor, like ngme,
            # then row_indices is also a 2d tensor with x, y for col and row
            # NOTE: row_indices = Output idxs that are found in the current cutoff
            if use_ngme:
                row_indices = [t_mask.nonzero().squeeze() for t_mask in target_mask]
            else:
                row_indices = target_mask.nonzero().squeeze()

                if row_indices.numel() == 0:
                    continue

            # NOTE: target[target_mask] = Actual token idx values
            if use_ngme:
                target_masks = [t[mask] for t, mask in zip(target, target_mask)]

            if i == 0:
                # NOTE: Apply the idxs to current idx position
                if use_ngme:
                    for row, target_mask in zip(row_indices, target_masks):
                        # if self.ignore_indexes:
                        #     for ignore_index in self.ignore_indexes:
                        #         row = row[row != ignore_index]

                        gather_inds.index_copy_(0, row, target_mask)
                else:
                    gather_inds.index_copy_(0, row_indices, target[target_mask])
            else:
                if use_ngme:
                    for row, target_mask in zip(row_indices, target_masks):

                        # if self.ignore_indexes:
                        #     for ignore_index in self.ignore_indexes:
                        #         row = row[row != ignore_index]

                        if row.numel() == 0 or row.dim() == 0 or len(row) == 0:
                            continue

                        relative_target = target_mask - low_idx

                        input_subset = input.index_select(0, row)
                        cluster_output = self.tail[i - 1](input_subset)
                        cluster_index = self.shortlist_size + i - 1

                        gather_inds.index_fill_(0, row, cluster_index)

                        cluster_logprob = nn.functional.log_softmax(
                            cluster_output, dim=1
                        )
                        if self.weight is not None:
                            if self.tail_weights[i-1].device != cluster_logprob.device:
                                self.tail_weights[i-1] = self.tail_weights[i-1].to(cluster_logprob.device)

                            cluster_logprob = cluster_logprob * self.tail_weights[i-1]

                        local_logprob = cluster_logprob.gather(
                            1, relative_target.unsqueeze(1)
                        ).half()
                        output.index_copy_(0, row, local_logprob.squeeze(1))
                else:
                    relative_target = target[target_mask] - low_idx

                    input_subset = input.index_select(0, row_indices)
                    cluster_output = self.tail[i - 1](input_subset)
                    cluster_index = self.shortlist_size + i - 1

                    gather_inds.index_fill_(0, row_indices, cluster_index)

                    cluster_logprob = nn.functional.log_softmax(cluster_output, dim=1)
                    local_logprob = cluster_logprob.gather(
                        1, relative_target.unsqueeze(1)
                    ).half()
                    output.index_copy_(0, row_indices, local_logprob.squeeze(1))

            if use_ngme:
                for row in row_indices:
                    used_rows += row.numel()
            else:
                used_rows += row_indices.numel()

        if use_ngme:
            batch_size = batch_size * target.size(0)

        if used_rows != batch_size:
            raise RuntimeError(
                "Target values should be in [0, {}], "
                "but values in range [{}, {}] "
                "were found. ".format(
                    self.n_classes - 1, target.min().item(), target.max().item()
                )
            )

        head_output = self.head(input)
        head_logprob = nn.functional.log_softmax(head_output, dim=1)
        
        if self.weight is not None:
            if self.head_weight.device != head_logprob.device:
                self.head_weight = self.head_weight.to(head_logprob.device)

            head_logprob = head_logprob * self.head_weight

        # Gather the log-probabilities of the correct classes
        output += head_logprob.gather(1, gather_inds.unsqueeze(1)).squeeze()

        loss = (-output).mean()

        if not is_batched:
            output = output.squeeze(0)

        return _ASMoutput(output, loss)

    def init_devices(self, device):
        self.head.to(device)
        self.tail.to(device)


class CrossEntropyLossSoft(nn.Module):
    def __init__(self, ignore_index=None, weight=None):
        super(CrossEntropyLossSoft, self).__init__()
        self.ignore_index = ignore_index
        self.weight = weight

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
