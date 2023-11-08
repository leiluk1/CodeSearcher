import numpy as np
import torch


class TextCodeContrastiveLoss(torch.nn.Module):

    def __init__(self, smooth=0):
        super().__init__()
        self.H = torch.nn.CrossEntropyLoss(label_smoothing=smooth)

    def forward(self, text_batch: torch.Tensor, code_batch: torch.Tensor, T):
        """

        :param text_batch: (N, D) logits batch with rows having norm=1
        :param code_batch: (N, D) logits batch with rows having norm=1
        :param T: Learnable temperature
        :return: scalar
        """

        similarities = text_batch @ code_batch.T

        p_t2c = torch.nn.functional.softmax(similarities / T, dim=-1)
        p_c2t = torch.nn.functional.softmax(similarities.T / T, dim=-1)

        gt_t = torch.eye(text_batch.shape[0])
        gt_c = torch.eye(code_batch.shape[0])
        loss = 0.5 * self.H(p_t2c, gt_t) + 0.5 * self.H(p_c2t, gt_c)

        return loss
