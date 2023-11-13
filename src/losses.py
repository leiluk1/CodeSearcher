import torch


class TextCodeContrastiveLoss(torch.nn.Module):

    def __init__(self, smooth=0):
        super().__init__()
        self.H = torch.nn.CrossEntropyLoss(label_smoothing=smooth, reduction='mean')

    def forward(self, text_batch: torch.Tensor, code_batch: torch.Tensor, T=0.08):
        """

        :param text_batch: (N, D) logits batch with rows having norm=1
        :param code_batch: (N, D) logits batch with rows having norm=1
        :param T: Learnable temperature
        :return: scalar
        """

        text2code = text_batch @ code_batch.T
        code2text = code_batch @ text_batch.T

        gt_t = torch.arange(text_batch.shape[0], device=text_batch.device)
        gt_c = torch.arange(code_batch.shape[0], device=code_batch.device)
        loss = 0.5 * self.H(text2code / T, gt_t) + 0.5 * self.H(code2text / T, gt_c) # 20 is just a scaling constant

        return loss


if __name__ == '__main__':
    # proof of convergence (can be treated as test)
    net = torch.nn.Linear(256, 64)
    batch_t = torch.rand(16, 256); batch_t /= batch_t.pow(2).sum(dim=1, keepdim=True).sqrt()
    batch_c = torch.rand(16, 256); batch_c /= batch_c.pow(2).sum(dim=1, keepdim=True).sqrt()

    loss_fn = TextCodeContrastiveLoss()
    opt = torch.optim.AdamW(net.parameters())

    for i in range(10):
        tb, cb = net(batch_t), net(batch_c)
        opt.zero_grad()
        _loss = loss_fn(tb, cb)
        _loss.backward()
        opt.step()
        print(_loss.item())
    print(tb @ cb.T)