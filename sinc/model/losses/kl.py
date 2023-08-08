import torch


class KLLoss:
    def __init__(self):
        pass

    def __call__(self, q, p, reduce_fx='mean'):
        div = torch.distributions.kl_divergence(q, p)
        if reduce_fx == 'mean':
            return div.mean()
        else:
            return div

    def __repr__(self):
        return "KLLoss()"


class KLLossMulti:
    def __init__(self):
        self.klloss = KLLoss()

    def __call__(self, qlist, plist):
        return sum([self.klloss(q, p)
                    for q, p in zip(qlist, plist)])

    def __repr__(self):
        return "KLLossMulti()"
