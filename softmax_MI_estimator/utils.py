import torch


def own_softmax(x, label_proportions):
    if not isinstance(label_proportions, torch.Tensor):
        label_proportions = torch.tensor(label_proportions).to('cuda')
    x_exp = torch.clamp(torch.exp(x), max=1e10, min=1e-5)
    weighted_x_exp = x_exp * label_proportions
    # weighted_x_exp = x_exp
    x_exp_sum = torch.sum(weighted_x_exp, 1, keepdim=True)

    return x_exp / (x_exp_sum + 1e-5)
