import torch


def softemax(x, device="cuda"):
    """
    Calculate SoftEMax: y_SoftEMax(x) = e^x / sum(e^x_i for i=1 to e)
    """
    x = x.to(device)
    exp_x = torch.exp(x)
    summation = torch.sum(
        torch.exp(torch.linspace(0, torch.e, steps=int(torch.e), device=device))
    )
    return exp_x / summation


def inversesoftemax(x, device="cuda"):
    """
    Calculate InverseSoftEMax: y_InverseSoftEMax(x) = e^(-x) / sum(e^x_i for i=1 to e)
    """
    x = x.to(device)
    exp_neg_x = torch.exp(-x)
    summation = torch.sum(
        torch.exp(torch.linspace(0, torch.e, steps=int(torch.e), device=device))
    )
    return exp_neg_x / summation


def plussoftemax(x, device="cuda"):
    """
    Calculate PlusSoftEMax: y_PlusSoftEMax(x) = e^x / (1 + sum(e^x_i for i=1 to e))
    """
    x = x.to(device)
    exp_x = torch.exp(x)
    summation = 1 + torch.sum(
        torch.exp(torch.linspace(0, torch.e, steps=int(torch.e), device=device))
    )
    return exp_x / summation


def PIESMax(x, device="cuda"):
    """
    Calculate PIESMax: y_PIESMax(x) = e^(-x) / (1 + sum(e^x_i for i=1 to e))
    """
    x = x.to(device)
    e_neg_x = torch.e**-x
    summation = 1 + torch.sum(
        torch.exp(torch.linspace(0, torch.e, steps=int(torch.e), device=device))
    )
    return e_neg_x / summation


x = torch.tensor([0.5, 1.0, 1.5], device="cuda")

y_softemax = softemax(x)
y_inversesoftemax = inversesoftemax(x)
y_plussoftemax = plussoftemax(x)
y_plusinversesoftemax = PIESMax(x)

print("SoftEMax:", y_softemax)
print("InverseSoftEMax:", y_inversesoftemax)
print("PlusSoftEMax:", y_plussoftemax)
print("PlusInverseSoftEMax:", y_plusinversesoftemax)
