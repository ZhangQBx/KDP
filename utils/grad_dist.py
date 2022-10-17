import torch
import torch.nn.functional as F
import pdb


def Compute_Dist(proxy_model, x, ytilde, label, temperature, cpu=False):
    T = temperature
    # print(temperature)
    if not cpu:
        proxy_model.cuda()
        x = x.cuda()
    x = x.view(1, *x.shape)
    ytilde = ytilde.view(1, *ytilde.shape)
    # ytilde /= T
    # pdb.set_trace()

    proxy_model.train()
    output = proxy_model(x)
    # output /= T
    _, K = output.shape


    # params = [p for p in proxy_model.parameters()]
    #Only for testing
    params = [p for p in proxy_model.parameters()][-2]

    log_s = F.log_softmax(output/T, dim=1)
    ytilde = F.softmax(ytilde/T, dim=1)

    jytilde = ytilde * log_s
    total_jytilde = jytilde.sum()

    total_jy = log_s[0, label]

    a = torch.autograd.grad(total_jytilde, params, create_graph=True)
    # pdb.set_trace()
    u = torch.autograd.grad(total_jy, params, retain_graph=True)

    a_norm = 0
    for gradient in a:
        a_norm += (gradient.data ** 2).sum()
    a_norm = a_norm ** (0.5)
    for gradient in a:
        gradient *= (1 / a_norm)

    u_norm = 0
    for gradient in u:
        u_norm += (gradient.data ** 2).sum()
    u_norm = u_norm ** (0.5)
    for gradient in u:
        gradient *= (1 / u_norm)

    dist = 0
    for aa, uu in zip(a, u):
        dist += ((aa - uu) ** 2).sum()

    # a_norm = 0
    # a_norm += (a.data ** 2).sum()
    # a_norm = a_norm ** (0.5)
    # a *= (1 / a_norm)
    #
    # u_norm = 0
    # u_norm += (u.data ** 2).sum()
    # u_norm = u_norm ** (0.5)
    # u *= (1 / u_norm)
    #
    # dist = ((a - u) ** 2).sum()

    return dist
