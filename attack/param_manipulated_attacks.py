import functools
from typing import List

from flex.common.utils import check_min_arguments
from flex.model import FlexModel
from copy import deepcopy
import torch

def manipulated_weights_attack(func):
    min_args = 1
    assert check_min_arguments(
        func, min_args),f"The decorated function: {func.__name__} is expected to have at least {min_args} argument/s."

    @functools.wraps(func)
    def _manipulate_weights_(aggregator_flex_model: FlexModel, _, *args, **kwargs):
        aggregator_flex_model["weights"] = func(
            aggregator_flex_model["weights"], *args, **kwargs
        )
        #aggregator_flex_model["weights"] = [] no va

    return _manipulate_weights_

def adecuate_params(list_param_client):
    list_param_clients = deepcopy(list_param_client)

    param_list = [torch.cat([param.reshape((-1, 1)) for param in params], dim=0) for params in list_param_client]

    return param_list

def adecuate_params_reverse(list_param_cat, original_params_shape):
    list_param_cat = deepcopy(list_param_cat)
    reverse=[]
    
    for param_cat in list_param_cat:
        idx=0
        params_for_this_client=[]
        for param_ori_client in original_params_shape:
            modif_param = param_cat[idx: (idx + torch.numel(param_ori_client))]
            params_for_this_client.append(modif_param.reshape(tuple(param_ori_client.size())))
            idx += torch.numel(param_ori_client)
        reverse.append(params_for_this_client)

    return reverse


#Primitives

def scaling_attack_scale(v, f, device):
    """
    Second part of the scaling attack which scales the gradients of the malicious clients to increase their impact.
    The attack is based on the description in https://arxiv.org/abs/2012.13995
    v: list of gradients
    net: model
    lr: learning rate
    f: number of malicious clients, where the first f are malicious
    device: device used in training and inference
    """
    scaling_factor = len(v)
    for i in range(f):
        v[i] = v[i] * scaling_factor
    return v

def trim_attack(v, f, device):
    """
    Local model poisoning attack against the trimmed mean aggregation rule.
    Based on the description in https://arxiv.org/abs/1911.11815
    v: list of gradients
    net: model
    lr: learning rate
    f: number of malicious clients, where the first f are malicious
    device: device used in training and inference
    """
    vi_shape = tuple(v[0].size())
    v_tran = torch.cat(v, dim=1)
    maximum_dim, _ = torch.max(v_tran, dim=1, keepdim=True)
    minimum_dim, _ = torch.min(v_tran, dim=1, keepdim=True)
    direction = torch.sign(torch.sum(torch.cat(v, dim=1), dim=-1, keepdim=True))
    directed_dim = (direction > 0) * minimum_dim + (direction < 0) * maximum_dim
    # let the malicious clients (first f clients) perform the attack
    for i in range(f):
        random_12 = (1. + torch.rand(*vi_shape)).to(device)
        v[i] = directed_dim * ((direction * directed_dim > 0) / random_12 + (direction * directed_dim < 0) * random_12)
    return v

def krum_attack(v, f, device): #aqui quite el learning rate, el atacante no conoce eso
    """
    Local model poisoning attack against the krum aggregation rule.
    Based on the description in https://arxiv.org/abs/1911.11815
    v: list of gradients
    net: model
    lr: learning rate
    f: number of malicious clients, where the first f are malicious
    device: device used in training and inference
    """
    threshold = 1e-5

    n = len(v)
    d = v[0].size()[0]
    dist = torch.zeros((n, n)).to(device)
    for i in range(n):  # compute euclidean distance of benign to benign devices
        for j in range(i + 1, n):
            d = torch.norm(v[i] - v[j], p=2)
            dist[i, j], dist[j, i] = d, d

    dist_benign_sorted, _ = torch.sort(dist[f:, f:])
    min_dist = torch.min(torch.sum(dist_benign_sorted[:, 0:(n - f - 1)], dim=-1))

    dist_w_re = []
    for i in range(f, n):
        dist_w_re.append(torch.norm(v[i], p=2))
    max_dist_w_re = torch.max(torch.stack(dist_w_re))


    max_lambda = min_dist / (((n - 2) * (f - 1)) * torch.sqrt(d)) + max_dist_w_re / torch.sqrt(d)#Ver esta línea porque en la múlti`licación de n y f estaba sin los paréntesis de por medio

    actual_lambda = max_lambda
    sorted_dist, _ = torch.sort(dist, dim=-1)
    update_before = v[torch.argmin(torch.sum(sorted_dist[:, 0:(n - f - 1)], dim=-1))]
    while actual_lambda > threshold:
        #print("Act lamb", actual_lambda)
        for i in range(f):
            v[i] = - actual_lambda * torch.sign(update_before)

        dist = torch.zeros((n, n)).to(device)
        for i in range(n):
            for j in range(i + 1, n):
                d = torch.norm(v[i] - v[j])
                dist[i, j], dist[j, i] = d, d
        sorted_dist, _ = torch.sort(dist, dim=-1)
        global_update = v[torch.argmin(torch.sum(sorted_dist[:, 0:(n - f - 1)], dim=-1))]
        if torch.equal(global_update, v[0]):
            break
        else:
            actual_lambda = actual_lambda / 2

    return v

def min_max_attack(v, f, device):
    """
    Local model poisoning attack from https://par.nsf.gov/servlets/purl/10286354
    The implementation is based of their repository (https://github.com/vrt1shjwlkr/NDSS21-Model-Poisoning)
    but refactored for clarity.
    v: list of gradients
    net: model
    lr: learning rate
    f: number of malicious clients, where the first f are malicious
    device: device used in training and inference
    """
    catv = torch.cat(v, dim=1)
    grad_mean = torch.mean(catv, dim=1)
    deviation = grad_mean / torch.norm(grad_mean, p=2)  # decided to use unit_vec distance which was their default
    # but they also had the option to use sign and standard deviation
    gamma = torch.Tensor([50.0]).float().to(device)
    threshold_diff = 1e-5
    gamma_fail = gamma
    gamma_succ = 0

    distances = []
    for update in v:
        distance = torch.norm(catv - update, dim=1, p=2) ** 2
        distances = distance[None, :] if not len(distances) else torch.cat((distances, distance[None, :]), 0)

    max_distance = torch.max(distances)  # determine max distance left side of optimization
    del distances

    # finding optimal gamma according to algorithm 1
    while torch.abs(gamma_succ - gamma) > threshold_diff:
        #print(gamma)
        #print("Grad_mean", grad_mean, "Gamma", gamma, "Desviation", deviation, "Grad_mean menos ganma", (grad_mean - gamma))
        mal_update = (grad_mean - gamma * deviation)# La adecuación está en esta línea, saber identidicar cuando es la diferencia apropiada
        distance = torch.norm(catv - mal_update[:, None], dim=1, p=2) ** 2 #Estoy cambiando este valor pero hay que revisar esto, al parecer la distancia influye mucho en este cálculo
        max_d = torch.max(distance)

        if max_d <= max_distance:
            gamma_succ = gamma
            gamma = gamma + gamma_fail / 2
            #print(gamma)
        else:
            gamma = gamma - gamma_fail / 2

        gamma_fail = gamma_fail / 2
    mal_update = (grad_mean - gamma_succ * deviation)#Adecuación

    for i in range(f):
        v[i] = mal_update[:, None]

    return v

def min_sum_attack(v, f, device):
    """
    Local model poisoning attack from https://par.nsf.gov/servlets/purl/10286354
    The implementation is based of their repository (https://github.com/vrt1shjwlkr/NDSS21-Model-Poisoning)
    but refactored for clarity.
    v: list of gradients
    net: model
    lr: learning rate
    f: number of malicious clients, where the first f are malicious
    device: device used in training and inference
    """
    catv = torch.cat(v, dim=1)
    grad_mean = torch.mean(catv, dim=1)
    deviation = grad_mean / torch.norm(grad_mean, p=2)  # decided to use unit_vec distance which was their default
    # but they also had the option to use sign and standard deviation
    gamma = torch.Tensor([0.01]).float().to(device)
    threshold_diff = 1e-5
    gamma_fail = gamma
    gamma_succ = 0

    distances = []
    for update in v:
        distance = torch.norm(catv - update, dim=1, p=2) ** 2
        distances = distance[None, :] if not len(distances) else torch.cat((distances, distance[None, :]), 0)

    scores = torch.sum(distances, dim=1)
    min_score = torch.min(scores)
    del distances

    # finding optimal gamma according to algorithm 1
    while torch.abs(gamma_succ - gamma) > threshold_diff:
        mal_update = (grad_mean - gamma * deviation)
        distance = torch.norm(catv - mal_update[:, None], dim=1, p=2) ** 2
        score = torch.sum(distance)

        if score <= min_score:
            gamma_succ = gamma
            gamma = gamma + gamma_fail / 2
        else:
            gamma = gamma - gamma_fail / 2

        gamma_fail = gamma_fail / 2
    mal_update = (grad_mean - gamma_succ * deviation)#Creo que si aquí el gamma es más pegado a 0 entonces más impacto va a tener

    for i in range(f):
        v[i] = mal_update[:, None]

    return v