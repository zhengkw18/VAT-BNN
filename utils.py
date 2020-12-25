import torch
import torch.nn.functional as F
import contextlib


@contextlib.contextmanager
def _disable_tracking_bn_stats(model):

    def switch_attr(m):
        if hasattr(m, 'track_running_stats'):
            m.track_running_stats ^= True

    model.apply(switch_attr)
    yield
    model.apply(switch_attr)


def get_normalized_vector(d):
    d /= (1e-12 + torch.amax(torch.abs(d), tuple(range(1, d.dim())), keepdim=True))
    d /= torch.sqrt(1e-6 + torch.sum(torch.pow(d, 2.0), tuple(range(1, d.dim())), keepdim=True))
    return d


def generate_virtual_adversarial_perturbation(x, logit, model, epsilon):
    d = torch.randn_like(x)
    d = 1e-6 * get_normalized_vector(d)
    logit_p = logit
    with _disable_tracking_bn_stats(model):
        d.requires_grad_(True)
        logit_m = model(x + d)
        dist = kl_divergence_with_logit(logit_p, logit_m)
        grad = torch.autograd.grad(dist, [d])[0]
    r_vadv = epsilon * get_normalized_vector(grad)
    return r_vadv.detach()


def virtual_adversarial_loss(x, logit, model, epsilon):
    r_vadv = generate_virtual_adversarial_perturbation(x, logit, model, epsilon)
    with _disable_tracking_bn_stats(model):
        logit_p = logit.detach()
        logit_m = model(x + r_vadv)
        loss = kl_divergence_with_logit(logit_p, logit_m)
    if torch.isnan(loss):
        return 0
    return loss


def generate_adversarial_perturbation(x, target, model, epsilon):
    with _disable_tracking_bn_stats(model):
        x.requires_grad_(True)
        logit = model(x)
        loss = F.cross_entropy(logit, target)
        grad = torch.autograd.grad(loss, [x])[0]
    r_vadv = epsilon * get_normalized_vector(grad)
    return r_vadv.detach()


def entropy(logit):
    p = F.softmax(logit, dim=-1)
    logp = logsoftmax(logit)
    return -1 * torch.sum(p * logp, dim=-1)


def mutual_information(p_logit, q_logit):
    p = F.softmax(p_logit, dim=-1)
    q = F.softmax(q_logit, dim=-1)
    p_mean = (p + q) / 2.0
    ent_p_mean = entropy(torch.log(p_mean))
    # softmax_var = ((p - p_mean)**2 + (q - p_mean)**2) / 2
    # print(softmax_var.mean())
    entp = entropy(p_logit)
    entq = entropy(q_logit)
    return torch.mean(ent_p_mean - (entp + entq) / 2.0, dim=0)


def generate_mi_adv_target(model, input, epsilon):
    d = torch.zeros_like(input)
    d.requires_grad_(True)
    with _disable_tracking_bn_stats(model):
        p_logit = model(input + d)
        q_logit = model(input + d)
        mi = mutual_information(p_logit, q_logit)
        grad = torch.autograd.grad(mi, [d])[0]
    r_vadv = epsilon * get_normalized_vector(grad)
    return r_vadv.detach()


def mi_adversarial_loss(model, input, logit, epsilon, adv_target=False):
    kl_loss, mi_loss = 0.0, 0.0
    if adv_target:
        r_adv = generate_mi_adv_target(model, input, epsilon)
    else:
        r_adv = torch.zeros_like(input)
    with _disable_tracking_bn_stats(model):
        logit_q = model(input + r_adv)
        if adv_target:
            logit_p = model(input + r_adv)
            mi_loss = mutual_information(logit_p, logit_q)
            logit = logit.detach()
            kl_loss = (kl_divergence_with_logit(logit, logit_p) + kl_divergence_with_logit(logit, logit_q)) / 2
        else:
            mi_loss = mutual_information(logit, logit_q)
            kl_loss = torch.zeros_like(mi_loss)
    return mi_loss, kl_loss


def accuracy(logit, y):
    pred = torch.argmax(logit, dim=1)
    return torch.mean((pred == y).float())


def logsoftmax(x):
    xdev = x - torch.amax(x, dim=1, keepdim=True)
    lsm = xdev - torch.log(torch.sum(torch.exp(xdev), dim=1, keepdim=True))
    return lsm


def kl_divergence_with_logit(q_logit, p_logit):
    q = F.softmax(q_logit, dim=1)
    qlogq = torch.mean(torch.sum(q * logsoftmax(q_logit), dim=1))
    qlogp = torch.mean(torch.sum(q * logsoftmax(p_logit), dim=1))
    return qlogq - qlogp
