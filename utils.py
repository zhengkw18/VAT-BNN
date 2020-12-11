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
