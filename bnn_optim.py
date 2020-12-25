import torch


class PsiSGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0, weight_decay=0, nesterov=False, num_data=50000):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)
        super(PsiSGD, self).__init__(params, defaults)
        self.num_data = num_data

    @torch.no_grad()
    def step(self):
        loss = None

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad
                # kl divergence in ELBO
                if weight_decay != 0:
                    d_p = d_p.add((p * 2).exp(), alpha=weight_decay)
                d_p = d_p.sub(1. / self.num_data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p)
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf

                p.add_(d_p, alpha=-group['lr'])

        return loss
