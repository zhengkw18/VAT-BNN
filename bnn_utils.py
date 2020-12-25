import copy
from torch.nn import Linear

from bnn_layers import BayesLinear


def freeze(net):
    net.apply(_freeze)


def _freeze(m):
    if isinstance(m, BayesLinear):
        m.deterministic = True


def unfreeze(net):
    net.apply(_unfreeze)


def _unfreeze(m):
    if isinstance(m, BayesLinear):
        m.deterministic = False


def to_bayesian(input, psi_init_range=[-6, -5]):
    return _to_bayesian(copy.deepcopy(input), psi_init_range)


def _to_bayesian(input, psi_init_range=[-6, -5]):

    if isinstance(input, (Linear)):
        output = BayesLinear(input.in_features, input.out_features, input.bias is not None)

        setattr(output, 'weight_mu', getattr(input, 'weight'))
        setattr(output, 'bias_mu', getattr(input, 'bias'))

        output.weight_psi.data.uniform_(psi_init_range[0], psi_init_range[1])
        output.weight_psi.data = output.weight_psi.data.to(output.weight_mu.device)
        if output.bias_psi is not None:
            output.bias_psi.data.uniform_(psi_init_range[0], psi_init_range[1])
            output.bias_psi.data = output.bias_psi.data.to(output.bias_mu.device)

        return output
    else:
        for name, module in input.named_children():
            setattr(input, name, _to_bayesian(module, psi_init_range))
        return input


def to_deterministic(input):
    return _to_deterministic(copy.deepcopy(input))


def _to_deterministic(input):

    if isinstance(input, (BayesLinear)):
        output = Linear(input.in_features, input.out_features, input.bias)

        setattr(output, 'weight', getattr(input, 'weight_mu'))
        setattr(output, 'bias', getattr(input, 'bias_mu'))
        return output
    else:
        for name, module in input.named_children():
            setattr(input, name, _to_deterministic(module))
        return input
