import torch
from torch.nn import Module, Parameter
import torch.nn.functional as F


class MulExpAddFunction(torch.autograd.Function):
    # manually define function to save memory
    @staticmethod
    def forward(ctx, input, psi, mu):
        ctx.mark_dirty(input)
        output = input.mul_(psi.exp()).add_(mu)
        ctx.save_for_backward(mu, output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        mu, output = ctx.saved_tensors
        grad_psi = (grad_output * (output - mu)).sum(0)
        grad_mu = grad_output.sum(0)
        return None, grad_psi, grad_mu


class BayesLinear(Module):
    __constants__ = ['bias', 'in_features', 'out_features']

    def __init__(self, in_features, out_features, bias=True):
        super(BayesLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.deterministic = False

        self.weight_mu = Parameter(torch.Tensor(out_features, in_features))
        self.weight_psi = Parameter(torch.Tensor(out_features, in_features))

        self.bias = bias
        if bias:
            self.bias_mu = Parameter(torch.Tensor(out_features))
            self.bias_psi = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_psi', None)

        self.weight_size = list(self.weight_mu.shape)
        self.bias_size = list(self.bias_mu.shape) if self.bias else None
        self.mul_exp_add = MulExpAddFunction.apply

    def forward(self, input):
        if self.deterministic:
            out = F.linear(input, self.weight_mu, self.bias_mu)
        else:
            weight = self.mul_exp_add(torch.randn(input.shape[0], *self.weight_size, device=input.device, dtype=input.dtype), self.weight_psi, self.weight_mu)
            out = torch.bmm(weight, input.unsqueeze(2)).squeeze()
            if self.bias:
                bias = self.mul_exp_add(torch.randn(input.shape[0], *self.bias_size, device=input.device, dtype=input.dtype), self.bias_psi, self.bias_mu)
                out = out + bias
        return out
