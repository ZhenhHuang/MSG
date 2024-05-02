import torch


EPS = {torch.float32: 1e-4, torch.float64: 1e-7}


class SinhDiv(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        y_stable = torch.ones_like(x)
        y = torch.sinh(x) / x
        return torch.where(x.abs() < EPS[x.dtype], y_stable, y)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        y = (x * torch.cosh(x) - torch.sinh(x)) / x ** 2
        y_stable = torch.zeros_like(x)
        return torch.where(x.abs() < EPS[x.dtype], y_stable, y) * grad_output


sinh_div = SinhDiv.apply


class SinDiv(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        y_stable = torch.ones_like(x)
        y = torch.sin(x) / x
        return torch.where(x.abs() < EPS[x.dtype], y_stable, y)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        y = (x * torch.cos(x) - torch.sin(x)) / x ** 2
        y_stable = torch.zeros_like(x)
        return torch.where(x.abs() < EPS[x.dtype], y_stable, y) * grad_output


sin_div = SinDiv.apply