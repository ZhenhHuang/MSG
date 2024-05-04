import torch


EPS = {torch.float32: 1e-4, torch.float64: 1e-7}


cosh_bounds = {torch.float32: 85, torch.float64: 700}
sinh_bounds = {torch.float32: 85, torch.float64: 500}


def cosh(x):
    x.data.clamp_(max=cosh_bounds[x.dtype])
    return torch.cosh(x)


def sinh(x):
    x.data.clamp_(max=sinh_bounds[x.dtype])
    return torch.sinh(x)


def tanh(x):
    return x.tanh()


class SinhDiv(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        y_stable = torch.ones_like(x)
        y = sinh(x) / x
        return torch.where(x.abs() < EPS[x.dtype], y_stable, y)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        y = (x * cosh(x) - sinh(x)) / x ** 2
        y_stable = torch.zeros_like(x)
        return torch.where(x.abs() < EPS[x.dtype], y_stable, y) * grad_output


sinh_div = SinhDiv.apply


class SinhDivSquare(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        y_stable = torch.zeros_like(x)
        y = sinh(x) / x ** 2
        return torch.where(x.abs() < EPS[x.dtype], y_stable, y)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        y = (x * cosh(x) - 2 * sinh(x)) / x ** 3
        y_stable = torch.ones_like(x) / 6
        return torch.where(x.abs() < EPS[x.dtype], y_stable, y) * grad_output


sinh_div_square = SinhDivSquare.apply


class CoshDiv(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        y_stable = torch.zeros_like(x)
        y = cosh(x) / x
        return torch.where(x.abs() < EPS[x.dtype], y_stable, y)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        y = (x * sinh(x) - cosh(x)) / x ** 2
        y_stable = torch.ones_like(x) * 0.5
        return torch.where(x.abs() < EPS[x.dtype], y_stable, y) * grad_output


cosh_div = CoshDiv.apply


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