import geoopt
import torch
from utils.math_utils import sinh_div, cosh_div, sinh_div_square


EPS = {torch.float32: 1e-4, torch.float64: 1e-7}


class Lorentz(geoopt.Lorentz):
    def __init__(self, k=1.0, learnable=False):
        super(Lorentz, self).__init__(k, learnable)

    def cinner(self, x, y):
        x = x.clone()
        x.narrow(-1, 0, 1).mul_(-1)
        return x @ y.transpose(-1, -2)

    def to_poincare(self, x, dim=-1):
        dn = x.size(dim) - 1
        return x.narrow(dim, 1, dn) / (x.narrow(dim, 0, 1) + torch.sqrt(self.k))

    def from_poincare(self, x, dim=-1, eps=1e-6):
        x_norm_square = torch.sum(x * x, dim=dim, keepdim=True)
        res = (
                torch.sqrt(self.k)
                * torch.cat((1 + x_norm_square, 2 * x), dim=dim)
                / (1.0 - x_norm_square + eps)
        )
        return res

    def Frechet_mean(self, x, weights=None, keepdim=False):
        if weights is None:
            z = torch.sum(x, dim=0, keepdim=True)
        else:
            z = torch.sum(x * weights, dim=0, keepdim=keepdim)
        denorm = self.inner(None, z, keepdim=keepdim)
        denorm = denorm.abs().clamp_min(1e-8).sqrt()
        z = z / denorm
        return z

    def jacobian_expmap_v(self, x, v):
        u = self.norm(v, keepdim=True)  # (N, 1)
        v_hat = v.clone()
        v_hat.narrow(-1, 0, 1).mul_(-1)  # (N, D)
        eye = torch.eye(x.shape[1], device=x.device).unsqueeze(0)
        first_term = sinh_div(u).unsqueeze(-1) * (eye + torch.einsum("ij, ik->ijk", x, v_hat))
        second_term = (cosh_div(u) - sinh_div_square(u)).unsqueeze(-1) * torch.einsum("ij, ik->ijk", v, v_hat)
        return first_term + second_term

    def jacobian_expmap_x(self, x, v):
        u = self.norm(v, keepdim=True)  # (N, 1)
        eye = torch.eye(x.shape[1], device=x.device).unsqueeze(0)
        return torch.cosh(u).unsqueeze(-1) * eye
