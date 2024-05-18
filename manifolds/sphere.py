from typing import Union, Tuple, Optional

import geoopt
import torch
from utils.math_utils import sin_div, cos_div_square, sin_div_cube
from geoopt.manifolds.stereographic.math import geodesic


EPS = {torch.float32: 1e-4, torch.float64: 1e-7}


class Sphere(geoopt.Sphere):
    def __init__(self):
        super(Sphere, self).__init__()
        self.k = torch.nn.Parameter(torch.tensor([1.0]), requires_grad=False)

    def origin(
        self,
        *size: Union[int, Tuple[int]],
        dtype=None,
        device=None,
        seed: Optional[int] = 42
    ) -> torch.Tensor:
        pole = torch.zeros(*size, dtype=dtype).to(device)
        pole.narrow(-1, 0, 1).add_(-1)
        return pole

    def geodesic(self, t, x, y):
        return geodesic(t, x, y, k=torch.tensor([1.]).to(x.device))

    def expmap0(self, u: torch.Tensor, dim=-1):
        """Choose South Pole"""
        pole = torch.zeros_like(u)
        pole.narrow(dim, 0, 1).add_(-1)
        return self.expmap(pole, u)

    def expmap(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        norm_u = u.norm(dim=-1, keepdim=True)
        exp = x * torch.cos(norm_u) + u * sin_div(norm_u)
        retr = self.projx(x + u)
        cond = norm_u > EPS[norm_u.dtype]
        return torch.where(cond, exp, retr)

    def logmap0(self, y: torch.Tensor):
        x = self.origin(y.shape, dtype=y.dtype, device=y.device)
        u = self.proju(x, y - x)
        dist = self.dist(x, y, keepdim=True)
        cond = dist.gt(EPS[x.dtype])
        result = torch.where(
            cond, u * dist / u.norm(dim=-1, keepdim=True).clamp_min(EPS[x.dtype]), u
        )
        return result

    def proju0(self, u: torch.Tensor) -> torch.Tensor:
        x = self.origin(u.shape, dtype=u.dtype, device=u.device)
        u = u - (x * u).sum(dim=-1, keepdim=True) * x
        return self._project_on_subspace(u)

    def norm(self, u: torch.Tensor, x: torch.Tensor = None, *, keepdim=False) -> torch.Tensor:
        return torch.norm(u, dim=-1, keepdim=keepdim)

    def random_normal(
            self, *size, mean=0, std=1, dtype=None, device=None
    ):
        tens = (
                torch.randn(*size, device=device, dtype=dtype)
                * std
                + mean
        )
        return geoopt.ManifoldTensor(self.expmap0(tens), manifold=self)


    def jacobian_expmap_v(self, x, v):
        v_norm = torch.norm(v, keepdim=True, dim=-1)  # (N, 1)
        # print(f"v_norm: {v_norm.min(), v_norm.max()}")
        eye = torch.eye(x.shape[1], device=x.device).unsqueeze(0)
        first_term = sin_div(v_norm).unsqueeze(-1) * (eye - x.unsqueeze(-1) @ v.unsqueeze(1))
        second_term = (cos_div_square(v_norm) - sin_div_cube(v_norm)).unsqueeze(-1) * v.unsqueeze(-1) @ v.unsqueeze(1)
        # print(f"first_term: {first_term.max()}, second_term: {second_term.max()}")
        return first_term + second_term

    def jacobian_expmap_x(self, x, v):
        u = torch.sum(v ** 2, dim=-1, keepdim=True).clamp(min=1e-4).sqrt()  # (N, 1)
        eye = torch.eye(x.shape[1], device=x.device).unsqueeze(0)
        return torch.cos(u).unsqueeze(-1) * eye


if __name__ == '__main__':
    from torch.autograd.functional import jacobian
    def func(x, v):
        m = Sphere()
        return m.expmap(x, v)
    x, v = torch.tensor([[1., 0., 0.]]), torch.tensor([[0., 0., 2.]])
    j = jacobian(func, (x, v))[1]
    jv = Sphere().jacobian_expmap_v(x, v)
    print(j, j.shape)
    print(jv, jv.shape)