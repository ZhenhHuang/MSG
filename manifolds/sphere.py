import geoopt
import torch
from utils.math_utils import sin_div, cos_div_square, sin_div_cube


class Sphere(geoopt.Sphere):
    def __init__(self):
        super(Sphere, self).__init__()

    def jacobian_expmap_v(self, x, v):
        v_norm = torch.norm(v, keepdim=True, dim=-1)  # (N, 1)
        eye = torch.eye(x.shape[1], device=x.device).unsqueeze(0)
        first_term = sin_div(v_norm).unsqueeze(-1) * (eye - torch.einsum("ij, ik->ijk", x, v))
        second_term = (cos_div_square(v_norm) - sin_div_cube(v_norm)).unsqueeze(-1) * torch.einsum("ij, ik->ijk", v)
        return first_term + second_term

    def jacobian_expmap_x(self, x, v):
        u = torch.sum(v ** 2, dim=-1, keepdim=True)  # (N, 1)
        eye = torch.eye(x.shape[1], device=x.device).unsqueeze(0)
        return torch.cos(u) * eye
