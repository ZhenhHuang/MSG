import geoopt
import torch
import geoopt.manifolds.lorentz.math as lmath
from utils.math_utils import sin_div


class Sphere(geoopt.Sphere):
    def __init__(self):
        super(Sphere, self).__init__()

    def jacobian_expmap_v(self, x, v):
        pass

    def jacobian_expmap_x(self, x, v):
        u = torch.sum(v ** 2, dim=-1, keepdim=True)  # (N, 1)
        eye = torch.eye(x.shape[1], device=x.device).unsqueeze(0)
        return torch.cos(u) * eye
