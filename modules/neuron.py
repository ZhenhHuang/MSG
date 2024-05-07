import torch
import torch.nn as nn


class IFFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, manifold, x_seq, v_seq, z_seq, v_threshold=1.0):
        ctx.manifold = manifold
        ctx.save_for_backward(x_seq, v_seq, z_seq)
        men_potential = torch.zeros_like(v_seq).to(v_seq.device)
        s_seq = []
        for t in range(x_seq.shape[0]):
            men_potential = men_potential + x_seq[t]
            spike = (men_potential > v_threshold).float()
            s_seq.append(spike)
            men_potential = men_potential - v_threshold * spike
        output = torch.stack(s_seq, dim=0)
        z_output = manifold.expmap(z_seq, v_seq)
        return output, z_output

    @staticmethod
    def backward(ctx, grad_output, grad_z_output):
        # print(f"grad_output: {grad_output.norm(p=2)}, grad_z_output: {grad_z_output.norm(p=2)}")
        x_seq, v_seq, z_seq = ctx.saved_tensors
        manifold = ctx.manifold
        jacob_v = manifold.jacobian_expmap_v(z_seq, v_seq)
        jacob_x = manifold.jacobian_expmap_x(z_seq, v_seq)
        # print(f"jacob_v: {jacob_v.norm(p=2)}, jacob_x: {jacob_x.norm(p=2)}")
        # print(f"jacob_x: {jacob_v.shape}, grad_z_output: {grad_z_output.shape}")
        grad_v = jacob_v.transpose(-1, -2) @ grad_z_output.unsqueeze(-1)
        grad_z = jacob_x.transpose(-1, -2) @ grad_z_output.unsqueeze(-1)
        # print(f"grad_v: {grad_v.norm(p=2)}, grad_z: {grad_z.norm(p=2)}")
        return None, None, grad_v.squeeze(), grad_z.squeeze(), None


class RiemannianIFNode(nn.Module):
    def __init__(self, manifold, v_threshold: float = 1.):
        super(RiemannianIFNode, self).__init__()
        self.manifold = manifold
        self.v_threshold = v_threshold

    def forward(self, x_seq: torch.Tensor, v_seq: torch.Tensor, z_seq: torch.Tensor):
        """

        :param x_seq: [T, N, D]
        :param v_seq: [N, D]
        :param z_seq: [N, D]
        :return:
        """
        out_seq, z_out_seq = IFFunction.apply(self.manifold, x_seq, v_seq, z_seq, self.v_threshold)
        return out_seq, z_out_seq








