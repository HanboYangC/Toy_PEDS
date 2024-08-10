import torch
from FD_1D import Diffusion_FD_1D as FD1
import numpy as np
from Geometry_1D import Geometry_1D as G1D

class LF_Layer (torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor):
        # 保存输入，以便在backward中使用
        ctx.save_for_backward(input)
        device = input.device
        input_np = input.detach().cpu().numpy().reshape(-1)
        fd = FD1(thre=1 - 1e-5)
        N = len(input_np)
        U = fd.solve(N=N, D=input_np, plot=False)
        result_tensor = torch.from_numpy(U).to(device)
        return result_tensor
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        # 计算这个操作的梯度
        grad_input = 2 * input * grad_output  # 根据链式法则计算梯度
        return grad_input



