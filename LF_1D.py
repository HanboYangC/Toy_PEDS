import torch
from FD_1D import Diffusion_FD_1D as FD1
import numpy as np
from Geometry_1D import Geometry_1D as G1D

class LF_Layer (torch.autograd.Function):
    @staticmethod
    def forward(ctx,input: torch.Tensor,params:dict):
        device = input.device
        input_np = input.detach().cpu().numpy().reshape(-1)
        fd = FD1(thre=1 - 1e-5)
        N = len(input_np)
        U = fd.solve(N=N, D=input_np, plot=False)
        U = torch.from_numpy(U).to(device)

        ctx.save_for_backward(input)
        ctx.result_tensor = U
        ctx.lengths = params
        return U

    @staticmethod
    def backward(ctx, grad_output=None):
        input, = ctx.saved_tensors
        params = ctx.lengths
        U = ctx.result_tensor
        h = params['h']
        D = input.detach()  # Use only torch operations
        D_out = torch.tensor([1.0], device=input.device,
                             dtype=torch.float32)  # Ensuring tensor is on the same device and type
        N = params['N']
        gu = torch.zeros((N, N), device=input.device, dtype=torch.float32)
        for i in range(N):
            D_l = D_out if i - 1 < 0 else D[i - 1]
            D_r = D_out if i + 1 > N - 1 else D[i + 1]
            gu[i, i] = -D_l - D_r
            if i + 2 < N:
                gu[i, i + 2] = D_r
            if i - 2 >= 0:
                gu[i, i - 2] = D_l
        gu=(1/(4*h*h))*gu
        ul = torch.tensor([0], device=input.device, dtype=torch.float32)
        ur = torch.tensor([1], device=input.device, dtype=torch.float32)
        U_long = torch.cat([ul, ul, U, ur, ur]).view(1, -1).to(dtype=torch.float32)  # Convert to float32
        nabla = torch.zeros((N + 4, N + 2), device=input.device, dtype=torch.float32)
        for i in range(N + 2):
            nabla[i, i] = -1
            nabla[i + 2, i] = 1
        dU = (1/(2*h))*torch.matmul(U_long, nabla)[0, 1:-1]  # Ensure all tensors are in float32 before multiplication
        gd=torch.zeros((N,N))
        for i in range(N):
            if i+1 < N:
                gd[i, i+1] = -dU[i]
            if i-1>=0:
                gd[i,i-1]=dU[i]
        gd=(1/(2*h))*gd

        gu_inv = torch.linalg.inv(gu)
        grad_output_vector = grad_output.view(-1, 1).to(dtype=torch.float32)  # Convert grad_output to float32
        lam = -torch.matmul(gu_inv, grad_output_vector)
        df=torch.matmul(gd.t(),lam)
        return df


