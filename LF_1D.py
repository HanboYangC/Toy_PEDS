import torch
from FD_1D import Diffusion_FD_1D as FD1
import numpy as np
from Geometry_1D import Geometry_1D as G1D
import utils as ut

class LF_Layer (torch.autograd.Function):
    @staticmethod
    def forward(ctx,input: torch.Tensor,params:dict):
        ctx.save_for_backward(input)
        device = input.device
        input_np = input.detach().cpu().numpy().reshape((-1,params['LF_N']))
        fd = FD1(thre=1 - 1e-5)
        N = params['LF_N']
        mid=int(N/2)-1
        result_U=[]
        result_k=[]
        for D in input_np:
            U = fd.solve(N=N, D=D, plot=False)
            result_U.append(U)
            k=ut.give_K(D,U,params)
            result_k.append(k)

        U = torch.tensor(result_U,dtype=torch.float32).to(device)
        k=torch.tensor(result_k,dtype=torch.float32).to(device)
        ctx.result_tensor = U
        ctx.result_k=k
        ctx.lengths = params
        ctx.mid=mid
        return k

    @staticmethod
    def backward(ctx, grad_output=None):
        input, = ctx.saved_tensors
        params = ctx.lengths
        mid=ctx.mid

        batch_size = input.size(0)
        U = ctx.result_tensor
        h = params['grid_width']
        D = input.detach()  # Use only torch operations
        D_out = torch.tensor([1.0], device=input.device,
                             dtype=torch.float32)  # Ensuring tensor is on the same device and type
        N = params['LF_N']
        df_list=[]

        for i in range(len(D)):
            d=D[i]

            ku=torch.zeros((N,1))
            ku[mid-1,0]= -d[mid-1] / (4 * h)
            ku[mid ,0]=- d[mid]/ (4 * h)
            ku[mid+1, 0]= d[mid+1]/(4*h)
            ku[mid+2, 0]= d[mid+2]/(4*h)
            # A,b=ut.get_Ab(d)
            # inv_AAT = torch.inverse(torch.matmul(A, A.T))
            # kup=ku-torch.matmul(torch.matmul(A.T,inv_AAT),torch.matmul(A,ku))
            # norm_kup = torch.norm(kup, p=2)
            # if norm_kup > 0:
            #     kup = kup / norm_kup

            gu = torch.zeros((N, N), device=input.device, dtype=torch.float32)
            for j in range(N):
                d_l = D_out if j - 1 < 0 else d[j - 1]
                d_r = D_out if j + 1 > N - 1 else d[j + 1]
                gu[j, j] = -d_l - d_r
                if j + 2 < N:
                    gu[j, j + 2] = d_r
                if j - 2 >= 0:
                    gu[j, j - 2] = d_l

            u=U[i]
            ul = torch.tensor([0], device=input.device, dtype=torch.float32)
            ur = torch.tensor([1], device=input.device, dtype=torch.float32)
            u_long = torch.cat([ul, ul, u, ur, ur]).view(1, -1).to(dtype=torch.float32)  # Convert to float32

            nabla = torch.zeros((N + 2, N + 4), device=input.device, dtype=torch.float32)
            for j in range(N + 2):
                nabla[j, j] = -1
                nabla[j, j+2] = 1
            du = torch.matmul(nabla, u_long.T)[1:-1,0].view(-1)
            # du = (1 / (2 * h)) * torch.matmul(nabla, u_long.T)[1:-1,0].view(-1)# Ensure all tensors are in float32 before multiplication
            Ad=torch.zeros((N,N))
            for k in range(N):
                if k+1 < N:
                    Ad[k, k+1] = du[k + 1]
                if k-1>=0:
                    Ad[k,k-1]= -du[k - 1]
            bd=torch.zeros((N,N))
            bd[-2,-1]=-1
            gd=Ad-bd
            # gd=(1/(2*h))*gd
            det = torch.det(gu)

            epsilon=1e-4
            if torch.abs(det) < epsilon:
                # print("Matrix is near singular or singular, using pseudo inverse.")
                gu_inv = torch.linalg.pinv(gu)
            else:
                gu_inv = torch.linalg.inv(gu)
            ud=-torch.matmul(gu_inv,gd)
            df=grad_output[i] *torch.matmul(ud.T,ku)
            df_list.append(df)

        df_avg = torch.mean(torch.stack(df_list), dim=0)
        scale_factor = 1
        df_avg = df_avg * scale_factor
        df_avg_expanded = df_avg.view(1, -1).expand(batch_size, -1)


        return df_avg_expanded, None



