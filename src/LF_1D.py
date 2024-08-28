from FD_1D import Diffusion_FD_1D as FD1
import numpy as np
from Geometry_1D import Geometry_1D as G1D
import utils as ut
import torch
import torch.nn as nn


class LF_Layer(nn.Module):
    def __init__(self, params: dict):
        super(LF_Layer, self).__init__()
        self.params = params
        self.fd = FD1(thre=1 - 1e-5)
        self.N = params['LF_N']
        self.mid = int(self.N / 2) - 1

    def forward(self, input: torch.Tensor):
        # device = input.device
        batch_size, _ = input.shape

        # result_U_list = []
        result_k_list = []

        for i in range(batch_size):
            D = input[i]
            U = self.fd.solve(N=self.N, D=D, plot=False)
            k = ut.give_K(D, U, self.params)
            result_k_list.append(k)
            # result_U_list.append(U)

        # result_U = torch.stack(result_U_list)
        result_k = torch.stack(result_k_list)
        return result_k



