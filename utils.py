from FD_1D import Diffusion_FD_1D as FD1
from Geometry_1D import Geometry_1D as G1D
import torch
import numpy as np

def label_D(D):
    hf=FD1()
    return hf.solve(D)

def label_lengths(lengths,params):
    N=params['HF_N']
    if N<100:
        print('Warning: The N is too small for high fidelity solver')
    geo=G1D(lengths,params)
    D=geo.get_D(N)
    hf=FD1()
    y=hf.solve(D=D,plot=False)
    return y

def give_K(D,U,params):
    if not isinstance(D, type(U)):
        print('Error: D and U are not of the same type')
        return
    N=len(D)
    if N==params['LF_N']:
        pass
    elif N==params['HF_N']:
        pass
    else:
        print('Error:Wrong N')
        return
    if N%2!=0:
        print('Warning: The N is not even, which may cause unexpected problems')
    mid=int(N/2)-1
    D_mid=(D[mid-1]+D[mid+1])/2
    # Ul = U[mid]
    Ul=U[mid-1]
    Ur=U[mid+1]
    h=params['width']/N
    # if isinstance(D, np.ndarray):
    #     temp = np.zeros_like(D)
    # elif isinstance(D, torch.Tensor):
    #     temp = torch.zeros_like(D)
    k=(D_mid/(h))*(Ur-Ul)
    return k


def get_Ab(D, D_out=1):

    if isinstance(D, np.ndarray):
        is_numpy = True
        N = len(D)
        A = np.zeros((N, N))
        b = np.zeros((N, 1))
    elif isinstance(D, torch.Tensor):
        is_numpy = False
        N = len(D)
        A = torch.zeros((N, N))
        b = torch.zeros((N, 1))
    else:
        raise TypeError("Input must be a numpy array or a torch tensor")

    for i in range(N):
        if i == 0:
            A[i, i] = -D_out - D[i + 1]
            if i + 2 < N:
                A[i, i + 2] = D[i + 1]
        elif i == N - 1:
            A[i, i] = -D_out - D[i - 1]
            if i - 2 >= 0:
                A[i, i - 2] = D[i - 1]
        else:
            A[i, i] = -D[i + 1] - D[i - 1]
            if i - 2 >= 0:
                A[i, i - 2] = D[i - 1]
            if i + 2 < N:
                A[i, i + 2] = D[i + 1]

        b[-1, 0] = -D_out
        b[-2, 0] = -D[N - 1]
    return A,b


def downsample(arr, target_length):
    """
    Downsamples a 1D array by averaging values to achieve the specified target length.

    Parameters:
        arr (np.ndarray): Input 1D array to be downsampled.
        target_length (int): Desired length of the downsampled array.

    Returns:
        np.ndarray: Downsampled 1D array with averaged values.
    """
    if target_length <= 0:
        raise ValueError("Target length must be a positive integer.")
    if target_length > len(arr):
        raise ValueError("Target length must be less than or equal to the length of the input array.")

    # Calculate the length of each segment
    segment_length = len(arr) / target_length

    # Initialize an empty list to hold the downsampled values
    downsampled_arr = []

    for i in range(target_length):
        # Calculate the start and end indices of the current segment
        start_index = int(i * segment_length)
        end_index = int((i + 1) * segment_length)

        # Slice the array and compute the average of the current segment
        segment_avg = np.mean(arr[start_index:end_index])
        downsampled_arr.append(segment_avg)

    return np.array(downsampled_arr)

# def find_smaller_region(all_loss)