# nemo_forced_aligner/utils/viterbi_decoding.py

import torch
import numpy as np
from typing import List
from numba import cuda, njit
import math


def viterbi_decoding(
        log_probs_batch: List[torch.Tensor],
        y_batch: List[List[int]],
        T_batch: List[int],
        U_batch: List[int],
        device: torch.device
) -> List[List[int]]:
    """
    Perform Viterbi decoding for forced alignment

    Args:
        log_probs_batch: List of log probability tensors
        y_batch: List of target token sequences
        T_batch: List of time steps
        U_batch: List of target sequence lengths
        device: Device to use for computation

    Returns:
        List of alignment paths
    """
    alignments = []

    for log_probs, y, T, U in zip(log_probs_batch, y_batch, T_batch, U_batch):
        # Move to specified device
        log_probs = log_probs.to(device)

        # Perform Viterbi alignment
        alignment = viterbi_align_torch(log_probs, y, device)
        alignments.append(alignment)

    return alignments


def viterbi_align_torch(log_probs: torch.Tensor, y: List[int], device: torch.device) -> List[int]:
    """
    Torch-based Viterbi alignment implementation

    Args:
        log_probs: Log probability tensor (T, V) where T is time steps, V is vocabulary size
        y: Target token sequence
        device: Device for computation

    Returns:
        Alignment path
    """
    T = log_probs.shape[0]
    U = len(y)
    V = log_probs.shape[1]

    # Initialize DP table
    alpha = torch.full((T, U), float('-inf')).to(device)
    path = torch.zeros((T, U), dtype=torch.long).to(device)

    # Handle blank token (assumed to be at index 0)
    blank_idx = 0

    # Initialize first column (blank tokens)
    alpha[0, 0] = log_probs[0, blank_idx]

    # Initialize first row
    for u in range(1, U):
        if u < T:
            alpha[u, u] = alpha[u - 1, u - 1] + log_probs[u, y[u - 1]]
            path[u, u] = u - 1

    # Fill DP table
    for t in range(1, T):
        for u in range(U):
            # Transition from blank
            if u < U:
                blank_score = alpha[t - 1, u] + log_probs[t, blank_idx]
                alpha[t, u] = blank_score
                path[t, u] = u

            # Transition from previous token
            if u > 0 and u - 1 < U:
                token_score = alpha[t - 1, u - 1] + log_probs[t, y[u]]
                if token_score > alpha[t, u]:
                    alpha[t, u] = token_score
                    path[t, u] = u - 1

    # Backtrack to find alignment
    alignment = []
    t = T - 1
    u = U - 1

    while t >= 0:
        alignment.append(u)

        if t > 0:
            u = path[t, u].item()

        t -= 1

    alignment.reverse()
    return alignment


def gpu_viterbi_align(log_probs: torch.Tensor, y: List[int]) -> List[int]:
    """
    GPU-accelerated Viterbi alignment using CUDA kernels

    Note: This is a simplified version that would need proper CUDA kernel implementation
    """
    device = log_probs.device
    if device.type == 'cuda':
        # Use optimized CUDA implementation if available
        try:
            import numba.cuda as cuda_numba
            # Would implement CUDA kernel here
            # For now, fall back to torch implementation
            return viterbi_align_torch(log_probs, y, device)
        except ImportError:
            return viterbi_align_torch(log_probs, y, device)
    else:
        return viterbi_align_torch(log_probs, y, device)


def batch_viterbi_decoding(
        log_probs_batch: List[torch.Tensor],
        y_batch: List[List[int]],
        device: torch.device,
        use_cuda_kernels: bool = False
) -> List[List[int]]:
    """
    Perform batch Viterbi decoding with optional CUDA acceleration
    """
    alignments = []

    for log_probs, y in zip(log_probs_batch, y_batch):
        log_probs = log_probs.to(device)

        if use_cuda_kernels and device.type == 'cuda':
            alignment = gpu_viterbi_align(log_probs, y)
        else:
            alignment = viterbi_align_torch(log_probs, y, device)

        alignments.append(alignment)

    return alignments