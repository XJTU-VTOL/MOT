import torch
from numba import cuda, int64
import math

@cuda.jit()
def find_id_kernel(all_id, cur_id, match):
    x = cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x
    query_len = len(cur_id)
    value_len = len(all_id)
    if x >= query_len:
        return

    id = cur_id[x]
    cur_index = (value_len - 1) // 2
    seg_len = value_len // 2
    while id != all_id[cur_index]:
        if id > all_id[cur_index]:
            cur_index += (seg_len // 2 + 1)
        else:
            cur_index -= (seg_len // 2 + 1)

        if cur_index < 0:
            cur_index = 0
        elif cur_index > value_len - 1:
            cur_index = value_len - 1

        seg_len = seg_len // 2

    cuda.syncthreads()
    match[x] = cur_index


def find_id(all_sorted_id: torch.Tensor, cur_id):
    if not all_sorted_id.is_contiguous():
        all_sorted_id = all_sorted_id.contiguous()

    if not cur_id.is_contiguous():
        cur_id = cur_id.contiguous()

    threads_per_block = 32

    num_threads = len(cur_id)
    num_block = math.ceil(num_threads / threads_per_block)
    block_per_grid = num_block

    match = torch.zeros(cur_id.shape, dtype=torch.int64, device=cur_id.device)

    all_id_c = cuda.as_cuda_array(all_sorted_id)
    cur_id_c = cuda.as_cuda_array(cur_id)
    match_c = cuda.as_cuda_array(match)

    find_id_kernel[threads_per_block, block_per_grid](all_id_c, cur_id_c, match_c)
    return match
