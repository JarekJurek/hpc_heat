from os.path import join
import sys

import numpy as np
from time import perf_counter
from numba import cuda


def load_data(load_dir, bid):
    SIZE = 512
    u = np.zeros((SIZE + 2, SIZE + 2))
    u[1:-1, 1:-1] = np.load(join(load_dir, f"{bid}_domain.npy"))
    interior_mask = np.load(join(load_dir, f"{bid}_interior.npy"))
    return u, interior_mask


@cuda.jit
def jacobi_kernel(u, u_new, interior_mask):
    i, j = cuda.grid(2)
    
    if 1 <= i < u.shape[0] - 1 and 1 <= j < u.shape[1] - 1:
        u_new[i, j] = u[i, j]
        
        if interior_mask[i-1, j-1]:
            u_new[i, j] = 0.25 * (u[i+1, j] + u[i-1, j] + u[i, j+1] + u[i, j-1])

def jacobi_helper(u, interior_mask, max_iter):
    u = np.copy(u)
    
    u_d = cuda.to_device(u)
    u_new_d = cuda.to_device(u) 
    mask_d = cuda.to_device(interior_mask)
    
    threads_per_block = (16, 16)
    blocks_per_grid_x = (u.shape[0] + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (u.shape[1] + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
    
    for _ in range(max_iter):
        jacobi_kernel[blocks_per_grid, threads_per_block](u_d, u_new_d, mask_d)

        u_d, u_new_d = u_new_d, u_d
    
    result = u_d.copy_to_host()
    return result


def summary_stats(u, interior_mask):
    u_interior = u[1:-1, 1:-1][interior_mask]
    mean_temp = u_interior.mean()
    std_temp = u_interior.std()
    pct_above_18 = np.sum(u_interior > 18) / u_interior.size * 100
    pct_below_15 = np.sum(u_interior < 15) / u_interior.size * 100
    return {
        'mean_temp': mean_temp,
        'std_temp': std_temp,
        'pct_above_18': pct_above_18,
        'pct_below_15': pct_below_15,
    }

if __name__ == '__main__':
    # Load data
    LOAD_DIR = '/dtu/projects/02613_2025/data/modified_swiss_dwellings/'
    with open(join(LOAD_DIR, 'building_ids.txt'), 'r') as f:
        building_ids = f.read().splitlines()

    if len(sys.argv) < 2:
        N = 1
    else:
        N = int(sys.argv[1])
    building_ids = building_ids[:N]
    
    # Load floor plans
    load_start = perf_counter()
    all_u0 = np.empty((N, 514, 514))
    all_interior_mask = np.empty((N, 512, 512), dtype='bool')
    for i, bid in enumerate(building_ids):
        u0, interior_mask = load_data(LOAD_DIR, bid)
        all_u0[i] = u0
        all_interior_mask[i] = interior_mask
    load_end = perf_counter()
    load_time = load_end - load_start
    print(f"Time to load {N} floorplans: {load_time:.2f} seconds")

    # Run jacobi iterations for each floor plan
    MAX_ITER = 20_000

    all_u = np.empty_like(all_u0)
    building_times = []

    # Run and time
    start = perf_counter()
    for i, (u0, interior_mask) in enumerate(zip(all_u0, all_interior_mask)):
        u = jacobi_helper(u0, interior_mask, MAX_ITER)
        all_u[i] = u
    end = perf_counter()
    process_time = end - start
    print(f"Time to process {N} buildings: {process_time:.2f} seconds")

    # Estimate total time for all 4571 floor plans
    total_load_time = (load_time / N) * 4571
    total_process_time = (process_time / N) * 4571

    print(f"Estimated time to load all 4571 floorplans: {total_load_time:.2f} seconds or {total_load_time / 60:.2f} minutes or {total_load_time / 3600:.2f} hours or {total_load_time / 86400:.2f} days")
    print(f"Estimated time to process all 4571 floorplans: {total_process_time:.2f} seconds or {total_process_time / 60:.2f} minutes or {total_process_time / 3600:.2f} hours or {total_process_time / 86400:.2f} days")
    
