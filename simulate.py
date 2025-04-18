from os.path import join
import sys

import numpy as np
from time import perf_counter
from visualize import visualize_data, visualize_results


def load_data(load_dir, bid):
    SIZE = 512
    u = np.zeros((SIZE + 2, SIZE + 2))
    u[1:-1, 1:-1] = np.load(join(load_dir, f"{bid}_domain.npy"))
    interior_mask = np.load(join(load_dir, f"{bid}_interior.npy"))
    return u, interior_mask


# @profile  # TASK 4
def jacobi(u, interior_mask, max_iter, atol=1e-6):
    u = np.copy(u)

    for i in range(max_iter):
        # Compute average of left, right, up and down neighbors, see eq. (1)
        u_new = 0.25 * (u[1:-1, :-2] + u[1:-1, 2:] + u[:-2, 1:-1] + u[2:, 1:-1])
        u_new_interior = u_new[interior_mask]
        delta = np.abs(u[1:-1, 1:-1][interior_mask] - u_new_interior).max()
        u[1:-1, 1:-1][interior_mask] = u_new_interior

        if delta < atol:
            break
    return u


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
    
    # TASK 1: Visualize some examples
    # num_to_visualize = 3
    # if len(sys.argv) > 1:
    #     num_to_visualize = int(sys.argv[1])

    # building_ids_to_visualize = building_ids[:num_to_visualize]
    # for bid in building_ids:
    #     print(bid)
    #     u, interior_mask = load_data(LOAD_DIR, bid)
    #     visualize_data(bid, u, interior_mask)

    # Load floor plans
    load_start = perf_counter()
    all_u0 = np.empty((N, 514, 514))
    all_interior_mask = np.empty((N, 512, 512), dtype='bool')
    for i, bid in enumerate(building_ids):
        u0, interior_mask = load_data(LOAD_DIR, bid)
        all_u0[i] = u0
        all_interior_mask[i] = interior_mask
    load_end = perf_counter()


    # Run jacobi iterations for each floor plan
    MAX_ITER = 20_000
    ABS_TOL = 1e-4

    all_u = np.empty_like(all_u0)
    building_times = []
    
    for i, (u0, interior_mask) in enumerate(zip(all_u0, all_interior_mask)):
        start_time = perf_counter()
        u = jacobi(u0, interior_mask, MAX_ITER, ABS_TOL)
        end_time = perf_counter()
        building_times.append(end_time - start_time)
        all_u[i] = u
        print(f"Building {building_ids[i]} processed in {building_times[-1]:.2f} seconds")

    # Get statistics
    all_stats = []
    for u, mask in zip(all_u, all_interior_mask):
        stats = summary_stats(u, mask)
        all_stats.append(stats)

    # TASK3: Visualize results
    # visualize_results(building_ids, all_u0, all_u, all_interior_mask, all_stats)
    
    # TASK 2: Print summary timing
    avg_time = np.mean(building_times)
    total_buildings = 4571
    estimated_total_time = avg_time * total_buildings
    
    print("\n-------------------- Timing results --------------------")
    print(f"Time to load {N} buildings: {load_end - load_start:.2f} seconds")
    print(f"Average time per building: {avg_time:.2f} seconds")
    print(f"Estimated time for all {total_buildings} buildings: {estimated_total_time:.2f} seconds")
    print(f"That's approximately {estimated_total_time/3600:.2f} hours or {estimated_total_time/(3600*24):.2f} days")
    print("------------------------------------------------------------\n")
    

    # Print summary statistics in CSV format
    stat_keys = ['mean_temp', 'std_temp', 'pct_above_18', 'pct_below_15']
    print('building_id, ' + ', '.join(stat_keys))  # CSV header
    for bid, u, interior_mask in zip(building_ids, all_u, all_interior_mask):
        stats = summary_stats(u, interior_mask)
        print(f"{bid},", ", ".join(str(stats[k]) for k in stat_keys))