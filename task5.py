import argparse
import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter
from os.path import join
from multiprocessing import Pool

from simulate import load_data, jacobi, summary_stats

LOAD_DIR = '/dtu/projects/02613_2025/data/modified_swiss_dwellings/'
MAX_ITER = 20_000
ABS_TOL = 1e-4

# Global shared arrays
all_u0 = None
all_interior_mask = None
building_ids = None


def process_index(i):
    u0 = all_u0[i]
    mask = all_interior_mask[i]
    bid = building_ids[i]
    u = jacobi(u0, mask, MAX_ITER, ABS_TOL)
    stats = summary_stats(u, mask)
    return bid, u, stats

def parallel_run(N, n_proc):
    global all_u0, all_interior_mask, building_ids

    with open(join(LOAD_DIR, 'building_ids.txt'), 'r') as f:
        building_ids = f.read().splitlines()[:N]

    all_u0 = np.empty((N, 514, 514))
    all_interior_mask = np.empty((N, 512, 512), dtype=bool)
    for i, bid in enumerate(building_ids):
        u0, interior_mask = load_data(LOAD_DIR, bid)
        all_u0[i] = u0
        all_interior_mask[i] = interior_mask

    start = perf_counter()
    with Pool(processes=n_proc) as pool:
        results = pool.map(process_index, range(N))
    end = perf_counter()

    return end - start


if __name__ == '__main__':
    N = 60
    max_proc = 20
    num_processes = list(range(1, max_proc + 1))

    base_time = None
    times = []
    speedups = []

    print(f"Running Task 5a-d on {N} buildings\n")

    for p in num_processes:
        print(f"Running with {p} process(es)...")
        t = parallel_run(N, p)
        print(f"Time: {t:.2f} seconds")
        times.append(t)
        if p == 1:
            base_time = t
        speedups.append(base_time / t)

    # Plotting Task 5a
    plt.figure(figsize=(8, 5))
    plt.plot(num_processes, speedups, marker="o", linestyle="-", label="Speedup")
    plt.axhline(y=max(speedups), color="r", linestyle="--", label="Max Speedup")
    plt.xlabel("Number of Processes")
    plt.ylabel("Speedup")
    plt.title(f"Speedup vs Number of Processes for {N} buildings")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("task5a_speedup_plot.png")
    plt.show()

    # Task 5b–d: Amdahl's Law analysis
    S_max = max(speedups)
    n_at_max = num_processes[speedups.index(S_max)]
    p_est = (n_at_max * (1 - 1 / S_max)) / (n_at_max - 1)
    theoretical_max_speedup = 1 / (1 - p_est)

    print("\n-------------------- Amdahl's Law --------------------")
    print(f"Estimated parallel fraction (p): {p_est:.4f}")
    print(f"Theoretical max speed-up: {theoretical_max_speedup:.2f}")
    print(f"Achieved speed-up: {S_max:.2f} with {n_at_max} processes")

    estimated_full_time = min(times) * 4571 / N
    print(f"\nEstimated time to process all 4571 buildings: {estimated_full_time:.2f} seconds")
    print(f"≈ {estimated_full_time / 3600:.2f} hours or {estimated_full_time / (3600 * 24):.2f} days")
    print("------------------------------------------------------")
