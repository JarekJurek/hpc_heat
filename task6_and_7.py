import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter
from os.path import join
from multiprocessing import Pool
import sys
from numba import jit
from functools import partial

from simulate import load_data, jacobi, summary_stats

LOAD_DIR = '/dtu/projects/02613_2025/data/modified_swiss_dwellings/'
MAX_ITER = 20_000
ABS_TOL = 1e-4

# Global shared arrays
all_u0 = None
all_interior_mask = None
building_ids = None


def process_index(i,numba=False):
    u0 = all_u0[i]
    mask = all_interior_mask[i]
    bid = building_ids[i]

    if numba: 
        u=jacobi_numba(u0, mask, MAX_ITER, ABS_TOL) 
    else: 
        u=jacobi(u0, mask, MAX_ITER, ABS_TOL)

    stats = summary_stats(u, mask)
    return bid, u, stats

def parallel_run(N, n_proc, numba=False):
    
    for i, bid in enumerate(building_ids):
        u0, interior_mask = load_data(LOAD_DIR, bid)
        all_u0[i] = u0
        all_interior_mask[i] = interior_mask

    func = partial(process_index, numba=numba)
    start = perf_counter()
    with Pool(processes=n_proc) as pool:
        results = pool.map(func, range(N))
    end = perf_counter()

    return end - start

# Task 6 - Dynamic Scheduling
def parallel_run_dynamic_scheduling(N, n_proc):
    chunksize = max(1, N // n_proc)
    start = perf_counter()
    with Pool(processes=n_proc) as pool:
        list(pool.imap_unordered(process_index, range(N)))
    end = perf_counter()
    return end - start


# Task 7 
# It looks different from the original jacobi function because NUMBA doesn't support advanced indexing
@jit(nopython=True)
def jacobi_numba(u, interior_mask, max_iter, atol=1e-6):
    u = np.copy(u)
    idx = np.where(interior_mask)  # precompute valid indices

    for _ in range(max_iter):
        u_new = 0.25 * (u[1:-1, :-2] + u[1:-1, 2:] + u[:-2, 1:-1] + u[2:, 1:-1])
        delta = 0.0
        for i in range(len(idx[0])):
            y, x = idx[0][i], idx[1][i]
            d = abs(u[1 + y, 1 + x] - u_new[y, x])
            if d > delta:
                delta = d
            u[1 + y, 1 + x] = u_new[y, x]

        if delta < atol:
            break
    return u


def Amdahl_analysis(times, speedups, num_processes,type=None):
        S_max = max(speedups)
        n_at_max = num_processes[speedups.index(S_max)]
        p_est = (n_at_max * (1 - 1 / S_max)) / (n_at_max - 1)
        theoretical_max_speedup = 1 / (1 - p_est)

        print("\n-------------------- Amdahl's Law --------------------")
        if type  != None:
            print(f"Amdahl's Law analysis for {type} experiment")
        print(f"Estimated parallel fraction (p): {p_est:.4f}")
        print(f"Theoretical max speed-up: {theoretical_max_speedup:.2f}")
        print(f"Achieved speed-up: {S_max:.2f} with {n_at_max} processes")

        estimated_full_time = min(times) * 4571 / N
        print(f"\nEstimated time to process all 4571 buildings: {estimated_full_time:.2f} seconds")
        print(f"≈ {estimated_full_time / 3600:.2f} hours or {estimated_full_time / (3600 * 24):.2f} days")
        print("------------------------------------------------------")
        return theoretical_max_speedup

def run_experiment(num_processes,N,type = None):
    times = []
    speedups = []
    base_time = None

    if type == None:
        print(f"Running Task 5a on {N} buildings\n")
    else:
        print(f"Running Task {type} on {N} buildings\n")
    for p in num_processes:
        print(f"Running with {p} process(es)...")
        if type == "dynamic":
            t = parallel_run_dynamic_scheduling(N, p)
        elif type == "numba":
            t = parallel_run(N, p, numba=True)
        else:
            t = parallel_run(N, p)
        print(f"Time: {t:.2f} seconds")
        times.append(t)
        if p == 1:
            base_time = t
        speedups.append(base_time / t)
    return times, speedups

def plot_speedup(num_processes, speedups, theoretical_max,  N):
    fig, ax = plt.subplots(figsize=(8, 5))
    plt.plot(num_processes, speedups, marker="o", linestyle="-", color="red",label="Speedup")
    plt.axhline(y=theoretical_max, color="r", linestyle="--", label="Theoretical max speedup")
    plt.xlabel("Number of Processes")
    plt.ylabel("Speedup")
    plt.title(f"Speedup vs Number of Processes for {N} buildings")
    plt.grid()
    plt.tight_layout()
    return fig, ax


if __name__ == '__main__':
    N = 60
    N_NUMBA = 20
    max_proc = 20
    num_processes = list(range(1, max_proc + 1))
    dynamic = "dynamic" in sys.argv
    numba = "numba" in sys.argv

    if not dynamic and not numba:
        print("To enable dynamic scheduling or numba, run the script with 'dynamic' and\or 'numba' as an argument.")


    with open(join(LOAD_DIR, 'building_ids.txt'), 'r') as f:
        building_ids = f.read().splitlines()[:N]

    all_u0 = np.empty((N, 514, 514))
    all_interior_mask = np.empty((N, 512, 512), dtype=bool)
    for i, bid in enumerate(building_ids):
        u0, interior_mask = load_data(LOAD_DIR, bid)
        all_u0[i] = u0
        all_interior_mask[i] = interior_mask

    # Run the experiment for Task 5a
    times, speedups = run_experiment(num_processes,N)
    theoretical_max = Amdahl_analysis(times, speedups, num_processes)
    fig, ax = plot_speedup(num_processes, speedups,theoretical_max, N)    

    # Run the experiment for Task 6
    if dynamic:
        times_dynamic, speedups_dynamic = run_experiment(num_processes,N,type="dynamic")
        
        theoretical_max = Amdahl_analysis(times_dynamic, speedups_dynamic,  num_processes,type="dynamic scheduling")
        ax.plot(num_processes, speedups_dynamic, marker="o", linestyle="-", color="green",label="Dynamic Scheduling Speedup")
        ax.axhline(y=theoretical_max, color="g", linestyle="--", label="Theoretical max Dynamic Speedup")
        ax.legend()
        fig.savefig(f"task6_with_dynamic_scheduling.png")

        
    # Run the experiment for Task 7
    if numba:
        times_numba_base, speedups_numba_base = run_experiment(num_processes,N_NUMBA)
        times_numba, speedups_numba = run_experiment(num_processes,N_NUMBA,type="numba")

        theoretical_max_base = Amdahl_analysis(times_numba_base,speedups_numba_base, num_processes,type="Numba base")
        theoretical_max = Amdahl_analysis(times_numba,speedups_numba, num_processes,type="Numba")
        
        fig, ax = plot_speedup(num_processes, speedups_numba_base, theoretical_max_base, N_NUMBA)
        ax.plot(num_processes, speedups_numba, marker="o", linestyle="-", color="blue",label="Numba Speedup")
        ax.axhline(y=theoretical_max, color="b", linestyle="--", label="Max Numba Speedup")
        ax.legend()
        fig.savefig(f"task7_with_numba.png")

