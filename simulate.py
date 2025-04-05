from os.path import join
import sys
import matplotlib.pyplot as plt
import numpy as np
from time import perf_counter

def load_data(load_dir, bid):
    SIZE = 512
    u = np.zeros((SIZE + 2, SIZE + 2))
    u[1:-1, 1:-1] = np.load(join(load_dir, f"{bid}_domain.npy"))
    interior_mask = np.load(join(load_dir, f"{bid}_interior.npy"))
    return u, interior_mask


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

##### TASK 1 #####
def visualize_data(load_dir, building_ids):
    for bid in building_ids:
        u, interior_mask = load_data(load_dir, bid)
        
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        im1 = ax1.imshow(u[1:-1, 1:-1], cmap='hot', vmin=0, vmax=30)
        ax1.set_title(f"Building {bid} - Initial Temperature", fontweight='bold')
        plt.colorbar(im1, ax=ax1, label="Temperature (°C)")
        
        _ = ax2.imshow(interior_mask, cmap='binary')
        ax2.set_title(f"Building {bid} - Interior Mask", fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f"Figures/building_{bid}_input.png")
        plt.close()


##### TASK 3 #####
def visualize_results(building_ids, all_u0, all_u, all_interior_mask, output_dir="Figures/results"):
    for i, bid in enumerate(building_ids):
        u0 = all_u0[i]
        u = all_u[i]
        interior_mask = all_interior_mask[i]
        
        # Get statistics for this building
        stats = summary_stats(u, interior_mask)
        
        # Create a figure with 3 subplots
        _, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        # Plot initial temperature
        im1 = ax1.imshow(u0[1:-1, 1:-1], cmap='hot', vmin=0, vmax=30)
        ax1.set_title(f"Building {bid} - Initial Temperature", fontweight='bold')
        plt.colorbar(im1, ax=ax1, label="Temperature (°C)")
        
        # Plot final temperature
        im2 = ax2.imshow(u[1:-1, 1:-1], cmap='hot', vmin=0, vmax=30)
        ax2.set_title(f"Building {bid} - Final Temperature", fontweight='bold')
        plt.colorbar(im2, ax=ax2, label="Temperature (°C)")
        
        # Plot temperature histogram for interior points
        u_interior = u[1:-1, 1:-1][interior_mask]
        ax3.hist(u_interior.flatten(), bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        
        # Add lines for key statistics
        ax3.axvline(stats['mean_temp'], color='red', linestyle='dashed', linewidth=2, 
                    label=f"Mean: {stats['mean_temp']:.2f}°C")
        ax3.axvline(18, color='green', linestyle='dashed', linewidth=2, 
                    label=f"Above 18°C: {stats['pct_above_18']:.1f}%")
        ax3.axvline(15, color='orange', linestyle='dashed', linewidth=2, 
                    label=f"Below 15°C: {stats['pct_below_15']:.1f}%")
        
        ax3.set_title(f"Building {bid} - Temperature Distribution", fontweight='bold')
        ax3.set_xlabel("Temperature (°C)")
        ax3.set_ylabel("Frequency")
        ax3.legend()
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/building_{bid}_results.png")
        plt.close()


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
    
    # Visualize some examples (Task 1)
    # num_to_visualize = 3
    # if len(sys.argv) > 1:
    #     num_to_visualize = int(sys.argv[1])

    # building_ids_to_visualize = building_ids[:num_to_visualize]
    # visualize_data(LOAD_DIR, building_ids_to_visualize)
    
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

    # Visualize results (Task 3)
    visualize_results(building_ids, all_u0, all_u, all_interior_mask)

    # Print summary timing (Task 2) 
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