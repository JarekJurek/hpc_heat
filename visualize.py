import os
import matplotlib.pyplot as plt


# TASK 1
def visualize_data(bid, u, interior_mask):
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    im1 = ax1.imshow(u[1:-1, 1:-1], cmap='hot', vmin=0, vmax=30)
    ax1.set_title(
        f"Building {bid} - Initial Temperature", fontweight='bold')
    plt.colorbar(im1, ax=ax1, label="Temperature (°C)")

    _ = ax2.imshow(interior_mask, cmap='binary')
    ax2.set_title(f"Building {bid} - Interior Mask", fontweight='bold')

    plt.tight_layout()
    plt.savefig(f"Figures/building_{bid}_input.png")
    plt.close()


# TASK 3
def visualize_results(building_ids, all_u0, all_u, all_interior_mask, stats, output_dir="Figures/results"):
    os.makedirs(output_dir, exist_ok=True)
    for i, bid in enumerate(building_ids):
        u0 = all_u0[i]
        u = all_u[i]
        interior_mask = all_interior_mask[i]
        stat = stats[i]

        # Create a figure with 3 subplots
        _, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

        # Plot initial temperature
        im1 = ax1.imshow(u0[1:-1, 1:-1], cmap='hot', vmin=0, vmax=30)
        ax1.set_title(
            f"Building {bid} - Initial Temperature", fontweight='bold')
        plt.colorbar(im1, ax=ax1, label="Temperature (°C)")

        # Plot final temperature
        im2 = ax2.imshow(u[1:-1, 1:-1], cmap='hot', vmin=0, vmax=30)
        ax2.set_title(f"Building {bid} - Final Temperature", fontweight='bold')
        plt.colorbar(im2, ax=ax2, label="Temperature (°C)")

        # Plot temperature histogram for interior points
        u_interior = u[1:-1, 1:-1][interior_mask]
        ax3.hist(u_interior.flatten(), bins=50,
                 color='skyblue', edgecolor='black', alpha=0.7)

        # Add lines for key statistics
        ax3.axvline(stat['mean_temp'], color='red', linestyle='dashed', linewidth=2,
                    label=f"Mean: {stat['mean_temp']:.2f}°C")
        ax3.axvline(18, color='green', linestyle='dashed', linewidth=2,
                    label=f"Above 18°C: {stat['pct_above_18']:.1f}%")
        ax3.axvline(15, color='orange', linestyle='dashed', linewidth=2,
                    label=f"Below 15°C: {stat['pct_below_15']:.1f}%")

        ax3.set_title(
            f"Building {bid} - Temperature Distribution", fontweight='bold')
        ax3.set_xlabel("Temperature (°C)")
        ax3.set_ylabel("Frequency")
        ax3.legend()

        plt.tight_layout()
        plt.savefig(f"{output_dir}/building_{bid}_results.png")
        plt.close()
