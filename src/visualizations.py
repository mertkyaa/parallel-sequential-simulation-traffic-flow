"""
All visualisations for the traffic simulation report.

Outputs (saved to output/plots/):
  1. traffic_density_heatmap.png   – node visit density across the grid
  2. speedup_graph.png             – sequential vs parallel wall time + speedup ratio
  3. congestion_timeline.png       – avg queue length over simulated time
  4. sector_workload.png           – vehicle distribution across sectors
  5. wait_time_distribution.png    – histogram comparing wait times
  6. partition_analysis.png        – cross-sector edge count for different partitions
"""

import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # no display required
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns

import config


# ── colour palette
PALETTE = {
    "seq": "#E74C3C",
    "par": "#2ECC71",
    "accent": "#3498DB",
    "boundary": "#F39C12",
    "grid": "#ECF0F1",
}

plt.rcParams.update({
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11,
})


# ---------------------------------------------------------------------------
# 1. Traffic density heatmap
# ---------------------------------------------------------------------------

def plot_heatmap(
    node_visits: Dict[Tuple, int],
    grid_size: int,
    title: str = "Traffic Density Heatmap",
    filename: str = "traffic_density_heatmap.png",
    sector_partitioner=None,
) -> None:
    grid = np.zeros((grid_size, grid_size))
    for (i, j), cnt in node_visits.items():
        grid[i][j] = cnt

    fig, ax = plt.subplots(figsize=(10, 9))
    sns.heatmap(
        grid,
        ax=ax,
        cmap="YlOrRd",
        linewidths=0,
        xticklabels=False,
        yticklabels=False,
        cbar_kws={"label": "Vehicle Visits", "shrink": 0.8},
    )

    # Draw sector boundaries
    if sector_partitioner is not None:
        n = grid_size
        nx_s, ny_s = sector_partitioner.num_x, sector_partitioner.num_y
        for sx in range(1, nx_s):
            x = sx * (n / nx_s)
            ax.axvline(x=x, color=PALETTE["boundary"], linewidth=2.5,
                       linestyle="--", label="Sector boundary" if sx == 1 else "")
        for sy in range(1, ny_s):
            y = sy * (n / ny_s)
            ax.axhline(y=y, color=PALETTE["boundary"], linewidth=2.5, linestyle="--")
        handles = [mpatches.Patch(color=PALETTE["boundary"],
                                  label="Sector boundary (sync point)")]
        ax.legend(handles=handles, loc="upper right", fontsize=9)

    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("X (East-West blocks)")
    ax.set_ylabel("Y (North-South blocks)")
    fig.tight_layout()
    _save(fig, filename)


# ---------------------------------------------------------------------------
# 2. Speedup graph
# ---------------------------------------------------------------------------

def plot_speedup(
    vehicle_counts: List[int],
    seq_times: List[float],
    par_times: List[float],
    num_sectors: int,
    filename: str = "speedup_graph.png",
) -> None:
    speedups = [s / p for s, p in zip(seq_times, par_times)]
    theoretical = [min(num_sectors, s / p) for s, p in zip(seq_times, par_times)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Wall time comparison
    ax1.plot(vehicle_counts, seq_times, "o-", color=PALETTE["seq"],
             linewidth=2, markersize=7, label="Sequential")
    ax1.plot(vehicle_counts, par_times, "s-", color=PALETTE["par"],
             linewidth=2, markersize=7, label=f"Parallel ({num_sectors} sectors)")
    ax1.fill_between(vehicle_counts, seq_times, par_times,
                     alpha=0.15, color=PALETTE["par"])
    ax1.set_xlabel("Number of Vehicles")
    ax1.set_ylabel("Wall-clock Time (seconds)")
    ax1.set_title("Execution Time: Sequential vs Parallel", fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Speedup ratio
    ax2.plot(vehicle_counts, speedups, "D-", color=PALETTE["accent"],
             linewidth=2, markersize=7, label="Measured speedup")
    ax2.axhline(y=num_sectors, color="gray", linewidth=1.5, linestyle="--",
                label=f"Theoretical max ({num_sectors}×)")
    ax2.axhline(y=1.0, color="black", linewidth=1, linestyle=":")
    ax2.fill_between(vehicle_counts, 1, speedups, alpha=0.2, color=PALETTE["accent"])
    ax2.set_xlabel("Number of Vehicles")
    ax2.set_ylabel("Speedup (×)")
    ax2.set_title("Parallel Speedup", fontweight="bold")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=0)

    _annotate_amdahl(ax2, vehicle_counts, speedups, num_sectors)

    fig.suptitle("Sequential vs Parallel Performance Comparison", fontsize=14,
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, filename)


def _annotate_amdahl(ax, x_vals, speedups, num_sectors):
    if len(speedups) >= 2:
        peak_idx = int(np.argmax(speedups))
        ax.annotate(
            f"Peak: {speedups[peak_idx]:.2f}×",
            xy=(x_vals[peak_idx], speedups[peak_idx]),
            xytext=(x_vals[peak_idx], speedups[peak_idx] + 0.15),
            fontsize=9, ha="center",
            arrowprops=dict(arrowstyle="->", color="black", lw=1),
        )


# ---------------------------------------------------------------------------
# 3. Congestion timeline
# ---------------------------------------------------------------------------

def plot_congestion_timeline(
    queue_snapshots: List[Tuple],      # (sim_time, node, queue_len)
    seq_snapshots: Optional[List[Tuple]] = None,
    filename: str = "congestion_timeline.png",
) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))

    def _aggregate(snapshots, label, color, linestyle="-"):
        if not snapshots:
            return
        bins: Dict[float, List[int]] = defaultdict(list)
        for t, _, q in snapshots:
            bucket = round(t / 5) * 5
            bins[bucket].append(q)
        times = sorted(bins.keys())
        avgs = [np.mean(bins[t]) for t in times]
        ax.plot(times, avgs, linestyle=linestyle, color=color,
                linewidth=2, label=label)
        ax.fill_between(times, avgs, alpha=0.1, color=color)

    _aggregate(seq_snapshots, "Sequential", PALETTE["seq"])
    _aggregate(queue_snapshots, "Parallel", PALETTE["par"], linestyle="--")

    # Rush-hour band
    ax.axvspan(config.RUSH_HOUR_START, config.RUSH_HOUR_END,
               alpha=0.08, color="orange", label="Rush hour window")
    ax.axvline(x=(config.RUSH_HOUR_START + config.RUSH_HOUR_END) / 2,
               color="orange", linewidth=1.5, linestyle=":", alpha=0.6)

    ax.set_xlabel("Simulated Time (minutes)")
    ax.set_ylabel("Average Intersection Queue Length")
    ax.set_title("Congestion Timeline – Demonstrating Peak-Load Build-up",
                 fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, filename)


# ---------------------------------------------------------------------------
# 4. Sector workload balance
# ---------------------------------------------------------------------------

def plot_sector_workload(
    vehicles_per_sector: Dict[int, int],
    nodes_per_sector: Dict[int, int],
    filename: str = "sector_workload.png",
) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    sectors = sorted(vehicles_per_sector.keys())
    labels = [f"Sector {s}" for s in sectors]
    vcounts = [vehicles_per_sector[s] for s in sectors]
    ncounts = [nodes_per_sector[s] for s in sectors]

    colors = [PALETTE["par"], PALETTE["accent"], PALETTE["seq"], PALETTE["boundary"]]

    ax1.bar(labels, vcounts, color=colors[:len(sectors)], alpha=0.85, edgecolor="white")
    for i, v in enumerate(vcounts):
        ax1.text(i, v + 2, str(v), ha="center", va="bottom", fontsize=10)
    ax1.set_ylabel("Initial Vehicle Assignments")
    ax1.set_title("Vehicle Load per Sector", fontweight="bold")
    ax1.set_ylim(0, max(vcounts) * 1.15)
    ax1.grid(axis="y", alpha=0.3)

    ax2.bar(labels, ncounts, color=colors[:len(sectors)], alpha=0.85, edgecolor="white")
    for i, v in enumerate(ncounts):
        ax2.text(i, v + 2, str(v), ha="center", va="bottom", fontsize=10)
    ax2.set_ylabel("Node Count")
    ax2.set_title("Intersection Count per Sector", fontweight="bold")
    ax2.set_ylim(0, max(ncounts) * 1.15)
    ax2.grid(axis="y", alpha=0.3)

    imbalance = np.std(vcounts) / np.mean(vcounts) * 100
    fig.suptitle(
        f"Sector Workload Balance  (load imbalance = {imbalance:.1f} %)",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    _save(fig, filename)


# ---------------------------------------------------------------------------
# 5. Wait-time distribution comparison
# ---------------------------------------------------------------------------

def plot_wait_distribution(
    seq_wait: List[float],
    par_wait: List[float],
    filename: str = "wait_time_distribution.png",
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.hist(seq_wait, bins=40, alpha=0.65, color=PALETTE["seq"],
            label=f"Sequential  (μ={np.mean(seq_wait):.2f} min)", density=True)
    ax.hist(par_wait, bins=40, alpha=0.65, color=PALETTE["par"],
            label=f"Parallel    (μ={np.mean(par_wait):.2f} min)", density=True)

    ax.axvline(np.mean(seq_wait), color=PALETTE["seq"], linewidth=2, linestyle="--")
    ax.axvline(np.mean(par_wait), color=PALETTE["par"], linewidth=2, linestyle="--")

    ax.set_xlabel("Wait Time at Intersections (minutes)")
    ax.set_ylabel("Density")
    ax.set_title("Vehicle Wait-Time Distribution – Sequential vs Parallel",
                 fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, filename)


# ---------------------------------------------------------------------------
# 6. Partition analysis (justification)
# ---------------------------------------------------------------------------

def plot_partition_analysis(
    partition_configs: List[Tuple[str, int, int, int]],   # (label, nx, ny, cross_edges)
    filename: str = "partition_analysis.png",
) -> None:
    labels = [c[0] for c in partition_configs]
    cross_edges = [c[3] for c in partition_configs]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, cross_edges, color=PALETTE["accent"], alpha=0.8,
                  edgecolor="white", width=0.55)

    # Highlight chosen partition
    chosen_label = "2×2 Grid\n(chosen)"
    for i, lbl in enumerate(labels):
        if "chosen" in lbl:
            bars[i].set_color(PALETTE["par"])
            bars[i].set_edgecolor(PALETTE["boundary"])
            bars[i].set_linewidth(2)

    for bar, val in zip(bars, cross_edges):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                str(val), ha="center", va="bottom", fontsize=10)

    ax.set_ylabel("Cross-Sector Edges (sync overhead proxy)")
    ax.set_title(
        "Partition Strategy Comparison\n"
        "Fewer cross-sector edges → lower synchronisation overhead → better speedup",
        fontweight="bold",
    )
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, filename)


# ---------------------------------------------------------------------------
# 7. Summary table (text saved as PNG)
# ---------------------------------------------------------------------------

def plot_summary_table(
    seq_metrics: Dict,
    par_metrics: Dict,
    filename: str = "summary_table.png",
) -> None:
    seq_wt = seq_metrics["wall_time"]
    par_wt = par_metrics["wall_time"]
    speedup = seq_wt / par_wt if par_wt > 0 else float("nan")

    def _safe_mean(lst):
        return np.mean(lst) if lst else float("nan")

    rows = [
        ["Metric", "Sequential", "Parallel", "Notes"],
        ["Wall-clock time (s)", f"{seq_wt:.2f}", f"{par_wt:.2f}",
         f"Speedup: {speedup:.2f}×"],
        ["Vehicles completed", str(seq_metrics["completed"]),
         str(par_metrics["completed"]), "Correctness check"],
        ["Avg wait time (min)", f"{_safe_mean(seq_metrics['wait_times']):.3f}",
         f"{_safe_mean(par_metrics['wait_times']):.3f}", "Should be similar"],
        ["Avg travel time (min)", f"{_safe_mean(seq_metrics['travel_times']):.3f}",
         f"{_safe_mean(par_metrics['travel_times']):.3f}", ""],
        ["Peak queue length", str(_peak_queue(seq_metrics["queue_snapshots"])),
         str(_peak_queue(par_metrics["queue_snapshots"])), "Congestion indicator"],
    ]

    fig, ax = plt.subplots(figsize=(13, 3.5))
    ax.axis("off")
    tbl = ax.table(
        cellText=rows[1:],
        colLabels=rows[0],
        cellLoc="center",
        loc="center",
        bbox=[0, 0, 1, 1],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.auto_set_column_width(col=list(range(len(rows[0]))))

    # Style header
    for j in range(len(rows[0])):
        tbl[(0, j)].set_facecolor("#2C3E50")
        tbl[(0, j)].set_text_props(color="white", fontweight="bold")

    # Style speedup row
    tbl[(1, 3)].set_facecolor("#D5F5E3")

    ax.set_title("Simulation Results Summary", fontsize=13, fontweight="bold", pad=10)
    fig.tight_layout()
    _save(fig, filename)


def _peak_queue(snapshots):
    if not snapshots:
        return 0
    return max(q for _, _, q in snapshots)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _save(fig: plt.Figure, filename: str) -> None:
    os.makedirs(config.PLOTS_DIR, exist_ok=True)
    path = os.path.join(config.PLOTS_DIR, filename)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {path}")
