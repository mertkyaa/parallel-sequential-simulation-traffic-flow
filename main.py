"""
main.py – full simulation run: sequential → parallel → all plots.

Usage:
    python main.py                  # 1 000 vehicles, default config
    python main.py --vehicles 500   # custom vehicle count
    python main.py --no-parallel    # sequential only
"""

import argparse
import json
import os
import platform
import multiprocessing as mp

import numpy as np

# macOS / Linux: prefer 'fork' (copies parent memory; faster than spawn).
# Windows: only 'spawn' is available, so leave the default in place.
if mp.get_start_method(allow_none=True) is None and platform.system() != "Windows":
    mp.set_start_method("fork")

import config
from src.network import CityNetwork
from src.sector import SectorPartitioner
from src.sequential import SequentialSimulation
from src.parallel import ParallelSimulation
from src.visualizations import (
    plot_heatmap,
    plot_congestion_timeline,
    plot_sector_workload,
    plot_wait_distribution,
    plot_summary_table,
    plot_partition_analysis,
)


def banner(msg):
    print("\n" + "─" * 60)
    print(f"  {msg}")
    print("─" * 60)


def main(num_vehicles=config.NUM_VEHICLES, run_parallel=True):
    os.makedirs(config.PLOTS_DIR, exist_ok=True)
    os.makedirs(config.DATA_DIR, exist_ok=True)

    # ── 1. Build network ─────────────────────────────────────
    banner("Building City Road Network")
    net = CityNetwork(grid_size=config.GRID_SIZE, seed=config.SEED)
    partitioner = SectorPartitioner(
        net, num_x=config.NUM_SECTORS_X, num_y=config.NUM_SECTORS_Y
    )
    report = partitioner.partition_report()

    print(f"  Grid size   : {config.GRID_SIZE}×{config.GRID_SIZE} = {net.num_nodes} nodes")
    print(f"  Road edges  : {net.num_edges}")
    print(f"  Vehicles    : {num_vehicles}")
    print(f"  Sectors     : {report['num_sectors']}  ({config.NUM_SECTORS_X}×{config.NUM_SECTORS_Y})")
    print(f"  Boundary nodes      : {report['boundary_nodes']}")
    print(f"  Cross-sector edges  : {report['cross_sector_edges']} / {report['total_edges']}")

    print("\n  Generating partition analysis ...")
    pconfigs = []
    for nx_s, ny_s in [(1, 1), (1, 2), (2, 2), (2, 3), (3, 3)]:
        if nx_s == 1 and ny_s == 1:
            cross = 0
            label = "No split\n(sequential)"
        else:
            p = SectorPartitioner(net, num_x=nx_s, num_y=ny_s)
            cross = p.cross_sector_edges()
            label = f"{nx_s}×{ny_s} Grid" + ("\n(chosen)" if nx_s == 2 and ny_s == 2 else "")
        pconfigs.append((label, nx_s, ny_s, cross))
    plot_partition_analysis(pconfigs)

    # ── 2. Sequential simulation ──────────────────────────────
    banner("Running Sequential Simulation")
    seq_sim = SequentialSimulation(net, num_vehicles=num_vehicles)
    seq_m = seq_sim.run()

    print(f"  Wall time   : {seq_m['wall_time']:.3f} s")
    print(f"  Completed   : {seq_m['completed']} trips")
    if seq_m["wait_times"]:
        print(f"  Avg wait    : {np.mean(seq_m['wait_times']):.3f} min")
        print(f"  Avg travel  : {np.mean(seq_m['travel_times']):.3f} min")

    print("\n  Generating sequential heatmap ...")
    plot_heatmap(
        seq_m["node_visits"], config.GRID_SIZE,
        title="Traffic Density – Sequential Simulation",
        filename="heatmap_sequential.png",
        sector_partitioner=partitioner,
    )

    # ── 3. Parallel simulation ────────────────────────────────
    par_m = None
    if run_parallel:
        banner("Running Parallel Simulation")
        specs = seq_m["vehicle_specs"]
        light_offsets = seq_m["light_offsets"]
        par_sim = ParallelSimulation(net, partitioner, specs, light_offsets)
        par_m = par_sim.run()

        print(f"  Wall time   : {par_m['wall_time']:.3f} s")
        print(f"  Completed   : {par_m['completed']} trips")
        if par_m["wait_times"]:
            print(f"  Avg wait    : {np.mean(par_m['wait_times']):.3f} min")
            print(f"  Avg travel  : {np.mean(par_m['travel_times']):.3f} min")

        speedup = seq_m["wall_time"] / par_m["wall_time"]
        print(f"\n  *** Speedup : {speedup:.2f}× ***")

        print("\n  Generating parallel heatmap ...")
        plot_heatmap(
            par_m["node_visits"], config.GRID_SIZE,
            title="Traffic Density – Parallel Simulation",
            filename="heatmap_parallel.png",
            sector_partitioner=partitioner,
        )

    # ── 4. Comparison plots ───────────────────────────────────
    banner("Generating Comparison Plots")

    print("  Congestion timeline ...")
    plot_congestion_timeline(
        par_m["queue_snapshots"] if par_m else [],
        seq_snapshots=seq_m["queue_snapshots"],
    )

    if par_m:
        print("  Sector workload ...")
        plot_sector_workload(par_m["vehicles_per_sector"], report["nodes_per_sector"])

        print("  Wait-time distribution ...")
        plot_wait_distribution(seq_m["wait_times"], par_m["wait_times"])

        print("  Summary table ...")
        plot_summary_table(seq_m, par_m)

        summary = {
            "sequential": {
                "wall_time": seq_m["wall_time"],
                "completed": seq_m["completed"],
                "avg_wait": float(np.mean(seq_m["wait_times"])) if seq_m["wait_times"] else 0,
                "avg_travel": float(np.mean(seq_m["travel_times"])) if seq_m["travel_times"] else 0,
            },
            "parallel": {
                "wall_time": par_m["wall_time"],
                "completed": par_m["completed"],
                "avg_wait": float(np.mean(par_m["wait_times"])) if par_m["wait_times"] else 0,
                "avg_travel": float(np.mean(par_m["travel_times"])) if par_m["travel_times"] else 0,
                "speedup": seq_m["wall_time"] / par_m["wall_time"],
                "vehicles_per_sector": {str(k): v for k, v in par_m["vehicles_per_sector"].items()},
            },
            "partition": {k: (v if not isinstance(v, set) else len(v))
                          for k, v in report.items()},
        }
        out = os.path.join(config.DATA_DIR, "simulation_results.json")
        with open(out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n  [saved] {out}")

    banner("Complete")
    print(f"  All plots in: {config.PLOTS_DIR}/")
    for fname in sorted(os.listdir(config.PLOTS_DIR)):
        print(f"    {fname}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Traffic Flow Simulation")
    parser.add_argument("--vehicles", type=int, default=config.NUM_VEHICLES)
    parser.add_argument("--no-parallel", action="store_true")
    args = parser.parse_args()
    main(num_vehicles=args.vehicles, run_parallel=not args.no_parallel)
