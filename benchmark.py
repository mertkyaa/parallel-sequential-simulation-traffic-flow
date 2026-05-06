"""
benchmark.py – measures sequential vs parallel wall-clock time at multiple
vehicle counts and produces the speedup graph + partition analysis.

Usage:
    python benchmark.py
"""

import json
import os
import platform
import multiprocessing as mp

# fork is faster on macOS/Linux; Windows only supports spawn.
if mp.get_start_method(allow_none=True) is None and platform.system() != "Windows":
    mp.set_start_method("fork")

import config
from src.network import CityNetwork
from src.sector import SectorPartitioner
from src.sequential import SequentialSimulation
from src.parallel import ParallelSimulation
from src.visualizations import plot_speedup, plot_partition_analysis

VEHICLE_COUNTS = [250, 500, 750, 1000, 1250, 1500]
RUNS = 2


def bench_par(net, partitioner, light_offsets, specs):
    times = []
    for _ in range(RUNS):
        m = ParallelSimulation(net, partitioner, specs, light_offsets).run()
        times.append(m["wall_time"])
    return sum(times) / len(times)


def partition_analysis(net):
    configs = []
    for nx_s, ny_s in [(1, 1), (1, 2), (2, 2), (2, 3), (3, 3)]:
        if nx_s == 1 and ny_s == 1:
            cross, label = 0, "No split\n(sequential)"
        else:
            p = SectorPartitioner(net, num_x=nx_s, num_y=ny_s)
            cross = p.cross_sector_edges()
            label = f"{nx_s}×{ny_s} Grid" + ("\n(chosen)" if nx_s == 2 and ny_s == 2 else "")
        configs.append((label, nx_s, ny_s, cross))
    return configs


def main():
    print("=" * 60)
    print("  Traffic Flow Simulation – Benchmark")
    print("=" * 60)

    net = CityNetwork(grid_size=config.GRID_SIZE, seed=config.SEED)
    partitioner = SectorPartitioner(net, num_x=config.NUM_SECTORS_X, num_y=config.NUM_SECTORS_Y)
    rep = partitioner.partition_report()

    print(f"\nNetwork : {net.num_nodes} nodes, {net.num_edges} edges")
    print(f"Sectors : {rep['num_sectors']}  | boundary nodes: {rep['boundary_nodes']}"
          f"  | cross-sector edges: {rep['cross_sector_edges']}")

    print("\n[1/3] Partition analysis ...")
    plot_partition_analysis(partition_analysis(net))

    print(f"\n[2/3] Speedup benchmark ({len(VEHICLE_COUNTS)} configs × {RUNS} runs each) ...")
    seq_times, par_times = [], []

    for nv in VEHICLE_COUNTS:
        print(f"  n={nv:5d}", end="", flush=True)
        seq_result = SequentialSimulation(net, num_vehicles=nv).run()
        t_seq = seq_result["wall_time"]
        print(f"  seq={t_seq:.2f}s", end="", flush=True)
        seq_times.append(t_seq)

        t_par = bench_par(net, partitioner,
                          seq_result["light_offsets"],
                          seq_result["vehicle_specs"])
        print(f"  par={t_par:.2f}s  speedup={t_seq/t_par:.2f}×")
        par_times.append(t_par)

    plot_speedup(VEHICLE_COUNTS, seq_times, par_times, partitioner.num_sectors)

    os.makedirs(config.DATA_DIR, exist_ok=True)
    results = {
        "vehicle_counts": VEHICLE_COUNTS,
        "seq_times": seq_times,
        "par_times": par_times,
        "speedups": [s / p for s, p in zip(seq_times, par_times)],
        "num_sectors": partitioner.num_sectors,
    }
    path = os.path.join(config.DATA_DIR, "benchmark_results.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  [saved] {path}")

    print("\n[3/3] Done.  Plots in", config.PLOTS_DIR)
    print("  Speedups:", [f"{s/p:.2f}×" for s, p in zip(seq_times, par_times)])


if __name__ == "__main__":
    main()
