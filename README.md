# Parallel Simulation of Traffic Flow in Smart Cities

Parallel Programing Project

## Overview

This project models a city road network as a directed graph and simulates concurrent vehicle movement using **SimPy**. The simulation demonstrates measurable congestion under peak (rush-hour) load and achieves parallelism by dividing the network into independently simulated sectors with periodic boundary synchronisation.

**Stack:** Python 3.11+ · SimPy · NetworkX · NumPy · Matplotlib · Seaborn · `multiprocessing`

---

## Features

- **35×35 grid network** — 1,225 nodes, 4,760 directed edges with major arterials and local streets
- **BPR congestion model** — travel time increases dynamically with edge load
- **SimPy agents** — each vehicle is a process with spawning, routing, queueing, and re-routing
- **Traffic lights** — staggered green/red cycles per intersection
- **Rush-hour modelling** — 70% of vehicles spawn in a Gaussian peak window (minutes 40–80)
- **Parallel execution** — 4 OS processes (2×2 grid partition), true parallelism with no GIL
- **Barrier synchronisation** — sectors progress in lockstep every 10 simulated minutes
- **8 output plots** — heatmaps, speedup curves, congestion timelines, and more

---

## Results

### Single run (1,000 vehicles)

| Metric | Sequential | Parallel |
|---|---:|---:|
| Wall-clock time | 2.665 s | 0.917 s |
| **Speedup** | — | **2.91×** |
| Trips completed | 960 | 960 |
| Avg wait / vehicle | 9.56 min | 9.17 min |
| Heatmap similarity | — | 99.78% |

### Speedup curve (benchmark.py)

| Vehicles | Seq (s) | Par (s) | Speedup |
|---:|---:|---:|---:|
| 250 | 0.704 | 0.266 | 2.64× |
| 500 | 1.509 | 0.595 | 2.53× |
| 750 | 1.995 | 0.610 | 3.27× |
| 1000 | 2.721 | 0.931 | 2.92× |
| 1250 | 3.528 | 1.034 | **3.41×** |
| 1500 | 4.322 | 1.310 | 3.30× |

Peak parallel efficiency: **~85%** (theoretical max 4× for 4 sectors).

---

## Project Structure

```
traffic-flow-simulation/
├── config.py              # All simulation parameters
├── main.py                # Full pipeline: build → seq → par → plots
├── benchmark.py           # Speedup measurement (250→1500 vehicles)
├── requirements.txt
├── src/
│   ├── network.py         # CityNetwork: grid graph + BPR model
│   ├── sector.py          # SectorPartitioner: 2×2 spatial partition
│   ├── sequential.py      # SequentialSimulation (single SimPy env)
│   ├── parallel.py        # ParallelSimulation (4 processes + Barrier)
│   └── visualizations.py  # 7 plotting functions → 8 PNG files
└── output/
    ├── plots/             # 8 generated PNGs
    └── data/              # JSON summaries
```

---

## Installation & Usage

```bash
pip install -r requirements.txt

# Full run: sequential + parallel + all plots (1000 vehicles)
python main.py

# Custom vehicle count
python main.py --vehicles 500

# Sequential only
python main.py --no-parallel

# Full speedup benchmark (250 → 1500 vehicles)
python benchmark.py
```

---

## Generated Plots

| File | Description |
|---|---|
| `heatmap_sequential.png` | Traffic-density heatmap (sequential) |
| `heatmap_parallel.png` | Traffic-density heatmap (parallel) |
| `speedup_graph.png` | Wall-time & speedup-ratio vs vehicle count |
| `congestion_timeline.png` | Avg queue length over time, rush-hour highlighted |
| `sector_workload.png` | Per-sector vehicle & node counts |
| `wait_time_distribution.png` | Overlaid wait-time histograms |
| `partition_analysis.png` | Cross-sector edges across 5 partition strategies |
| `summary_table.png` | Sequential vs parallel comparison table |

---

## Architecture

### Parallelism Strategy

The network is split into a **2×2 rectangular partition** (4 sectors). Each sector runs in its own OS process with its own SimPy environment. Vehicles are assigned to the sector of their spawn node and use the full graph for globally optimal routing.

The 2×2 split keeps cross-sector edges at ~140 / 4,760 (**≈ 3%**), balancing synchronisation overhead against load distribution.

### Synchronisation

A `multiprocessing.Barrier(4)` synchronises all sector processes every `SYNC_INTERVAL = 10` simulated minutes — ensuring lockstep progress without passing data across process boundaries.

---

## Key Parameters (`config.py`)

| Parameter | Default | Description |
|---|---|---|
| `GRID_SIZE` | 35 | Network is GRID_SIZE × GRID_SIZE |
| `NUM_VEHICLES` | 1000 | Vehicle count |
| `SIM_DURATION` | 120 min | Total simulated time |
| `SEED` | 42 | Global random seed |
| `NUM_SECTORS_X/Y` | 2 / 2 | Partition shape |
| `SYNC_INTERVAL` | 10 min | Barrier sync cadence |
| `RUSH_HOUR_START/END` | 40 / 80 | Peak window (minutes) |
| `BPR_ALPHA / _BETA` | 0.15 / 4 | BPR congestion curve |
| `REROUTE_INTERVAL` | 4 | Re-route every N hops |
