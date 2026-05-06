GRID_SIZE = 35          # 35x35 = 1225 intersections (heavier computation)
NUM_VEHICLES = 1000
SIM_DURATION = 120      # simulated minutes
SEED = 42

NUM_SECTORS_X = 2
NUM_SECTORS_Y = 2
SYNC_INTERVAL = 10      # minutes between boundary-load sync points

RUSH_HOUR_START = 40
RUSH_HOUR_END = 80
RUSH_HOUR_FRACTION = 0.70  # 70 % of vehicles during rush hour

INTERSECTION_CAPACITY = 4  # max concurrent vehicles per node
TRAFFIC_LIGHT_GREEN = 1.5  # minutes
TRAFFIC_LIGHT_RED = 1.5    # minutes

BPR_ALPHA = 0.15
BPR_BETA = 4

REROUTE_INTERVAL = 4    # recalculate shortest path every N hops (makes computation heavy)

OUTPUT_DIR = "output"
PLOTS_DIR = "output/plots"
DATA_DIR = "output/data"
