"""
Preprocess raw NYC TLC Yellow Taxi trip records.

Produces:
  1. Demand tensor: (T, 200, 2) — pickup/dropoff counts per 30min per grid cell
  2. OD flow matrix: (200, 200) — average origin-destination flows between grid cells
  3. Hourly OD matrices: (48, 200, 200) — OD flows by half-hour slot

The OD matrix is used by Flow Gating module to dynamically modulate
spatial information propagation based on real traffic flow patterns.

Usage:
    python preprocess_raw.py
    python preprocess_raw.py --input data/raw/yellow_tripdata_2015-01.csv
    python preprocess_raw.py --input data/raw/yellow_tripdata_2015-01.parquet

Download data:
    https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
    Get: Yellow Taxi Trip Records, 2015-01 (CSV or Parquet)
    Place in: data/raw/
"""
import sys, os, argparse, glob
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from lib.utils import get_project_path, ensure_dir

# ============================================================
# Grid configuration (must match ST-SSL preprocessing)
# ============================================================
# 10 rows x 20 columns covering Manhattan and surrounding areas
GRID_ROWS = 10
GRID_COLS = 20
NUM_NODES = GRID_ROWS * GRID_COLS  # 200

# Approximate bounding box for the 10x20 grid
# These values are consistent with STDN/ST-SSL papers
LAT_MIN = 40.60
LAT_MAX = 40.90
LON_MIN = -74.06
LON_MAX = -73.77

TIME_INTERVAL = 30  # minutes


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--input', default=None,
                   help='Path to raw CSV/Parquet file. Auto-detects if not set.')
    p.add_argument('--output_dir', default='data/processed/NYCTaxi_OD')
    return p.parse_args()


def find_raw_file():
    """Auto-detect raw data file in data/raw/."""
    raw_dir = get_project_path('data', 'raw')
    patterns = ['*.csv', '*.parquet', '*.CSV', '*.PARQUET']
    for pat in patterns:
        files = glob.glob(os.path.join(raw_dir, pat))
        if files:
            return files[0]
    return None


def load_trips(path: str) -> pd.DataFrame:
    """Load trip records from CSV or Parquet."""
    print(f"Loading: {path}")
    ext = os.path.splitext(path)[1].lower()

    if ext == '.parquet':
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, low_memory=False)

    print(f"  Raw rows: {len(df):,}")

    # Standardize column names (NYC TLC changed formats over the years)
    col_map = {}
    for col in df.columns:
        cl = col.strip().lower()
        if 'pickup' in cl and ('time' in cl or 'datetime' in cl):
            col_map[col] = 'pickup_datetime'
        elif 'dropoff' in cl and ('time' in cl or 'datetime' in cl):
            col_map[col] = 'dropoff_datetime'
        elif 'pickup' in cl and ('lon' in cl or 'lng' in cl):
            col_map[col] = 'pickup_longitude'
        elif 'pickup' in cl and 'lat' in cl:
            col_map[col] = 'pickup_latitude'
        elif 'dropoff' in cl and ('lon' in cl or 'lng' in cl):
            col_map[col] = 'dropoff_longitude'
        elif 'dropoff' in cl and 'lat' in cl:
            col_map[col] = 'dropoff_latitude'

    df = df.rename(columns=col_map)

    required = ['pickup_datetime', 'pickup_longitude', 'pickup_latitude',
                'dropoff_longitude', 'dropoff_latitude']
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"  [WARNING] Missing columns: {missing}")
        print(f"  Available columns: {list(df.columns)}")
        raise ValueError("Cannot find required columns in raw data.")

    # Parse datetime
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])

    return df


def coords_to_grid(lat, lon):
    """
    Map (lat, lon) to grid cell index.
    Returns: row, col (or -1, -1 if outside bounds)
    """
    row = ((lat - LAT_MIN) / (LAT_MAX - LAT_MIN) * GRID_ROWS).astype(int)
    col = ((lon - LON_MIN) / (LON_MAX - LON_MIN) * GRID_COLS).astype(int)

    # Clip to valid range
    valid = (row >= 0) & (row < GRID_ROWS) & (col >= 0) & (col < GRID_COLS)
    row = np.where(valid, row, -1)
    col = np.where(valid, col, -1)

    return row, col, valid


def process_trips(df: pd.DataFrame):
    """
    Process trip records into:
      1. Demand tensor: (T, N, 2) — pickup/dropoff counts
      2. OD flow matrix: (N, N) — average flows
      3. Hourly OD: (48, N, N) — flows by half-hour slot
    """
    print("\nMapping coordinates to grid...")

    # Map pickup locations to grid
    p_row, p_col, p_valid = coords_to_grid(
        df['pickup_latitude'].values, df['pickup_longitude'].values)
    d_row, d_col, d_valid = coords_to_grid(
        df['dropoff_latitude'].values, df['dropoff_longitude'].values)

    both_valid = p_valid & d_valid
    print(f"  Valid trips (both in grid): {both_valid.sum():,} / {len(df):,} "
          f"({both_valid.mean()*100:.1f}%)")

    df = df[both_valid].copy()
    p_row, p_col = p_row[both_valid], p_col[both_valid]
    d_row, d_col = d_row[both_valid], d_col[both_valid]

    # Compute node indices
    pickup_node = p_row * GRID_COLS + p_col
    dropoff_node = d_row * GRID_COLS + d_col

    # Time slot index
    dt = df['pickup_datetime']
    time_slot = dt.dt.hour * 2 + dt.dt.minute // TIME_INTERVAL
    half_hour_slot = time_slot.values  # 0-47 within a day

    # --- 1. Demand tensor ---
    print("\nBuilding demand tensor...")
    time_min = dt.min().floor('D')
    time_max = dt.max().ceil('D')
    total_slots = int((time_max - time_min).total_seconds() / (TIME_INTERVAL * 60))
    abs_slot = ((dt - time_min).dt.total_seconds() / (TIME_INTERVAL * 60)).astype(int).values

    demand = np.zeros((total_slots, NUM_NODES, 2), dtype=np.float32)
    for i in range(len(df)):
        t = abs_slot[i]
        if 0 <= t < total_slots:
            demand[t, pickup_node.iloc[i] if hasattr(pickup_node, 'iloc') else pickup_node[i], 0] += 1
            demand[t, dropoff_node.iloc[i] if hasattr(dropoff_node, 'iloc') else dropoff_node[i], 1] += 1

    print(f"  Demand shape: {demand.shape}")
    print(f"  Total pickups: {demand[:,:,0].sum():.0f}")
    print(f"  Total dropoffs: {demand[:,:,1].sum():.0f}")

    # --- 2. Overall OD flow matrix ---
    print("\nBuilding OD flow matrix...")
    od_matrix = np.zeros((NUM_NODES, NUM_NODES), dtype=np.float32)
    for i in range(len(pickup_node)):
        pn = pickup_node.iloc[i] if hasattr(pickup_node, 'iloc') else pickup_node[i]
        dn = dropoff_node.iloc[i] if hasattr(dropoff_node, 'iloc') else dropoff_node[i]
        od_matrix[pn, dn] += 1

    # Normalize: average per time slot
    num_days = max((time_max - time_min).days, 1)
    od_avg = od_matrix / (num_days * 48)  # avg per half-hour
    print(f"  OD matrix: {od_avg.shape}, non-zero: {(od_avg > 0).sum()}")

    # --- 3. Hourly OD matrices (48 slots per day) ---
    print("\nBuilding hourly OD matrices...")
    od_hourly = np.zeros((48, NUM_NODES, NUM_NODES), dtype=np.float32)
    slot_counts = np.zeros(48, dtype=np.float32)

    for i in range(len(pickup_node)):
        pn = pickup_node.iloc[i] if hasattr(pickup_node, 'iloc') else pickup_node[i]
        dn = dropoff_node.iloc[i] if hasattr(dropoff_node, 'iloc') else dropoff_node[i]
        s = half_hour_slot[i]
        od_hourly[s, pn, dn] += 1

    for s in range(48):
        count = max((half_hour_slot == s).sum(), 1)
        od_hourly[s] /= (count / len(df) * num_days)

    print(f"  Hourly OD shape: {od_hourly.shape}")

    return demand, od_avg, od_hourly


def main():
    args = parse_args()

    # Find input file
    if args.input:
        raw_path = args.input
        if not os.path.isabs(raw_path):
            raw_path = get_project_path(raw_path)
    else:
        raw_path = find_raw_file()

    if raw_path is None or not os.path.exists(raw_path):
        print("=" * 60)
        print("Raw data not found!")
        print("=" * 60)
        print("\nPlease download NYC Yellow Taxi trip data:")
        print("  https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page")
        print("\nGet: Yellow Taxi Trip Records, 2015-01")
        print(f"Place in: {get_project_path('data', 'raw')}/")
        print("\nSupported formats: .csv, .parquet")
        return

    # Process
    df = load_trips(raw_path)
    demand, od_avg, od_hourly = process_trips(df)

    # Save
    output_dir = get_project_path(args.output_dir)
    ensure_dir(output_dir)

    np.savez(os.path.join(output_dir, 'demand.npz'), data=demand)
    np.savez(os.path.join(output_dir, 'od_avg.npz'), od=od_avg)
    np.savez(os.path.join(output_dir, 'od_hourly.npz'), od=od_hourly)

    print(f"\n{'='*60}")
    print(f"Saved to {output_dir}/")
    print(f"  demand.npz    — ({demand.shape}) pickup/dropoff counts")
    print(f"  od_avg.npz    — ({od_avg.shape}) average OD flow matrix")
    print(f"  od_hourly.npz — ({od_hourly.shape}) per-half-hour OD matrices")
    print(f"{'='*60}")

    # Quick stats
    print(f"\nGrid: {GRID_ROWS}x{GRID_COLS} = {NUM_NODES} cells")
    print(f"Bounds: lat [{LAT_MIN}, {LAT_MAX}], lon [{LON_MIN}, {LON_MAX}]")
    print(f"Top-5 OD pairs (avg trips per 30min):")
    flat = od_avg.flatten()
    top5 = np.argsort(flat)[-5:][::-1]
    for idx in top5:
        i, j = divmod(idx, NUM_NODES)
        r_i, c_i = divmod(i, GRID_COLS)
        r_j, c_j = divmod(j, GRID_COLS)
        print(f"  ({r_i},{c_i}) -> ({r_j},{c_j}): {flat[idx]:.2f} trips/30min")


if __name__ == '__main__':
    main()
