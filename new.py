import pandas as pd
import numpy as np

# Paths
MICRO_PATH = r"D:\exoplanet_ai\data\microlensing.csv"
OUT_PATH = r"D:\exoplanet_ai\data\microlensing_labels_planet.csv"

# Read full catalog
df = pd.read_csv(MICRO_PATH, comment='#', low_memory=False)

# Create binary planet labels
if 'ml_modeldef' in df.columns:
    df['planet_flag'] = df['ml_modeldef'].apply(lambda x: 1 if x == 1 else 0)
else:
    # fallback: heuristic using mass ratio
    df['planet_flag'] = df['ml_massratio'].apply(lambda q: 1 if q > 1e-3 else 0)

# Keep minimal useful columns
keep_cols = ['pl_name','ra','dec','ml_modeldef','ml_massratio','planet_flag']
keep_cols = [c for c in keep_cols if c in df.columns]

df_out = df[keep_cols].copy()
df_out.to_csv(OUT_PATH, index=False)
print(f"[SAVED] Planet label dataset → {OUT_PATH}")
print(df_out['planet_flag'].value_counts())
