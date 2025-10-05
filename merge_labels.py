import pandas as pd

# Paths
LABELS_PATH = r"D:\exoplanet_ai\data\microlensing_labels_planet.csv"
META_PATH = r"D:\exoplanet_ai\processed\metadata.csv"
OUT_PATH = r"D:\exoplanet_ai\processed\metadata_labeled.csv"

labels = pd.read_csv(LABELS_PATH)
meta = pd.read_csv(META_PATH)

# Simplify IDs so they can match (e.g., "BLG-0001" ↔ "OGLE-2024-BLG-0001")
meta['event_clean'] = meta['event'].astype(str).str.extract(r'(\d+)', expand=False)
labels['pl_name_clean'] = labels['pl_name'].astype(str).str.extract(r'(\d+)', expand=False)

merged = pd.merge(meta, labels, left_on='event_clean', right_on='pl_name_clean', how='left')
merged['label'] = merged['planet_flag'].fillna(0).astype(int)

merged[['file','event','label']].to_csv(OUT_PATH, index=False)
print(f"[SAVED] Merged labeled metadata → {OUT_PATH}")
print(merged['label'].value_counts())
