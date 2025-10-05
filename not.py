import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import urllib.request, ssl, os

# ===========================
# CONFIGURATION
# ===========================
BASE_URL = "https://www.astrouw.edu.pl/ogle/ogle4/ews/2024"
EVENT_RANGE = range(1, 11)   # BLG-0001 to BLG-0010
OUTPUT_DIR = r"D:\exoplanet_ai\processed"
SAMPLES = 256
PLOT_EXAMPLES = True  # set False to skip plotting
# ===========================

# disable SSL verification (OGLE site has unverified cert)
ssl._create_default_https_context = ssl._create_unverified_context

os.makedirs(OUTPUT_DIR, exist_ok=True)
metadata = []

def process_event(event_id):
    """Download, clean, normalize, resample, and save one event"""
    event_name = f"BLG-{event_id:04d}"
    url = f"{BASE_URL}/blg-{event_id:04d}/phot.dat"
    print(f"\n📥 Processing {event_name}...")

    try:
        # --- download ---
        tmp_file = os.path.join(OUTPUT_DIR, f"{event_name}.dat")
        urllib.request.urlretrieve(url, tmp_file)
        df = pd.read_csv(tmp_file, delim_whitespace=True, comment="#", header=None)
    except Exception as e:
        print(f"❌ Failed to download {event_name}: {e}")
        return None

    # --- basic clean ---
    df = df.iloc[:, :3]
    df.columns = ["HJD", "Mag", "Err"]
    if len(df) < 20:
        print(f"⚠️ {event_name} too few data points, skipping.")
        return None

    # --- convert magnitudes to flux-like, normalize ---
    time = df["HJD"].values
    mag = df["Mag"].values
    err = df["Err"].values

    time_centered = time - np.mean(time)
    flux_like = -mag
    std = np.std(flux_like)
    flux_norm = (flux_like - np.mean(flux_like)) / (std if std > 1e-6 else 1e-6)
    err_norm = err / np.std(flux_like)

    # --- resample to fixed grid ---
    time_grid = np.linspace(time_centered.min(), time_centered.max(), SAMPLES)
    interp_flux = interp1d(time_centered, flux_norm, kind="linear", fill_value="extrapolate")
    interp_err = interp1d(time_centered, err_norm, kind="linear", fill_value="extrapolate")
    flux_resampled = interp_flux(time_grid)
    err_resampled = interp_err(time_grid)

    # --- save ---
    x = np.stack([flux_resampled, err_resampled], axis=1)
    npy_path = os.path.join(OUTPUT_DIR, f"{event_name}.npy")
    np.save(npy_path, x)
    metadata.append({"file": npy_path, "event": event_name, "points": len(df)})

    print(f"✅ Saved {event_name} → {npy_path}")

    # --- optional plot ---
    if PLOT_EXAMPLES:
        plt.figure(figsize=(7,5))
        plt.plot(time_grid, flux_resampled, color="black")
        plt.xlabel("Centered Time")
        plt.ylabel("Normalized Brightness")
        plt.title(f"{event_name} (Normalized for CNN)")
        plt.tight_layout()
        plt.show()

    return x


# ---------- MAIN EXECUTION ----------
if __name__ == "__main__":
    print("🌌 Starting OGLE batch processor...")
    for eid in EVENT_RANGE:
        process_event(eid)

    # Save metadata
    meta_path = os.path.join(OUTPUT_DIR, "metadata.csv")
    pd.DataFrame(metadata).to_csv(meta_path, index=False)
    print(f"\n📄 Metadata saved to: {meta_path}")
    print("✨ Done! All light curves processed and CNN-ready.")
