import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
from scipy.signal import find_peaks
import os
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# ======================================================
# 1️⃣ Load Pretrained Model
# ======================================================
MODEL_PATH = "models/microlens_planet_boosted.pkl"
print(f"🔭 Loading model from {MODEL_PATH} ...")
model = joblib.load(MODEL_PATH)
print("✅ Model loaded successfully!\n")

# ======================================================
# 2️⃣ Load OGLE photometry file (TXT or DAT)
# ======================================================
# Example: phot.dat or any OGLE light curve
FILE_PATH = FILE_PATH = r"C:\Users\Asus\Downloads\microlensing.txt"
  # <== change this to your file name

if not os.path.exists(FILE_PATH):
    raise FileNotFoundError(f"❌ File not found: {FILE_PATH}")

print(f"📂 Reading photometry data from: {FILE_PATH}")

# Try to read flexibly
try:
    df = pd.read_csv(FILE_PATH, sep=r"\s+", comment="#", header=None, engine="python")
except Exception as e:
    raise RuntimeError(f"❌ Could not read file: {e}")

# Keep only first 3 columns (OGLE format: HJD Mag Err)
df = df.iloc[:, :3]
df.columns = ["HJD", "Mag", "Err"]

# Clean numeric
df = df.apply(pd.to_numeric, errors="coerce")

# Remove rows where Mag or HJD are missing or zero
df = df.dropna(subset=["HJD", "Mag", "Err"])
df = df[(df["Mag"] > 0) & (df["Err"] > 0)]

print(f"✅ Loaded {len(df)} observations")

if len(df) < 10:
    raise ValueError("❌ File seems empty or malformed — please check OGLE file format.")

# ======================================================
# 3️⃣ Plot raw light curve
# ======================================================
plt.figure(figsize=(8, 4))
plt.errorbar(df["HJD"], df["Mag"], yerr=df["Err"], fmt=".", color="black", ecolor="gray", alpha=0.7)
plt.gca().invert_yaxis()
plt.title("OGLE Light Curve")
plt.xlabel("HJD")
plt.ylabel("Magnitude (I-band)")
plt.tight_layout()
plt.show()

# ======================================================
# 4️⃣ Compute Features from Light Curve
# ======================================================
time = df["HJD"].values
mag = df["Mag"].values
err = df["Err"].values

# Normalize
time = time - np.min(time)
mag_norm = (mag - np.mean(mag)) / np.std(mag)

# Find peaks and minima (brightening)
inv_flux = -mag_norm
peaks, _ = find_peaks(inv_flux, height=np.percentile(inv_flux, 90))
num_peaks = len(peaks)

# Light curve statistics
features = {
    "ml_te": np.ptp(time) / (num_peaks + 1e-3),  # proxy for Einstein timescale
    "ml_u0": np.min(mag_norm) * -1,              # proxy for minimum brightness (impact)
    "ml_s": np.std(mag_norm),                    # proxy for variation scale
    "ml_q": np.var(mag_norm),                    # proxy for shape strength
    "ml_re": np.max(inv_flux) - np.min(inv_flux),# proxy for relative magnification
}

# Derived science features
features["log_ml_te"] = np.log1p(features["ml_te"])
features["log_ml_u0"] = np.log1p(features["ml_u0"])
features["log_ml_s"] = np.log1p(features["ml_s"])
features["log_ml_q"] = np.log1p(features["ml_q"])
features["log_ml_re"] = np.log1p(features["ml_re"])
features["u0_sq"] = features["ml_u0"] ** 2
features["s_inv"] = 1 / (features["ml_s"] + 1e-6)

# Convert to DataFrame
input_df = pd.DataFrame([features])

print("\n🧮 Extracted light curve features:")
print(pd.Series(features).round(4))

# ======================================================
# 5️⃣ Predict Planet Probability
# ======================================================
pred_prob = model.predict_proba(input_df)[0][1]
pred_class = int(model.predict(input_df)[0])

print("\n🪐 Prediction Results:")
print(f"→ Probability of Planet: {pred_prob*100:.2f}%")
print(f"→ Final Prediction: {'🌍 Planet Detected!' if pred_class == 1 else '⭐ No Planet Detected.'}")

# ======================================================
# 6️⃣ Save features for record
# ======================================================
os.makedirs("results", exist_ok=True)
input_df.to_csv("results/last_features.csv", index=False)
print("\n📄 Saved extracted features to results/last_features.csv")
