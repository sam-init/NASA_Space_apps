import pandas as pd
import numpy as np
import os
import time
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

warnings.filterwarnings("ignore")

# =====================================================
# 1. Safe Downloader (with retries + logging)
# =====================================================
def safe_read_csv(url, retries=3, delay=5):
    """Safely read CSV from NASA API with auto-retry."""
    for attempt in range(retries):
        try:
            tag = url.split("from+")[1].split("&")[0]
            print(f"⬇️  Attempt {attempt + 1}: Fetching from {tag} ...")
            df = pd.read_csv(url, low_memory=False)
            print(f"✅  Success: {len(df)} rows loaded from {tag}.")
            return df
        except Exception as e:
            print(f"⚠️  Failed (Attempt {attempt + 1}): {e}")
            if attempt < retries - 1:
                print(f"🔁 Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                raise RuntimeError(f"❌ Could not load data after {retries} attempts.")


# =====================================================
# 2. NASA API URLs
# =====================================================
BASE_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query="

# Confirmed exoplanets
CUMULATIVE_URL = (
    BASE_URL
    + "select+pl_orbper,pl_rade,pl_bmasse,st_teff,st_rad,st_mass,discoverymethod"
    "+from+pscomppars&format=csv"
)

# Kepler/K2 data
K2_URL = (
    BASE_URL
    + "select+pl_orbper,pl_rade,pl_bmasse,st_teff,st_rad,st_mass,discoverymethod"
    "+from+k2pandc&format=csv"
)

# TESS Objects of Interest
TOI_URL = (
    BASE_URL
    + "select+TIC_ID,Period,Planet_Radius,Stellar_Teff,Stellar_Radius,Stellar_Mass,TFOPWG_Disposition"
    "+from+toi&format=csv"
)


# =====================================================
# 3. Cached Loader
# =====================================================
def load_or_cache(url, filename):
    """Load from cache if exists, else download & save."""
    os.makedirs("data", exist_ok=True)
    path = os.path.join("data", filename)
    if os.path.exists(path):
        print(f"🔁 Using cached file: {filename}")
        return pd.read_csv(path)
    print(f"🌌 Downloading {filename} from NASA...")
    df = safe_read_csv(url)
    df.to_csv(path, index=False)
    print(f"💾 Saved a local copy: {path}")
    return df


# =====================================================
# 4. Load NASA Datasets
# =====================================================
print("🌌 Downloading NASA datasets (Cumulative, K2, TOI)...")

cumulative = load_or_cache(CUMULATIVE_URL, "cumulative.csv")
k2 = load_or_cache(K2_URL, "k2.csv")
try:
    toi = load_or_cache(TOI_URL, "toi.csv")
except Exception as e:
    print(f"⚠️  TOI download failed ({e}). Falling back to old column names...")
    alt_url = (
        BASE_URL
        + "select+TIC_ID,Period,Planet_Radius_Earth,Stellar_Eff_Temp,Stellar_Radius,Stellar_Mass,TFOPWG_Disposition"
        "+from+toi&format=csv"
    )
    toi = safe_read_csv(alt_url)
    toi.to_csv("data/toi.csv", index=False)

print("\n✅ All datasets loaded successfully!")
print(f"  Cumulative planets: {len(cumulative)}")
print(f"  K2 planets:         {len(k2)}")
print(f"  TOI candidates:     {len(toi)}\n")


# =====================================================
# 5. Combine and Prepare Training Data
# =====================================================
FEATURES = ["pl_orbper", "pl_rade", "pl_bmasse", "st_teff", "st_rad", "st_mass"]
TARGET = "discoverymethod"

data = pd.concat([cumulative, k2], ignore_index=True)
data = data[FEATURES + [TARGET]].dropna()
print(f"✅ Combined dataset shape: {data.shape}")

# Encode target labels
le = LabelEncoder()
data[TARGET] = le.fit_transform(data[TARGET])
y = data[TARGET].values
X = data[FEATURES].values

# =====================================================
# 6. Safe Stratified Split
# =====================================================
class_counts = np.bincount(y)
if (class_counts < 2).any():
    print(f"[WARN] Some discovery methods have <2 samples — disabling stratify.")
    stratify = None
else:
    stratify = y

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=stratify
)

print(f"📊 Train set: {len(X_train)} | Test set: {len(X_test)}")

# =====================================================
# 7. Normalize
# =====================================================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =====================================================
# 8. Train Models
# =====================================================
print("🧠 Training Random Forest model...")
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)

print("🧠 Training XGBoost model...")
xgb = XGBClassifier(eval_metric="logloss", random_state=42)
xgb.fit(X_train, y_train)

# =====================================================
# 9. Evaluate
# =====================================================
print("\n📊 Random Forest Performance:")
print(classification_report(y_test, rf.predict(X_test), target_names=le.classes_))

print("\n📊 XGBoost Performance:")
print(classification_report(y_test, xgb.predict(X_test), target_names=le.classes_))

# =====================================================
# 10. Feature Importance Visualization
# =====================================================
os.makedirs("results", exist_ok=True)
importances = rf.feature_importances_

plt.figure(figsize=(8, 5))
sns.barplot(x=importances, y=FEATURES)
plt.title("Feature Importance (Random Forest)")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig("results/feature_importance.png")
plt.show()

# =====================================================
# 11. Save Trained Model + Scaler
# =====================================================
os.makedirs("models", exist_ok=True)
joblib.dump(rf, "models/exoplanet_rf.pkl")
joblib.dump(scaler, "models/scaler.pkl")
print("💾 Saved Random Forest model and scaler to /models")

# =====================================================
# 12. Predict on TOI dataset
# =====================================================
print("\n🔭 Predicting on TOI dataset...")

# Match TOI columns → fill missing features
toi_scaled = scaler.transform(
    pd.DataFrame(
        {
            "pl_orbper": toi["Period"].fillna(toi["Period"].mean()),
            "pl_rade": toi["Planet_Radius"].fillna(toi["Planet_Radius"].mean()),
            "pl_bmasse": np.full(len(toi), data["pl_bmasse"].mean()),  # fallback
            "st_teff": toi["Stellar_Teff"].fillna(toi["Stellar_Teff"].mean()),
            "st_rad": toi["Stellar_Radius"].fillna(toi["Stellar_Radius"].mean()),
            "st_mass": toi["Stellar_Mass"].fillna(toi["Stellar_Mass"].mean()),
        }
    )
)

toi_pred = rf.predict(toi_scaled)
toi_results = toi.copy()
toi_results["Predicted_Method"] = le.inverse_transform(toi_pred)

os.makedirs("results", exist_ok=True)
toi_results.to_csv("results/predictions.csv", index=False)

print("\n✅ Predictions saved to results/predictions.csv")
print("🌠 Exoplanet AI Pipeline Completed Successfully!")
