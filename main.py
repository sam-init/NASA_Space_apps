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

# ======================
# 1. Safe Downloader (with retries)
# ======================
def safe_read_csv(url, retries=3, delay=5):
    """Safely read CSV from NASA API with auto-retry."""
    for attempt in range(retries):
        try:
            print(f"⬇️  Attempt {attempt + 1}: Fetching from {url.split('from+')[1].split('&')[0]} ...")
            df = pd.read_csv(url, low_memory=False)
            print(f"✅  Success: {len(df)} rows loaded.")
            return df
        except Exception as e:
            print(f"⚠️  Failed (Attempt {attempt + 1}): {e}")
            if attempt < retries - 1:
                print(f"🔁 Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                raise RuntimeError(f"❌ Could not load data after {retries} attempts.")



# safe stratified split
unique_labels = meta["label"].value_counts()
if unique_labels.min() < 2:
    print(f"[WARN] Too few samples in one class ({unique_labels.to_dict()}). Disabling stratify.")
    stratify = None
else:
    stratify = meta["label"]

train_df, test_df = train_test_split(meta, test_size=0.2, random_state=RANDOM_SEED, stratify=stratify)
train_df, val_df = train_test_split(train_df, test_size=0.15, random_state=RANDOM_SEED,
                                   stratify=(train_df["label"] if stratify is not None else None))

# ======================
# 2. NASA API URLs (fixed + lightweight)
# ======================
base_url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query="

# Cumulative exoplanet data (confirmed planets)
cumulative_url = (
    base_url
    + "select+pl_orbper,pl_rade,pl_bmasse,st_teff,st_rad,st_mass,discoverymethod"
    "+from+pscomppars&format=csv"
)

# Kepler/K2 mission data
k2_url = (
    base_url
    + "select+pl_orbper,pl_rade,pl_bmasse,st_teff,st_rad,st_mass,discoverymethod"
    "+from+k2pandc&format=csv"
)

# TESS Objects of Interest (TOI) — fixed columns
toi_url = (
    base_url
    + "select+TIC_ID,Period,Planet_Radius,Stellar_Teff,Stellar_Radius,Stellar_Mass,TFOPWG_Disposition"
    "+from+toi&format=csv"
)


# ======================
# 3. Cached Loader
# ======================
def load_or_cache(url, filename):
    """Load data from cache if available, else download and save."""
    path = os.path.join("data", filename)
    os.makedirs("data", exist_ok=True)
    if os.path.exists(path):
        print(f"🔁 Using cached file: {filename}")
        return pd.read_csv(path)
    print(f"🌌 Downloading {filename} from NASA...")
    df = safe_read_csv(url)
    df.to_csv(path, index=False)
    print(f"💾 Saved a local copy: {path}")
    return df


# ======================
# 4. Load Datasets
# ======================
print("🌌 Downloading NASA datasets (Cumulative, K2, TOI)...")

cumulative = load_or_cache(cumulative_url, "cumulative.csv")
k2 = load_or_cache(k2_url, "k2.csv")
toi = load_or_cache(toi_url, "toi.csv")

print(f"\n✅ All datasets loaded successfully!")
print(f"Cumulative planets: {len(cumulative)}, K2: {len(k2)}, TOI: {len(toi)}\n")


# ======================
# 5. Prepare Combined Dataset (for Training)
# ======================
features = ["pl_orbper", "pl_rade", "pl_bmasse", "st_teff", "st_rad", "st_mass"]
target = "discoverymethod"

# Combine cumulative + K2 data
data = pd.concat([cumulative, k2], ignore_index=True)
data = data[features + [target]].dropna()

# Encode target labels
le = LabelEncoder()
y = le.fit_transform(data[target])
X = data[features]

# ======================
# 6. Split & Normalize
# ======================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ======================
# 7. Train Models
# ======================
print("🧠 Training Random Forest model...")
rf = RandomForestClassifier(n_estimators=150, random_state=42)
rf.fit(X_train, y_train)

print("🧠 Training XGBoost model...")
xgb = XGBClassifier(eval_metric="logloss", random_state=42)
xgb.fit(X_train, y_train)

# ======================
# 8. Evaluate Performance
# ======================
print("\n📊 Random Forest Performance:")
print(classification_report(y_test, rf.predict(X_test)))

print("\n📊 XGBoost Performance:")
print(classification_report(y_test, xgb.predict(X_test)))


# ======================
# 9. Feature Importance Visualization
# ======================
os.makedirs("results", exist_ok=True)
importances = rf.feature_importances_

plt.figure(figsize=(8, 5))
sns.barplot(x=importances, y=features)
plt.title("Feature Importance (Random Forest)")
plt.tight_layout()
plt.savefig("results/feature_importance.png")
plt.show()


# ======================
# 10. Save Trained Model
# ======================
os.makedirs("models", exist_ok=True)
joblib.dump(rf, "models/exoplanet_model.pkl")


# ======================
# 11. Predict on TOI Dataset
# ======================
print("\n🔭 Predicting on TOI dataset...")

# Match TOI columns to training features as closely as possible
toi_features = ["Period", "Planet_Radius", "Stellar_Teff", "Stellar_Radius", "Stellar_Mass"]
toi_data = toi[toi_features].dropna()

# Scale (pad missing columns to match model input size)
toi_scaled = scaler.transform(
    pd.DataFrame(
        {
            "pl_orbper": toi["Period"].fillna(toi["Period"].mean()),
            "pl_rade": toi["Planet_Radius"].fillna(toi["Planet_Radius"].mean()),
            "pl_bmasse": np.full(len(toi), data["pl_bmasse"].mean()),  # placeholder mean mass
            "st_teff": toi["Stellar_Teff"].fillna(toi["Stellar_Teff"].mean()),
            "st_rad": toi["Stellar_Radius"].fillna(toi["Stellar_Radius"].mean()),
            "st_mass": toi["Stellar_Mass"].fillna(toi["Stellar_Mass"].mean()),
        }
    )
)

toi_pred = rf.predict(toi_scaled)

toi_results = toi.loc[toi_data.index].copy()
toi_results["Predicted_Method"] = le.inverse_transform(toi_pred)

os.makedirs("results", exist_ok=True)
toi_results.to_csv("results/predictions.csv", index=False)

print("\n✅ Predictions saved to results/predictions.csv")
print("🌠 Exoplanet AI Pipeline Completed Successfully!\n")
