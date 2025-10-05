import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.utils import resample
import joblib
import warnings
warnings.filterwarnings("ignore")

# ======================================================
# 1️⃣ Load Data
# ======================================================
print("📂 Loading microlensing dataset...")
df = pd.read_csv("D:\exoplanet_ai\data\microlensing.csv", comment="#", low_memory=False)
print("✅ Loaded", len(df), "rows")

# ======================================================
# 2️⃣ Define Labels
# ======================================================
if "ml_modeldef" in df.columns:
    df["planet_flag"] = df["ml_modeldef"].apply(lambda x: 1 if x == 1 else 0)
elif "ml_massratio" in df.columns:
    df["planet_flag"] = df["ml_massratio"].apply(lambda x: 1 if x < 0.1 else 0)
else:
    raise ValueError("No valid label column found ('ml_modeldef' or 'ml_massratio')")

print(df["planet_flag"].value_counts())

# ======================================================
# 3️⃣ Select Useful Features
# ======================================================
base_features = [
    "ml_t0", "ml_te", "ml_u0", "ml_q", "ml_s", "ml_re",
    "ml_massratio", "ml_xtimeein", "ml_xtimesrc", "ml_tsepmin"
]
features = [f for f in base_features if f in df.columns]
print("Using base features:", features)

# ======================================================
# 4️⃣ Derived Physics Features (log & geometric)
# ======================================================
for col in ["ml_te", "ml_u0", "ml_massratio", "ml_s", "ml_re"]:
    if col in df.columns:
        df[f"log_{col}"] = np.log1p(df[col].abs())  # avoid negatives
        features.append(f"log_{col}")

if "ml_u0" in df.columns:
    df["u0_sq"] = df["ml_u0"] ** 2
    features.append("u0_sq")

if "ml_s" in df.columns:
    df["s_inv"] = 1 / (df["ml_s"] + 1e-6)
    features.append("s_inv")

features = list(dict.fromkeys(features))  # remove duplicates

# ======================================================
# 5️⃣ Drop Missing + Balance Classes
# ======================================================
df = df[features + ["planet_flag"]].dropna()
major = df[df.planet_flag == 0]
minor = df[df.planet_flag == 1]

print(f"Before balance: 0={len(major)}, 1={len(minor)}")
minor_up = resample(minor, replace=True, n_samples=len(major), random_state=42)
df_bal = pd.concat([major, minor_up]).sample(frac=1, random_state=42)
print(f"After balance: 0={sum(df_bal.planet_flag==0)}, 1={sum(df_bal.planet_flag==1)}")

X = df_bal[features]
y = df_bal["planet_flag"]

# ======================================================
# 6️⃣ Train/Test Split
# ======================================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Train={len(X_train)}  Test={len(X_test)}")

# ======================================================
# 7️⃣ Build Models
# ======================================================
rf = RandomForestClassifier(
    n_estimators=400,
    max_depth=10,
    min_samples_split=4,
    min_samples_leaf=2,
    class_weight="balanced_subsample",
    random_state=42
)

xgb = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42
)

ensemble = VotingClassifier(estimators=[("rf", rf), ("xgb", xgb)], voting="soft")

# ======================================================
# 8️⃣ Cross-Validation
# ======================================================
print("🔍 Performing 5-fold cross-validation (F1 score)...")
f1_scores = cross_val_score(ensemble, X, y, cv=5, scoring="f1")
print(f"Cross-validation mean F1 = {f1_scores.mean():.3f} ± {f1_scores.std():.3f}")

# ======================================================
# 9️⃣ Train Final Ensemble
# ======================================================
print("🧠 Training ensemble model...")
ensemble.fit(X_train, y_train)

# ======================================================
# 🔟 Evaluate on Test Set
# ======================================================
y_pred = ensemble.predict(X_test)
print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred, digits=3))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="viridis")
plt.title("Confusion Matrix – Microlensing Planet Detection")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# ======================================================
# 11️⃣ Feature Importance (from RF)
# ======================================================
rf.fit(X_train, y_train)
fi = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
plt.figure(figsize=(8, 5))
sns.barplot(x=fi.values[:15], y=fi.index[:15])
plt.title("Top Feature Importances (RandomForest)")
plt.tight_layout()
plt.show()

# ======================================================
# 12️⃣ Save Model
# ======================================================
os.makedirs("models", exist_ok=True)
joblib.dump(ensemble, "models/microlens_planet_boosted.pkl")
print("✅ Model saved to models/microlens_planet_boosted.pkl")

print("\n🌠 Microlensing Planet Classifier training complete!")
