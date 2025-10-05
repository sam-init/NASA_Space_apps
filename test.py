import pandas as pd
df = pd.read_csv(r"D:\exoplanet_ai\processed\metadata_labeled.csv")
print(df['label'].value_counts())
