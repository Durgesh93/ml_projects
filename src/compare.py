import pandas as pd

# Load CSVs
df_ensemble = pd.read_csv("eval_ensemble.csv")
df_pond     = pd.read_csv("eval_pond.csv")
df_roboflow = pd.read_csv("eval_roboflow.csv")

# Add model column
df_ensemble["Model"] = "Ensemble"
df_pond["Model"]     = "Pond"
df_roboflow["Model"] = "Roboflow"

# Combine all results
df_all = pd.concat([df_ensemble, df_pond, df_roboflow], ignore_index=True)

# Build four separate pivot tables
df_precision = df_all.pivot(index="Dataset", columns="Model", values="Precision").reset_index()
df_recall    = df_all.pivot(index="Dataset", columns="Model", values="Recall").reset_index()
df_map50     = df_all.pivot(index="Dataset", columns="Model", values="mAP50").reset_index()
df_map5095   = df_all.pivot(index="Dataset", columns="Model", values="mAP50-95").reset_index()

# Print all four
print("\n📊 Precision Comparison:\n", df_precision)
print("\n📊 Recall Comparison:\n", df_recall)
print("\n📊 mAP50 Comparison:\n", df_map50)
print("\n📊 mAP50-95 Comparison:\n", df_map5095)

