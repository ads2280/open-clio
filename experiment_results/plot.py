import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(
    "experiment_results/sonnet3.5-200-14-4/20250623_175800_2d_coordinates.csv"
)

plt.figure(figsize=(12, 8))
scatter = plt.scatter(df["x"], df["y"], c=df["cluster_id"], cmap="tab20", alpha=0.7)
plt.colorbar(scatter, label="Cluster ID")
plt.title("2D Cluster Visualization")
plt.xlabel("UMAP Dimension 1")
plt.ylabel("UMAP Dimension 2")
plt.show()
