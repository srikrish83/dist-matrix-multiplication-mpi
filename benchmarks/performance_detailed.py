import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("/Users/srikrishnans/Projects/Local/dist-matrix-multiplication-mpi/benchmarks/detailed_results.csv", header=None,
                 names=["Processes", "Scatter_Bcast", "Compute", "Gather", "Total"])

# Create stacked bars
fig, ax = plt.subplots(figsize=(10, 6))

bar1 = ax.bar(df["Processes"], df["Scatter_Bcast"], label="Scatter + Broadcast")
bar2 = ax.bar(df["Processes"], df["Compute"], bottom=df["Scatter_Bcast"], label="Computation")
bar3 = ax.bar(df["Processes"], df["Gather"],
              bottom=df["Scatter_Bcast"] + df["Compute"], label="Gather")

# Add total time as line
ax.plot(df["Processes"], df["Total"], marker="o", color="black", linestyle="--", label="Total Time")

ax.set_xlabel("MPI Processes")
ax.set_ylabel("Time (seconds)")
ax.set_title("Execution Time Breakdown per Phase")
ax.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("/Users/srikrishnans/Projects/Local/dist-matrix-multiplication-mpi/benchmarks/detailed_performance.png")
plt.show()

