import pandas as pd
import matplotlib.pyplot as plt

# Load CSV
df = pd.read_csv("/Users/srikrishnans/Projects/Local/dist-matrix-multiplication-mpi/benchmarks/results.csv")

# Separate serial and mpi
serial_time = df[df['type'] == 'serial']['time'].values[0]
mpi_df = df[df['type'] == 'mpi'].copy()

# Calculate speedup
mpi_df['speedup'] = serial_time / mpi_df['time']
mpi_df['efficiency'] = mpi_df['speedup'] / mpi_df['processes']

# Plotting
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(mpi_df['processes'], mpi_df['speedup'], marker='o')
plt.title("Speedup")
plt.xlabel("Processes")
plt.ylabel("Speedup")

plt.subplot(1, 2, 2)
plt.plot(mpi_df['processes'], mpi_df['efficiency'], marker='o', color='orange')
plt.title("Efficiency")
plt.xlabel("Processes")
plt.ylabel("Efficiency")

plt.tight_layout()
plt.savefig("/Users/srikrishnans/Projects/Local/dist-matrix-multiplication-mpi/benchmarks/performance.png")
plt.show()


