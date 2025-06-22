import sys
import numpy as np
import os

size = int(sys.argv[1])
A = np.random.randint(0, 100, (size, size), dtype=np.int32)
B = np.random.randint(0, 100, (size, size), dtype=np.int32)

np.save("/Users/srikrishnans/Projects/Local/dist-matrix-multiplication-mpi/input/A.npy", A)
np.save("/Users/srikrishnans/Projects/Local/dist-matrix-multiplication-mpi/input/B.npy", B)

print(f"Generated random {size}x{size} input matrices")

