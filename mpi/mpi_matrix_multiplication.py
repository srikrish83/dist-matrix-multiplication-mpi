from mpi4py import MPI
import numpy as np
import time
import os
import csv

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    A = np.load("/Users/srikrishnans/Projects/Local/dist-matrix-multiplication-mpi/input/A.npy")
    B_full = np.load("/Users/srikrishnans/Projects/Local/dist-matrix-multiplication-mpi/input/B.npy")
    N = A.shape[0]
else:
    A = None
    B_full = None
    N = None

N = comm.bcast(N, root=0)

local_A = np.empty((N // size, N), dtype=np.int32)
B = np.empty((N, N), dtype=np.int32)
local_C = np.empty((N // size, N), dtype=np.int32)

t0 = MPI.Wtime()
# Scatter rows of A
comm.Scatter(A, local_A, root=0)

# Broadcast B
comm.Bcast(B , root=0)
t1 = MPI.Wtime()

# Local computation
start = MPI.Wtime()
local_C = np.dot(local_A, B)
end = MPI.Wtime()
t2 = MPI.Wtime()

# Gather result matrix
if rank == 0:
    C = np.empty((N, N), dtype=np.int32)
else:
    C = None

comm.Gather(local_C, C, root=0)
t3 = MPI.Wtime()

#if rank == 0:
#    print(f"Distributed Execution Time: {end - start:.4f} seconds")
if rank == 0:
    print(f"Running with {size} MPI processes")
    print(f"Scatter + Broadcast: {t1 - t0:.4f} s")
    print(f"Computation: {t2 - t1:.4f} s")
    print(f"Gather: {t3 - t2:.4f} s")
    print(f"Total Time: {t3 - t0:.4f} s")
    
    with open("/Users/srikrishnans/Projects/Local/dist-matrix-multiplication-mpi/benchmarks/results.csv", "a") as f:
        writer = csv.writer(f)
        writer.writerow(["mpi", size, end - start])
    
    with open("/Users/srikrishnans/Projects/Local/dist-matrix-multiplication-mpi/benchmarks/detailed_results.csv", "a") as f:
        writer = csv.writer(f)
        writer.writerow(["mpi", size,t1 - t0,t2 - t1,t3 - t2,t3 - t0])  

