import numpy as np
import time
import csv

def load_input():
    A = np.load('/Users/srikrishnans/Projects/Local/dist-matrix-multiplication-mpi/input/A.npy')
    B = np.load('/Users/srikrishnans/Projects/Local/dist-matrix-multiplication-mpi/input/B.npy')
    return A, B

def serial_matrix_multiply(A, B):
    return np.dot(A, B)

if __name__ == "__main__":
    A, B = load_input()
    start = time.time()
    C = serial_matrix_multiply(A, B)
    end = time.time()
    print(f"Serial Execution Time: {end - start:.4f} seconds")
    
    with open("/Users/srikrishnans/Projects/Local/dist-matrix-multiplication-mpi/benchmarks/results.csv", "a") as f:
        writer = csv.writer(f)
        writer.writerow(["serial", 1, end - start])

