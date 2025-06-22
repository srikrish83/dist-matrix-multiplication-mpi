#!/bin/bash
echo "Generating input matrices "
python3 /Users/srikrishnans/Projects/Local/dist-matrix-multiplication-mpi/input/generate_input.py $1
echo "*******************************************************"

echo "type,processes,time" > /Users/srikrishnans/Projects/Local/dist-matrix-multiplication-mpi/benchmarks/results.csv
echo "Running Serial Version"
python3 /Users/srikrishnans/Projects/Local/dist-matrix-multiplication-mpi/serial/serial_matrix_multiplication.py
echo "*******************************************************"

echo "Running MPI Version"
mpirun -np 2 python3 /Users/srikrishnans/Projects/Local/dist-matrix-multiplication-mpi/mpi/mpi_matrix_multiplication.py
echo "*******************************************************"
mpirun -np 4 python3 /Users/srikrishnans/Projects/Local/dist-matrix-multiplication-mpi/mpi/mpi_matrix_multiplication.py
echo "*******************************************************"
mpirun -np 8 python3 /Users/srikrishnans/Projects/Local/dist-matrix-multiplication-mpi/mpi/mpi_matrix_multiplication.py
echo "*******************************************************"

echo "Running Matplotlib"
python /Users/srikrishnans/Projects/Local/dist-matrix-multiplication-mpi/benchmarks/performance_plot.py
python /Users/srikrishnans/Projects/Local/dist-matrix-multiplication-mpi/benchmarks/performance_detailed.py


