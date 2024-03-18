#!/bin/bash

# MPI Program
MPI_PROGRAM="./bin/test.x"

# Output file to store benchmark results
OUTPUT_FILE="benchmark_results.txt"

# Number of repetitions for each configuration
REPETITIONS=1

# List of number of processes to benchmark
# NUM_PROCS_LIST=(1 2 3 4 5 6 7 8 9 10 11 12 13 14)
NUM_PROCS_LIST=(1 2 3 4 5 6)

# Function to run the MPI program with given number of processes
run_mpi_program() {
    local num_procs=$1
    local output_file=$2
    local repetitions=$3
    local command="mpirun -n $num_procs $MPI_PROGRAM"

    # Run the program for specified repetitions and record execution time
    for ((i=1; i<=$repetitions; i++)); do
        echo "-------------------" 
        echo "Time for $num_procs : "
        $command
    done
}

# Clear output file
echo -n > $OUTPUT_FILE

# load the mpi module
module load mpi/openmpi-x86_64

# Run the MPI program for each number of processes in the list
for num_procs in "${NUM_PROCS_LIST[@]}"; do
    run_mpi_program $num_procs $OUTPUT_FILE $REPETITIONS
done

echo "Benchmarking complete. Results saved in $OUTPUT_FILE"
