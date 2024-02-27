#!/bin/bash

# MPI Program
MPI_PROGRAM="./ColorToGray.exe"

# Output file to store benchmark results
OUTPUT_FILE="benchmark_results.txt"

# Number of repetitions for each configuration
REPETITIONS=1

# List of number of processes to benchmark
NUM_PROCS_LIST=(2 4 8 16 32)

# Function to run the MPI program with given number of processes
run_mpi_program() {
    local num_procs=$1
    local output_file=$2
    local repetitions=$3
    local command="mpirun -n $num_procs $MPI_PROGRAM"

    # Run the program for specified repetitions and record execution time
    for ((i=1; i<=$repetitions; i++)); do
        echo "Running with $num_procs processes, repetition $i..."
        echo "-------------------" >> $output_file
        echo "Time for $num_procs : " >> $output_file
        { $command; } 2>&1  | grep "Time"| awk '{print $3}'s >> $output_file
    done
}

# Clear output file
echo -n > $OUTPUT_FILE

# Run the MPI program for each number of processes in the list
for num_procs in "${NUM_PROCS_LIST[@]}"; do
    echo "Benchmarking with $num_procs processes..."
    run_mpi_program $num_procs $OUTPUT_FILE $REPETITIONS
done

echo "Benchmarking complete. Results saved in $OUTPUT_FILE"
