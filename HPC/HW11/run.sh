# I have made this script for classroom machines
# And I have tested it on the class13 machine.

module load hpc_sdk/nvhpc-hpcx-cuda12/24.3

make clean
make

echo "Running the K-Means CUDA program..."

# Run the program with k = 7
echo "5" | ./bin/test.x
#     ^
#     | Change this value to change the number of clusters