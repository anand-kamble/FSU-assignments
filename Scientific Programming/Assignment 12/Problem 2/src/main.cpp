#include "stdio.h"
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <random>
#include <numeric>
#include <algorithm>
#include <omp.h>

#include "utils/timer_test.cpp"
#include "utils/table.cpp"

using namespace std;

/**
 * @brief Generates a vector of random integers.
 * @param random_numbers Vector to store the generated random integers.
 * @param n Number of random integers to generate.
 */
void random_ints(std::vector<int> &random_numbers, long n = 1000000)
{
    const int max_value = 255;

    // Step 1: Create a weighted distribution
    std::vector<int> weighted_distribution;
    for (int i = 0; i <= max_value; i++)
    {
        weighted_distribution.insert(weighted_distribution.end(), i, i);
    }

    // Step 2: Randomly sample from the distribution
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, weighted_distribution.size() - 1);

    for (long i = 0; i < n; i++)
    {
        int num = weighted_distribution[dis(gen)];
        random_numbers.push_back(num);
    }
}

/**
 * @brief Main function for parallel counting of occurrences of random integers.
 */
int main()
{
    int i;
    long n = 100000000;
    std::vector<unsigned char> s(n);
    std::vector<int> random_numbers;

    // Generate random integers between 0 and 255
    random_ints(random_numbers, n);

    // Print the first 10 random numbers
    for (int i = 0; i < 10; i++)
    {
        printf("random_ints[%d]: %d\n", i, random_numbers[i]);
    }

    // Allocate an array to count occurrences of each random number
    alignas(64) int *num = new int[256];

    // Initialize the count array
    std::fill(num, num + 256, 0);

    Timer t;
    Table resultTable({"No. of threads", "Time taken", "Speedup"});

    double time_for_single_thread = 1;

    // Perform parallel counting for different numbers of threads
    for (int numOfThreads = 1; numOfThreads < 12; numOfThreads *= 2)
    {
        omp_set_num_threads(numOfThreads);
        t.reset();

        // Parallelize the counting using OpenMP
#pragma omp parallel for private(i) shared(random_numbers) reduction(+ : num[:256])
        for (i = 0; i < n; i++)
        {
            num[random_numbers[i]]++;
        }

        double timeTaken = t.elapsed();

        // Record the results in the result table
        if (numOfThreads == 1)
            time_for_single_thread = timeTaken;
        resultTable.addRow({to_string(numOfThreads), to_string(t.elapsed()), to_string(time_for_single_thread / timeTaken)});
    }

    // Print the result table
    resultTable.printTable();

    // Deallocate the count array
    delete[] num;

    return 0;
}
