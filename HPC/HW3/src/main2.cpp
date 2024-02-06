/**
 * @file main.cpp
 * @brief This file contains the main function for a prime number gap calculation using OpenMP.
 *
 * This program calculates the maximum prime gap within a specified range [n, m].
 * It utilizes OpenMP for parallelization to enhance performance for various thread counts.
 *
 * @name Author: Anand Kamble
 * @date Date: 5th Feb 2024
 *
 * @note Make sure to include the necessary dependencies:
 *   - iostream
 *   - cmath
 *   - omp.h
 *
 * @note Functions:
 *   - `bool isPrime(long long int num)`: Checks if a number is prime.
 *   - `int main()`: Main function for prime gap calculation using OpenMP.
 *
 * @note Usage:
 *   - Set the range [n, m] and the desired thread counts in the main function.
 *   - Compile the code with appropriate flags for OpenMP support or use the provided Makefile.
 *   - Adjust the number of threads using the `num_threads` clause in the OpenMP directives.
 *
 * @note Parallelization Details:
 *   - Parallel loop is used for finding the maximum prime gap.
 *   - Custom reduction is employed to efficiently find the maximum prime gap.
 *
 * @warning Ensure proper compilation with OpenMP support for parallel execution.
 */

#include <iostream>
#include <cmath>
#include <omp.h>

// Structure to store information about the maximum prime gap
struct MaxPrimeGap
{
    long long int max;
    long long int index;
};

// Declare a custom reduction for finding the maximum prime gap
#pragma omp declare reduction(primeMax : struct MaxPrimeGap : omp_out = (omp_in.max > omp_out.max) ? omp_in : omp_out)

using namespace std;

// Function to check if a number is prime
bool isPrime(long long int num)
{
    if (num <= 1)
        return false;
    for (long long int i = 2; i <= sqrt(num); ++i)
    {
        if (num % i == 0)
            return false;
    }
    return true;
}

int main()
{
    // Range of numbers to check for prime gaps
    long long int n, m;

    n = 1;
    m = 1000000;

    // Array of thread counts for parallelization
    int threads = 12;

    // Record the start time for benchmarking
    double start = omp_get_wtime();

    // Variable to store information about the maximum prime gap
    MaxPrimeGap max_diff;
    max_diff.max = 0;
    max_diff.index = 0;

// Parallel loop to find the maximum prime gap
#pragma omp parallel for reduction(primeMax : max_diff) num_threads(threads) schedule(dynamic)
    for (int i = n; i < m; i++)
    {
        if (isPrime(i))
        {
            // Loop to find the next prime number and calculate the gap
            for (int j = i + 1; j < m; j++)
            {
                if (isPrime(j))
                {
                    if (j - i > max_diff.max)
                    {
                        max_diff.max = j - i;
                        max_diff.index = i;
                    }
                    break;
                }
            }
        }
    }

    // Record the end time for benchmarking
    double end = omp_get_wtime();

    // Print the execution time for the current thread count
    printf("Time for %d threads : %f\n", threads, end - start);

    // Print information about the maximum prime gap
    std::cout << "For n = " << n << ", m = " << m
              << ", the biggest prime number gap is between " << max_diff.index << " and " << max_diff.index + max_diff.max
              << ", which is " << max_diff.max << "." << std::endl;

    return 0;
}
