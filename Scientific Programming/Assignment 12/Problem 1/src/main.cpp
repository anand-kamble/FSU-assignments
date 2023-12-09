#include <iostream>
#include <cmath>
#include <vector>
#include <omp.h>
#include <string>

#include "utils/timer_test.cpp"
#include "utils/table.cpp"

using namespace std;

/**
 * @brief Function to integrate.
 * @param x The value at which to evaluate the function.
 * @return The result of the function evaluation.
 */
double function_to_integrate(double x)
{
    return x * x;
}

/**
 * @brief Simpson's rule for numerical integration.
 * @param a Lower limit of integration.
 * @param b Upper limit of integration.
 * @param n Number of intervals.
 * @return The numerical integration result.
 */
double simpsons_rule(double a, double b, int n)
{
    double h = (b - a) / n;
    double result = function_to_integrate(a) + function_to_integrate(b);

    // Parallelize the summation using OpenMP
#pragma omp parallel for reduction(+ : result)
    for (int i = 1; i < n; i += 2)
    {
        double x_i = a + i * h;
        result += 4 * function_to_integrate(x_i);
    }

#pragma omp parallel for reduction(+ : result)
    for (int i = 2; i < n - 1; i += 2)
    {
        double x_i = a + i * h;
        result += 2 * function_to_integrate(x_i);
    }

    return (h / 3) * result;
}

/**
 * @brief Trapezoidal rule for numerical integration.
 * @param a Lower limit of integration.
 * @param b Upper limit of integration.
 * @param n Number of intervals.
 * @return The numerical integration result.
 */
double trapezoidal_rule(double a, double b, int n)
{
    double h = (b - a) / n;
    double result = (function_to_integrate(a) + function_to_integrate(b)) / 2;

    // Parallelize the summation using OpenMP
#pragma omp parallel for reduction(+ : result)
    for (int i = 1; i < n; ++i)
    {
        double x_i = a + i * h;
        result += function_to_integrate(x_i);
    }

    return h * result;
}

int main()
{
    cout << "Parallelizing the integration using Trapezoidal and Simpson Rules" << endl
         << endl;

    // Example usage
    double a = 0.0;               // Lower limit of integration
    double b = 1.0;               // Upper limit of integration
    int n = (1000 * 1000 * 1000); // Number of intervals

    int NUMBER_OF_THREADS[4] = {1, 2, 4, 8};

    Timer t;
    Table resultTable({"No. of threads", "Time taken", "Speedup"});

    double time_for_single_thread = 1;

    for (size_t i = 0; i < 4; i++)
    {
        // Set the number of threads for OpenMP
        omp_set_num_threads(NUMBER_OF_THREADS[i]);

        // Measure the time taken for both integration methods
        t.reset();
        simpsons_rule(a, b, n);
        trapezoidal_rule(a, b, n);
        double timeTaken = t.elapsed();

        // Record the results in the result table
        if (i == 0)
            time_for_single_thread = timeTaken;
        resultTable.addRow({to_string(NUMBER_OF_THREADS[i]), to_string(t.elapsed()), to_string(time_for_single_thread / timeTaken)});
    }

    // Print the result table
    resultTable.printTable();

    return 0;
}
