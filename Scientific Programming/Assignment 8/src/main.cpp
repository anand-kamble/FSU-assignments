#include <iostream>
#include <vector>
#include <cmath>
#include "Classes/newton.cpp"
#include "Classes/secant.cpp"
#include "Classes/Test.cpp"

using namespace std;

/**
 * @brief Function to solve a system of equations using Newton's method and the Secant method.
 * 
 * This function demonstrates solving a system of equations using Newton's method and the Secant method.
 * It defines three functions and their derivatives, sets tolerances, maximum iterations, and expected outputs,
 * and then uses Newton's method and the Secant method to find roots for each function with different precision.
 * It also performs verification tests and outputs the results.
 * 
 * @tparam T The data type for the variables and the result.
 * @return 0 on successful completion.
 */
template <typename T>
T Solve()
{
    // Define target functions and their derivatives
    vector<T (*)(T)> function = {
        [](T x) -> T
        { return sin(x); },
        [](T x) -> T
        { return pow(x, 3) - 6 * pow(x, 2) + 11 * x - 6; },
        [](T x) -> T
        { return log(x) + pow(x, 2) - 3; },
    };

    vector<T (*)(T)> derivatives = {
        [](T x) -> T
        { return cos(x); },
        [](T x) -> T
        { return pow(x, 2) * 3 - 12 * x + 11; },
        [](T x) -> T
        { return 2 * x + (1 / x); },
    };

    // Set tolerances, maximum iterations, and expected outputs
    vector<T> tolerance = {0.01, 0.001};
    vector<int> maxIteration = {100, 500};
    vector<T> ExpectedOutput = {0, 11, -3};

    vector<T> NewtonRoots;
    vector<T> SecantRoots;

    // Iterate through each function
    for (size_t i = 0; i < function.size(); i++)
    {
        cout << "Solving for Function : " << i << endl;
        cout << "Method | Root (x) | Iterations" << endl;

        // Create instances of Newton and Secant solvers
        Solver<T> *NewtonSolver = new Newton<T>(function[i], derivatives[i]);
        Solver<T> *SecantSolver = new Secant<T>(function[i]);

        // Iterate through each tolerance
        for (size_t i = 0; i < tolerance.size(); i++)
        {
            // Compute roots using Newton's method and the Secant method
            T NRoot = NewtonSolver->ComputeRoot(1.0, tolerance[i], maxIteration[i]);
            T SRoot = SecantSolver->ComputeRoot(1.0, tolerance[i], maxIteration[i]);
            
            // Verify the roots
            NewtonSolver->Verify(ExpectedOutput[i]);
            SecantSolver->Verify(ExpectedOutput[i]);

            // Store the roots
            NewtonRoots.push_back(NRoot);
            SecantRoots.push_back(SRoot);
        }
        cout << "-------------------------------------" << endl;
    }

    // Run verification tests
    (new Test("Testing if f(x) = 0 for Function 1."))
        ->Expect(function[0](NewtonRoots[0])).ToBe(0);
    (new Test("Testing if f(x) = 0 for Function 2."))
        ->Expect(function[1](NewtonRoots[1])).ToBe(0);
    (new Test("Testing if f(x) = 0 for Function 3."))
        ->Expect(function[2](NewtonRoots[2])).ToBe(0);

    return 0;
}

/**
 * @brief Main function.
 * This is the main function that calls the Solve function for both float and double precision.
 * @return 0 on successful completion.
 */
int main()
{
    cout << "\n\n====== Calculating with float precision. ======" << endl;
    Solve<float>();

    cout << "\n\n====== Calculating with double precision. ======" << endl;
    Solve<double>();

    return 0;
}
