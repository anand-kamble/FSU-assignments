#ifndef SECANT_H
#define SECANT_H

#include <iostream>
#include <cmath>
#include "solver.cpp"

using namespace std;

/**
 * @brief Class implementing the Secant method for finding roots of a function.
 * 
 * This class inherits from the Solver base class and provides an implementation
 * of the Secant method for finding roots of a target function.
 * 
 * @tparam T The data type for the variables and the result.
 */
template <typename T>
class Secant : public Solver<T>
{
public:
    /**
     * @brief Constructor for the Secant class.
     * 
     * @param targetFunction The target function for which the root needs to be found.
     */
    Secant(T (*targetFunction)(T))
    {
        this->targetFunction = targetFunction;
    };

    /**
     * @brief Compute the root of the target function using the Secant method.
     * 
     * Overrides the virtual method in the base class to provide the Secant method implementation.
     * 
     * @param initialGuess The initial guess for the root.
     * @param epsilon The desired level of accuracy.
     * @param maxIterations The maximum number of iterations allowed.
     * @return The computed root of the equation.
     */
    T ComputeRoot(T initialGuess, T epsilon, int maxIterations) override
    {
        T x0 = initialGuess;
        T x1 = initialGuess + 1;
        T x2;

        for (int i = 0; i < maxIterations; ++i)
        {
            x2 = x1 - (this->targetFunction(x1) * (x1 - x0)) / (this->targetFunction(x1) - this->targetFunction(x0));

            if (abs(this->targetFunction(x2)) < epsilon)
            {
                this->root = this->targetFunction(x2);
                cout << "[Secant] Root : " << this->root << ", Iterations : " << i + 1 << endl;
                return this->root;
            }

            x0 = x1;
            x1 = x2;
        }

        cerr << "Secant method did not converge within the specified number of iterations." << endl;
        return x0;
    }

    /**
     * @brief Verify the computed solution against the expected solution.
     * 
     * Overrides the virtual method in the base class to provide verification of the computed solution.
     * 
     * @param expected_solution The expected solution for verification.
     * @return The error between the expected and computed solutions.
     */
    T Verify(T expected_solution) override
    {
        auto error = abs(expected_solution - this->root);
        cout << "[Secant] Error: " << error << endl;
        return error;
    }
};

#endif
