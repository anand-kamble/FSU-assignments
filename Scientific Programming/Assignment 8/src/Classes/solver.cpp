#ifndef SOLVER_H
#define SOLVER_H

#include <iostream>

using namespace std;

/**
 * @brief Base class for solving equations using numerical methods.
 * 
 * This class provides a common interface for solving equations using numerical methods,
 * such as Newton's method. Derived classes must implement the ComputeRoot and Verify methods.
 * 
 * @tparam T The data type for the variables and the result.
 */
template <typename T>
class Solver
{
public:
    /**
     * @brief Target function to find the root for.
     * 
     * This function represents the equation for which the root needs to be found.
     * 
     * @param x The input variable.
     * @return The result of the target function at the given input.
     */
    T (*targetFunction)(T);

    /**
     * @brief Computed root of the equation.
     * 
     * The root is the solution found by the numerical method.
     */
    T root;

    /**
     * @brief Pure virtual function to compute the root of the equation.
     * 
     * Derived classes must implement this method to compute the root using a specific numerical method.
     * 
     * @param initialGuess The initial guess for the root.
     * @param epsilon The desired level of accuracy.
     * @param maxIterations The maximum number of iterations allowed.
     * @return The computed root of the equation.
     */
    virtual T ComputeRoot(T initialGuess, T epsilon, int maxIterations) = 0;

    /**
     * @brief Pure virtual function to verify the computed solution.
     * 
     * Derived classes must implement this method to verify the correctness of the computed solution.
     * 
     * @param expected_solution The expected solution for verification.
     * @return True if the computed solution is verified successfully, false otherwise.
     */
    virtual T Verify(T expected_solution) = 0;
};

#endif
