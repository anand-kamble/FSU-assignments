#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

#define A(x, y) (A[(x * n) + y])
#define x(a, b) (x[(a) + 2 * b])

void Jacobi(int n, double *A, double *b, double *x)
{
    for (size_t i = 0; i < n; i++)
    {
        double temp = 0.0;
        for (size_t j = 0; j < n; j++)
            i != j ? temp += A(i, j) * x(0, j) : 0;
        x(0, i) = (1 / A(i, i)) * (b[i] - temp);
    }
}

void GaussSeidel(int n, double *A, double *b, double *x)
{
    for (int i = 0; i < n; i++)
    {
        double temp = 0.0;
        for (int j = 0; j < i; j++)
            temp = temp + (A(i, j) * x(1, j));

        for (int j = i + 1; j < n; j++)
            temp = temp + (A(i, j) * x(1, j));

        x(1, i) = (1 / A(i, i)) * (b[i] - temp);
    }
}

void initializeArrays(int n, double *A, double *B, double *x)
{
    memset(A, 0, n * n * sizeof(double));
    memset(B, 0, n * sizeof(B[0]));
    memset(x, 0, 2 * n * sizeof(x[0]));

    // Initializing matrix A with given conditions.
    for (size_t i = 0; i < n; i++)
    {
        A(i, i) = 2.0;
        if (i < (n - 1))
        {
            A(i, i + 1) = -1.0;
            A((i + 1), i) = -1.0;
        }
    }

    // Initializing matrix B
    B[0] = 1.0;
    B[n - 1] = 1.0;
}

int main()
{

    int n, niter;

    /* Read input from user and store it in the varible. */
    printf("Enter the value of 'n' : ");
    scanf("%d", &n);
    printf("Enter the value of 'niter' : ");
    scanf("%d", &niter);

    /* Allocating one block of memory for storing the matrix A.*/
    double *A = (double *)malloc((n * n) * sizeof(double));

    /* Allocating one block of memory for storing the matrix B.*/
    double *B = (double *)malloc((n) * sizeof(double));

    /* Allocating one block of memory for storing the matrix x.*/
    double *x = (double *)malloc((2 * n) * sizeof(double));

    /* Check if any of the pointer is a NULL, if so, deallocate memory and gracefully exit the program. */
    if (A == NULL || B == NULL || x == NULL)
    {
        free(A);
        free(B);
        free(x);
        printf("Failed to allocate memory.\n");
        exit(0);
    }

    initializeArrays(n, A, B, x);

    clock_t start_time, end_time;
    double cpu_time_used;

    // Record the start time
    start_time = clock();

    int errorAtIteration = 20;

    double err = 0.0;
    double l2Norm = 0.0;

    for (size_t i = 0; i < niter; i++)
    {
        if (i == errorAtIteration - 1)
        {

            double *xOld = (double *)malloc((2 * n) * sizeof(double));
            if (xOld == NULL)
            {
                printf("Failed to allocate memory before Jacobi.\n");
                exit(0);
            }
            memcpy(xOld, x, (2 * n) * sizeof(double));

            Jacobi(n, A, B, x);
            for (size_t i = 0; i < n; i++)
            {
                err += fabs(xOld[(0) + 2 * i] - x(0, i));
                l2Norm += fabs(xOld[(0) + 2 * i] - (0, i)) * fabs(xOld[(0) + 2 * i] - x(0, i));
            }
        }
        else
        {
            Jacobi(n, A, B, x);
        }
    }
    if (niter > errorAtIteration)
    {
        printf("--------------------------------------\n");
        printf("ERROR JACOBI at iteration %d =  %lf\n", errorAtIteration, err);
        printf("L2 norm JACOBI at iteration %d =  %lf\n", errorAtIteration, sqrt(l2Norm));
    }

    err = 0.0;
    l2Norm = 0.0;

    for (size_t i = 0; i < niter; i++)
    {
        if (i == errorAtIteration - 1)
        {

            double *xOld = (double *)malloc((2 * n) * sizeof(double));
            if (xOld == NULL)
            {
                printf("Failed to allocate memory before GaussSeidel.\n");
                exit(0);
            }
            memcpy(xOld, x, (2 * n) * sizeof(double));

            GaussSeidel(n, A, B, x);
            for (size_t i = 0; i < n; i++)
            {
                err += fabs(xOld[(1) + 2 * i] - x(1, i));
                l2Norm += fabs(xOld[(1) + 2 * i] - x(1, i)) * fabs(xOld[(1) + 2 * i] - x(1, i));
            }
        }
        else
        {
            GaussSeidel(n, A, B, x);
        }
    }
    if (niter > errorAtIteration)
    {
        printf("--------------------------------------\n");
        printf("ERROR GAUSS at iteration %d =  %lf\n", errorAtIteration, err);
        printf("L2 norm JACOBI at iteration %d =  %lf\n", errorAtIteration, sqrt(l2Norm));
    }

    // Record the end time
    end_time = clock();

    cpu_time_used = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;
    printf("--------------------------------------\n");
    printf("CPU time used: %f seconds\n", cpu_time_used);
    printf("--------------------------------------\n");
    printf("Solution : \n");
    printf("Jacobi    |  Gauss Seidel\n");
    printf("-------------------------\n");
    
    for (size_t i = 0; i < n; i++)
    {
        printf("%2.6lf  |  %2.6lf\n", x(0, i), x(1, i));
    }

    /* Releasing the memory blocks used by A,B,x */
    free(A);
    free(B);
    free(x);
    return 0;
}