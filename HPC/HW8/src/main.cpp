/**
 * @file main.cpp
 * @author Anand Kamble (amk23j@fsu.edu)
 * @brief Solves a 2D heat equation numerically using MPI parallelization
 * @version 0.1
 * @date 2024-03-21
 *
 * @ref
 * https://mpi.deino.net/mpi_functions/MPI_Dims_create.html
 */

#include <mpi.h>
#include <cmath>
#include <iostream>
#include <fstream>

#define MASTER 0 // Rank of master task

void decompose1d(int n, int m, int i, int *s, int *e)
{
    const int length = n / m;
    const int deficit = n % m;
    *s = i * length + (i < deficit ? i : deficit);
    *e = *s + length - (i < deficit ? 0 : 1);
    if ((*e >= n) || (i == m - 1))
        *e = n - 1;
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    double wtime;
    int numProcesses, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);
    int dims[2], coords[2];
    int periods[2] = {0, 0};

    /**
     *  Starting the timer here since we want to also include the time taken
     *  by the MPI setup for creating the division of processes.
     */
    wtime = MPI_Wtime(); // Start the timer

    MPI_Comm comm2d;

    /**
     * Not initializing the dims array to 0 was causing an error.
     * This post from stackoverflow helped me fix the issue.
     * https://stackoverflow.com/questions/46698735/mpi-dims-create-throws-error-on-remote-machine
     */
    dims[0] = dims[1] = 0;                                         // Initialize dims array to 0
    MPI_Dims_create(numProcesses, 2, dims);                        // Create a division of processes
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &comm2d); // Create a 2D cartesian communicator
    MPI_Comm_rank(comm2d, &rank);                                  // Get the rank of the process in the new communicator

    int x0, x1, y0, y1;                             // Start and end indices of the grid
    int NX = 100, NY = 134;                         // Size of the grid
    MPI_Cart_get(comm2d, 2, dims, periods, coords); // Get the coordinates of the process in the new communicator
    decompose1d(NX, dims[0], coords[0], &x0, &x1);  // Decompose the grid into smaller grids
    decompose1d(NY, dims[1], coords[1], &y0, &y1);  // Decompose the grid into smaller grids

    // Debugging
    // printf("rank = %d, coords = %d %d, x0 = %d, x1 = %d, y0 = %d, y1 = %d\n", rank, coords[0], coords[1], x0, x1, y0, y1);
    // printf("dims = %d %d\n", dims[0], dims[1]);
    int left, right, down, up, nx, ny;           // Neighbors and size of the grid
    MPI_Cart_shift(comm2d, 0, 1, &left, &right); // Get the neighbors of the process
    MPI_Cart_shift(comm2d, 1, 1, &down, &up);    // Get the neighbors of the process

    if (left >= 0)
        x0--;
    if (right >= 0)
        x1++;
    if (down >= 0)
        y0--;
    if (up >= 0)
        y1++;
    nx = x1 - x0 + 1;
    ny = y1 - y0 + 1;

    double **u = new double *[nx]; // 2D array to store the values
    double *p = new double[nx * ny];
    for (int i = 0; i < nx; i++)
        u[i] = p + i * ny;

    double **u_new = new double *[nx]; // Allocate memory for the new array
    double *p_new = new double[nx * ny];
    for (int i = 0; i < nx; i++)
        u_new[i] = p_new + i * ny;

    MPI_Datatype xSlice, ySlice;                     // Datatypes for sending and receiving slices
    MPI_Type_vector(nx, 1, ny, MPI_DOUBLE, &xSlice); // Create a datatype for sending and receiving slices
    MPI_Type_commit(&xSlice);                        // Commit the datatype
    MPI_Type_vector(ny, 1, 1, MPI_DOUBLE, &ySlice);
    MPI_Type_commit(&ySlice);

    // Debugging
    // printf("rank = %d\n", rank);
    // printf("coords = %d %d, nx = %d, ny = %d\n", coords[0], coords[1], nx, ny);
    // printf("x0 = %d, x1 = %d, y0 = %d, y1 = %d\n", x0, x1, y0, y1);
    // printf("left = %d, right = %d, down = %d, up = %d\n", left, right, down, up);
    // printf("dims = %d %d\n", dims[0], dims[1]);
    // printf("========================\n");

    // Initialize the values to zero so that we can set the boundary conditions
    for (int i = 0; i < nx; i++)
    {
        for (int j = 0; j < ny; j++)
        {
            u[i][j] = 0.0;
        }
    }

    // For the left and right boundaries using the X coordinate
    if (coords[0] == 0)
    {
        printf("Setting left boundary\n");
        for (int j = 0; j < ny; j++)
        {
            u[0][j] = 1.0; // Left boundary
        }
    }

    if (coords[0] == dims[0] - 1)
    {
        for (int j = 0; j < ny; j++)
        {
            u[nx - 1][j] = 1.0; // Right boundary
        }
    }

    // For the top and bottom boundaries using the Y coordinate
    if (coords[1] == 0)
    {
        for (int i = 0; i < nx; i++)
        {
            u[i][0] = 0.0; // Bottom boundary
        }
    }

    if (coords[1] == dims[1] - 1)
    {
        for (int i = 0; i < nx; i++)
        {
            u[i][ny - 1] = 1.0; // Top boundary
        }
    }

    // Copy the initial values to u_new
    for (int i = 0; i < nx; ++i)
    {
        for (int j = 0; j < ny; ++j)
        {
            u_new[i][j] = u[i][j];
        }
    }

    const double dt = 0.0001;                    // Time step
    const double T = 0.0;                       // Total time
    const int nsteps = static_cast<int>(T / dt); // Number of time steps

    double dx = 3.0 / NX; // Grid spacing
    double dy = 4.0 / NY; // Grid spacing

    // Iterate
    for (int step = 0; step < nsteps; step++)
    {
        MPI_Sendrecv(&u[0][ny - 2], 1, xSlice, up, 123, &u[0][0], 1, xSlice,
                     down, 123, comm2d, MPI_STATUS_IGNORE);
        MPI_Sendrecv(&u[0][1], 1, xSlice, down, 123, &u[0][ny - 1], 1, xSlice,
                     up, 123, comm2d, MPI_STATUS_IGNORE);
        MPI_Sendrecv(&u[nx - 2][0], 1, ySlice, right, 123, &u[0][0], 1, ySlice,
                     left, 123, comm2d, MPI_STATUS_IGNORE);
        MPI_Sendrecv(&u[1][0], 1, ySlice, left, 123, &u[nx - 1][0], 1, ySlice,
                     right, 123, comm2d, MPI_STATUS_IGNORE);
        // Update u to new values by finite difference scheme
        for (int i = 1; i < nx - 1; i++)
        {
            for (int j = 1; j < ny - 1; j++)
            {
                // Ref: https://levelup.gitconnected.com/solving-2d-heat-equation-numerically-using-python-3334004aa01a
                u_new[i][j] = u[i][j] + dt * ((u[i + 1][j] - 2 * u[i][j] + u[i - 1][j]) / (dx * dx)) +
                              dt * ((u[i][j + 1] - 2 * u[i][j] + u[i][j - 1]) / (dy * dy));
            }
        }

        for (int i = 0; i < nx; ++i)
        {
            for (int j = 0; j < ny; ++j)
            {
                u[i][j] = u_new[i][j];
            }
        }
    }

    /**
     * Stopping the timer here so that the time to write the results
     * to a file is not included in the time taken to solve the equation.
     */
    if (rank == MASTER)
    {
        wtime = MPI_Wtime() - wtime;
        printf("Time taken: %3.6f\n", wtime);
    }

    // Output the solution to a text file
    std::ofstream outFile;
    std::string filename = "solution_" + std::to_string(rank) + ".txt";
    outFile.open(filename);
    for (int i = 0; i < nx; i++)
    {
        for (int j = 0; j < ny; j++)
        {
            outFile << u[i][j] << " ";
        }
        outFile << "\n";
    }
    outFile.close();

    // Free memory
    delete[] p;
    delete[] u;
    delete[] p_new;
    delete[] u_new;

    MPI_Type_free(&xSlice);
    MPI_Type_free(&ySlice);
    MPI_Finalize();
    return 0;
}