

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <iostream>
#include <cstring>
#ifndef __JpegFile
#define __JpegFile
#include "../includes/Jpegfile.h"
#endif
#include "utils.cpp"

#define MASTER 0 // Rank of master task
typedef long long int COUNT;

int main(int argc, char *argv[])
{

    const int k = 3;           // Number of clusters
    const int ITERATIONS = 10; // Number of iterations for the k-means algorithm

    int id, p;
    double wtime;

    MPI_Init(&argc, &argv);             // Initialize the MPI environment
    MPI_Comm_rank(MPI_COMM_WORLD, &id); // Get the rank of the process
    MPI_Comm_size(MPI_COMM_WORLD, &p);  // Get the total number of processes

    // Variables to store image properties and pixel data
    UINT height;
    UINT width;
    BYTE *dataBuf;

    // DIST i = 0;

    if (id == MASTER)
    {
        // Read the file to dataBuf with RGB format. The width and height of the image are stored in the respective variables.
        dataBuf = JpegFile::JpegFileToRGB("Images/test.jpg", &width, &height);
    }

    wtime = MPI_Wtime(); // Start the timer

    // Broadcast the image height and width to all processes
    // printf("Size of UInt: %d\n", sizeof(UINT));
    /*
        Running the above line of code will give the following output:
        Size of UInt: 4
        Which means that the size of UINT is 4 bytes or 32 bits.
        Therefore, I have selected MPI_UINT32_T as the data type for broadcasting the height and width of the image.
    */
    MPI_Bcast(&height, 1, MPI_UINT32_T, MASTER, MPI_COMM_WORLD);
    MPI_Bcast(&width, 1, MPI_UINT32_T, MASTER, MPI_COMM_WORLD);

    int workerRows = height / p;    // Number of rows to be processed by each worker process
    int extraRows = height % p;     // Remaining rows after dividing the rows equally among the processes
    int rowsToProcess = workerRows; // Number of rows to be processed by each process
    if (id == MASTER)
    {
        rowsToProcess += extraRows; // Master process will process the remaining rows
    }

    // Allocate memory for the data to be processed by each worker process
    int workerDataSize = rowsToProcess * width * 3 * sizeof(int);
    int *workerData = (int *)calloc(rowsToProcess * width * 3, sizeof(int));
    // Scatter the data to all processes
    auto scatterCount = (COUNT)(rowsToProcess * width * 3); // Number of elements to be scattered to each process
    MPI_Scatter(dataBuf, scatterCount, MPI_BYTE, workerData, scatterCount, MPI_BYTE, MASTER, MPI_COMM_WORLD);

    if (id == MASTER && extraRows > 0)
    {
        printf("Master process: %d\n", id);
        printf("rowsToProcess: %d\n", rowsToProcess);
        memcpy(workerData + (workerRows * width * 3),
               dataBuf + (workerRows * p * width * 3),
               extraRows * width * 3 * sizeof(BYTE));
    }

    auto generators = (int *)calloc(k * 3, sizeof(int));
    auto newGenerators = (int *)calloc(k * 3, sizeof(int));

    // Initializing the generators with the first k pixels.
    if (id == MASTER)
    {
        for (int i = 0; i < k; i++)
        {
            generators[i * 3] = dataBuf[(((height * width * 3) / k) * i)];
            generators[i * 3 + 1] = dataBuf[(((height * width * 3) / k) * i) + 1];
            generators[i * 3 + 2] = dataBuf[(((height * width * 3) / k) * i) + 2];
        }
    }

    // An array which will hold the index of the generator pixel for each pixel in the image.
    int *pixelGeneratorIndex = nullptr;
    int *workerPixelGeneratorIndex = (int *)(calloc(rowsToProcess * width, sizeof(int)));
    if (id == MASTER)
    {
        pixelGeneratorIndex = (int *)(calloc(height * width, sizeof(int)));
    }

    // Perform the k-means algorithm
    for (int iter = 0; iter < ITERATIONS; iter++)
    {
        // Broadcast the generators to all processes
        // Doing this inside the loop because the generators will be updated in each iteration
        MPI_Bcast(generators, k * 3, MPI_BYTE, MASTER, MPI_COMM_WORLD);

        for (int row = 0; row < rowsToProcess; row++)
        {
            for (int c = 0; c < width; c++)
            {
                DIST minDistance = INFINITY;
                int minIndex = 0;
                for (int m = 0; m < k; m++)
                {
                    DIST distance = distanceBetweenPexels((BYTE *)workerData, (row * width) + c, generators + m * 3);
                    if (distance < minDistance)
                    {
                        minDistance = distance;
                        minIndex = m;
                    }
                }

                // I have combined the step 2 and 3 of the k-means algorithm in the following lines of code.
                newGenerators[minIndex * 3] += (workerData[(row * width + c) * 3]) / (rowsToProcess * width);
                newGenerators[minIndex * 3 + 1] += (workerData[(row * width + c) * 3 + 1]) / (rowsToProcess * width);
                newGenerators[minIndex * 3 + 2] += (workerData[(row * width + c) * 3 + 2]) / (rowsToProcess * width);
                workerPixelGeneratorIndex[(row * width) + c] = minIndex; // Store the index of the generator pixel for each pixel in the image
            }
        }

        // Reduce the newGenerators to the master process
        /**
         * The reduction part is not working as expected. I have tried fixing it but I am unable to do so.
         * I guess I have made a mistake in the types of the variables.
         * I have tried changing the types of the variables but it is still not working.
         */
        MPI_Reduce(newGenerators, generators, k * 3, MPI_INT, MPI_SUM, MASTER, MPI_COMM_WORLD);
        MPI_Gather(workerPixelGeneratorIndex, rowsToProcess * width, MPI_INT, pixelGeneratorIndex, rowsToProcess * width, MPI_INT, MASTER, MPI_COMM_WORLD);
    }

    // Master process will write the segmented image to a file
    if (id == MASTER)
    {
        for (int row = 0; row < height; row++)
        {
            for (int col = 0; col < width; col++)
            {
                // Assign the color of the generator pixel to each pixel in the image
                dataBuf[(row * width + col) * 3] = generators[pixelGeneratorIndex[row * width + col] * 3];
                dataBuf[(row * width + col) * 3 + 1] = generators[pixelGeneratorIndex[row * width + col] * 3 + 1];
                dataBuf[(row * width + col) * 3 + 2] = generators[pixelGeneratorIndex[row * width + col] * 3 + 2];
            }
        }

        wtime = MPI_Wtime() - wtime;
        printf("Time taken: %3.6f\n", wtime);

        // Write the segmented image to a file
        JpegFile::RGBToJpegFile("test_seg.jpg", dataBuf, width, height, true, 100);
    }

    free(workerData);
    free(generators);
    free(newGenerators);
    free(pixelGeneratorIndex);
    free(workerPixelGeneratorIndex);

    MPI_Finalize();

    return 0;
}
