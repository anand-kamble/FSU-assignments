/**
 * @file ColorToGray.cpp
 * @brief This file contains the main function for converting a color image to grayscale using MPI.
 *
 * @author Student Name: Anand Kamble
 * @date 27th Feb 2024
 *
 * @note Make sure to include the necessary dependencies:
 *   - stdio.h
 *   - stdlib.h
 *   - mpi.h
 *   - "Jpegfile.h" // Assuming this header file is required for JPEG file manipulation
 *
 * @warning Ensure proper compilation with MPI support for parallel execution.
 */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "Jpegfile.h"

/* Defined the master rank this way so that the code becomes more readable
and the master rank can be changed easily if required.*/
#define MASTER 0 // Rank of master task

int main(int argc, char *argv[])
{

	int id, p;
	double wtime;
	MPI_Init(&argc, &argv);				// Initialize the MPI environment
	MPI_Comm_rank(MPI_COMM_WORLD, &id); // Get the rank of the process
	MPI_Comm_size(MPI_COMM_WORLD, &p);	// Get the total number of processes

	UINT height = 0;
	UINT width = 0;
	BYTE *dataBuf;

	if (id == MASTER) // Only the master rank reads the image file.
	{
		dataBuf = JpegFile::JpegFileToRGB("test-huge.jpg", &width, &height);
	}

	wtime = MPI_Wtime(); // Start the timer

	// Broadcast the image height and width to all processes
	MPI_Bcast(&height, 1, MPI_UINT16_T, MASTER, MPI_COMM_WORLD);
	MPI_Bcast(&width, 1, MPI_UINT16_T, MASTER, MPI_COMM_WORLD);

	auto workerRows = height / p; // Number of rows to be processed by each worker process
	int extraRows = height % p;	  // Remaining rows after dividing the rows equally among the processes

	auto rowsToProcess = workerRows; // Number of rows to be processed by each process
	if (id == MASTER)
	{
		rowsToProcess += extraRows; // Master process will process the remaining rows
	}

	// Allocate memory for the data to be processed by each worker process
	auto workerDataSize = (int)(rowsToProcess * width * 3 * sizeof(BYTE));
	auto workerData = (BYTE *)malloc(workerDataSize);

	// Scatter the data to all processes
	auto scatterCount = workerRows * width * 3; // Number of elements to be scattered to each process
	MPI_Scatter(dataBuf, scatterCount, MPI_BYTE, workerData, scatterCount, MPI_BYTE, MASTER, MPI_COMM_WORLD);

	// Process the data
	UINT row, col;
	for (row = 0; row < rowsToProcess; row++)
	{
		for (col = 0; col < width; col++)
		{
			BYTE *pRed, *pGrn, *pBlu;
			pRed = workerData + row * width * 3 + col * 3;
			pGrn = workerData + row * width * 3 + col * 3 + 1;
			pBlu = workerData + row * width * 3 + col * 3 + 2;

			// luminance
			int lum = (int)(.299 * (double)(*pRed) + .587 * (double)(*pGrn) + .114 * (double)(*pBlu));

			*pRed = (BYTE)lum;
			*pGrn = (BYTE)lum;
			*pBlu = (BYTE)lum;
		}
	}

	// Gather the processed data from all processes to the master process
	MPI_Gather(workerData, scatterCount, MPI_BYTE, dataBuf, scatterCount, MPI_BYTE, MASTER, MPI_COMM_WORLD);

	// Stop the timer
	wtime = MPI_Wtime() - wtime;
	if (id == MASTER)
	{
		printf("Time taken: %3.6f\n", wtime);
	}

	// Write the processed data to a new file by the master process
	if (id == MASTER) // Only the master rank writes the image file.
	{
		JpegFile::RGBToJpegFile("testmono.jpg", dataBuf, width, height, true, 75);
	}

	// Free the memory allocated for the worker data
	delete dataBuf;
	delete workerData;

	// Finalize the MPI environment
	MPI_Finalize();

	return 0;
}